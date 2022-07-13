import numpy as np
import numpy.ma as ma

import sklearn.cluster as cluster
import sklearn.decomposition as dcomp
from skimage.measure import LineModelND, ransac

from h5flow.core import H5FlowStage, resources


class TrackletReconstruction(H5FlowStage):
    '''
        Reconstructs "tracklets" or short, collinear track segments from hit
        data using DBSCAN and RANSAC. The track direction is estimated using
        a PCA fit.

        Parameters:
         - ``tracklet_dset_name``: ``str``, path to output dataset
         - ``hits_dset_name``: ``str``, path to input charge hits dataset
         - ``hit_drift_dset_name``: ``str``, path to charge hits drift data
         - ``dbscan_eps``: ``float``, dbscan epsilon parameter
         - ``dbscan_min_samples``: ``int``, dbscan min neighbor points to consider as "core" point
         - ``ransac_min_samples``: ``int``, min points to run ransac algorithm
         - ``ransac_residual_threshold``: ``float``, max distance from trial axis
         - ``ransac_max_trials``: ``int``, number of ransac trials per cluster
         - ``max_iterations``: ``int``, max number of fitting iterations before giving up
         - ``max_nhit``: ``int``, skip track fitting on events with greater number of hits, ``None`` to apply no cut

        Both ``hits_dset_name`` and ``hits_drift_dset_name`` are required in the cache.

        Requires Geometry, RunData, and Units resource in workflow.

        ``tracklets`` datatype::

            id          u4,     unique identifier
            theta       f8,     track inclination w.r.t anode
            phi         f8,     track orientation w.r.t anode
            xp          f8,     intersection of track with ``x=0,y=0`` plane [mm]
            yp          f8,     intersection of track with ``x=0,y=0`` plane [mm]
            nhit        i8,     number of hits in track
            q           f8,     charge sum [mV]
            ts_start    f8,     PPS timestamp of track start [crs ticks]
            ts_end      f8,     PPS timestamp of track end [crs ticks]
            residual    f8(3,)  average track fit error in (x,y,z) [mm]
            length      f8      track length [mm]
            start       f8(3,)  track start point (x,y,z) [mm]
            end         f8(3,)  track end point (x,y,z) [mm]
            trajectory          f8(trajectory_pts, 3,)      track approximation points (x,y,z) [mm]
            trajectory_residual f8(trajectory_pts-1,)       track approximation average error [mm]
            dx                  f8(trajectory_pts-1, 3)     track approximation displacement (dx,dy,dz) [mm]
            dq                  f8(trajectory_pts-1,)       charge along track displacement [mV]
            dn                  i8(trajectory_pts-1,)       nhit along track displacement

    '''
    class_version = '1.0.0'

    default_tracklet_dset_name = 'combined/tracklets'
    default_hits_dset_name = 'charge/hits'
    default_hit_drift_dset_name = 'combined/hit_drift'

    default_dbscan_eps = 25
    default_dbscan_min_samples = 5
    default_ransac_min_samples = 2
    default_ransac_residual_threshold = 8
    default_ransac_max_trials = 100
    default_max_iterations = 100
    default_trajectory_pts = 5
    default_trajectory_dx = 10
    default_max_nhit = 3000
    default_trajectory_residual_mode = 1

    @staticmethod
    def tracklet_dtype(npts=default_trajectory_pts):
        return np.dtype([
            ('id', 'u4'),
            ('theta', 'f8'), ('phi', 'f8'),
            ('xp', 'f8'), ('yp', 'f8'),
            ('nhit', 'i8'), ('q', 'f8'),
            ('ts_start', 'f8'), ('ts_end', 'f8'),
            ('residual', 'f8', (3,)), ('length', 'f8'),
            ('start', 'f8', (3,)), ('end', 'f8', (3,)),
            ('trajectory', 'f8', (npts, 3)),
            ('trajectory_residual', 'f8', (npts - 1,)),
            ('dx', 'f8', (npts - 1, 3)),
            ('dq', 'f8', (npts - 1,)),
            ('dn', 'i8', (npts - 1,))
        ])

    def __init__(self, **params):
        super(TrackletReconstruction, self).__init__(**params)

        self.tracklet_dset_name = params.get('tracklet_dset_name', self.default_tracklet_dset_name)
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.hit_drift_dset_name = params.get('hit_drift_dset_name', self.default_hit_drift_dset_name)

        self._dbscan_eps = params.get('dbscan_eps', self.default_dbscan_eps)
        self._dbscan_min_samples = params.get('dbscan_min_samples', self.default_dbscan_min_samples)
        self._ransac_min_samples = params.get('ransac_min_samples', self.default_ransac_min_samples)
        self._ransac_residual_threshold = params.get('ransac_residual_threshold', self.default_ransac_residual_threshold)
        self._ransac_max_trials = params.get('ransac_max_trials', self.default_ransac_max_trials)
        self.max_iterations = params.get('max_iterations', self.default_max_iterations)
        self.max_nhit = params.get('max_nhit', self.default_max_nhit)

        self.trajectory_residual_mode = params.get('trajectory_residual_mode', self.default_trajectory_residual_mode)
        self.trajectory_pts = params.get('trajectory_pts', self.default_trajectory_pts)
        self.trajectory_dx = params.get('trajectory_dx', self.default_trajectory_dx)
        self.tracklet_dtype = self.tracklet_dtype(self.trajectory_pts)

        self.dbscan = cluster.DBSCAN(eps=self._dbscan_eps, min_samples=self._dbscan_min_samples)

    def init(self, source_name):
        super(TrackletReconstruction, self).init(source_name)

        self.data_manager.set_attrs(self.tracklet_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    hits_dset=self.hits_dset_name,
                                    hit_drift_dset=self.hit_drift_dset_name,
                                    dbscan_eps=self._dbscan_eps,
                                    dbscan_min_samples=self._dbscan_min_samples,
                                    ransac_min_samples=self._ransac_min_samples,
                                    ransac_residual_threshold=self._ransac_residual_threshold,
                                    ransac_max_trials=self._ransac_max_trials,
                                    max_iterations=self.max_iterations,
                                    max_nhit=self.max_nhit,
                                    trajectory_pts=self.trajectory_pts,
                                    trajectory_dx=self.trajectory_dx
                                    )

        self.data_manager.create_dset(self.tracklet_dset_name, self.tracklet_dtype)
        self.data_manager.create_ref(self.tracklet_dset_name, self.hits_dset_name)
        self.data_manager.create_ref(source_name, self.tracklet_dset_name)

    def run(self, source_name, source_slice, cache):
        super(TrackletReconstruction, self).run(source_name, source_slice, cache)

        events = cache[source_name]                         # shape: (N,)
        hits = cache[self.hits_dset_name]                   # shape: (N,M)
        hit_drift = cache[self.hit_drift_dset_name]         # shape: (N,M,1)
        hit_drift = hit_drift.reshape(hits.shape)
        if self.max_nhit is not None:
            hits = ma.array(hits, mask=(events['nhit'][..., np.newaxis] > self.max_nhit) | hits['id'].mask,
                            shrink=False)
            hit_drift = ma.array(hit_drift, mask=(events['nhit'][..., np.newaxis] > self.max_nhit) | hits['id'].mask,
                                 shrink=False)

        track_ids = self.find_tracks(hits, hit_drift['z'])
        tracks = self.calc_tracks(hits, hit_drift['z'], track_ids, self.trajectory_pts,
                                  self.trajectory_dx, self.trajectory_residual_mode)
        n_tracks = np.count_nonzero(~tracks['id'].mask)
        tracks_mask = ~tracks['id'].mask

        tracks_slice = self.data_manager.reserve_data(self.tracklet_dset_name, n_tracks)
        np.place(tracks['id'], tracks_mask, np.r_[tracks_slice].astype('u4'))
        self.data_manager.write_data(self.tracklet_dset_name, tracks_slice, tracks[tracks_mask])

        # track -> hit ref
        track_ref_id = np.take_along_axis(tracks['id'], track_ids, axis=-1)
        mask = (~track_ref_id.mask) & (track_ids != -1) & (~hits['id'].mask)
        ref = np.c_[track_ref_id[mask], hits['id'][mask]]
        self.data_manager.write_ref(self.tracklet_dset_name, self.hits_dset_name, ref)

        # event -> track ref
        ev_id = np.broadcast_to(np.expand_dims(np.r_[source_slice], axis=-1), tracks.shape)
        ref = np.c_[ev_id[tracks_mask], tracks['id'][tracks_mask]]
        self.data_manager.write_ref(source_name, self.tracklet_dset_name, ref)

    @staticmethod
    def hit_xyz(hits, hit_z):
        xyz = np.concatenate((
            np.expand_dims(hits['px'], axis=-1),
            np.expand_dims(hits['py'], axis=-1),
            np.expand_dims(hit_z, axis=-1),
        ), axis=-1)
        return xyz

    def find_tracks(self, hits, hit_z):
        '''
            Extract tracks from a given hits array

            :param hits: masked array ``shape: (N, n)``

            :param hit_z: masked array ``shape: (N, n)``

            :returns: mask array ``shape: (N, n)`` of track ids for each hit, a value of -1 means no track is associated with the hit
        '''
        xyz = self.hit_xyz(hits, hit_z)

        iter_mask = np.ones(hits.shape, dtype=bool)
        iter_mask = iter_mask & (~hits['id'].mask)
        track_id = np.full(hits.shape, -1, dtype='i8')
        for i in range(hits.shape[0]):

            if not np.any(iter_mask[i]):
                continue

            current_track_id = -1

            for _ in range(self.max_iterations):
                # dbscan to find clusters
                track_ids = self._do_dbscan(xyz[i], iter_mask[i])

                for id_ in np.unique(track_ids):
                    if id_ == -1:
                        continue
                    mask = track_ids == id_
                    if np.sum(mask) <= self._ransac_min_samples:
                        continue

                    # ransac for collinear hits
                    inliers = self._do_ransac(xyz[i], mask)
                    mask[mask] = inliers

                    if np.sum(mask) < 1:
                        continue

                    # and a final dbscan for re-clustering
                    final_track_ids = self._do_dbscan(xyz[i], mask)

                    for id_ in np.unique(final_track_ids):
                        if id_ == -1:
                            continue
                        mask = final_track_ids == id_

                        current_track_id += 1
                        track_id[i, mask] = current_track_id
                        iter_mask[i, mask] = False

                if np.all(track_ids == -1) or not np.any(iter_mask[i]):
                    break

        return ma.array(track_id, mask=hits['id'].mask, shrink=False)

    @classmethod
    def calc_tracks(cls, hits, hit_z, track_ids, trajectory_pts, trajectory_dx, trajectory_residual_mode):
        '''
            Calculate track parameters from hits

            :param hits: masked array, ``shape: (N,M)``

            :param hit_z: masked array, ``shape: (N,M)``

            :param track_ids: masked array, ``shape: (N,M)``

            :param trajectory_pts: int

            :param trajectory_dx: float

            :returns: masked array, ``shape: (N,m)``
        '''
        xyz = cls.hit_xyz(hits, hit_z)

        n_tracks = np.clip(track_ids.max() + 1, 1, np.inf).astype(int) if np.count_nonzero(~track_ids.mask) \
            else 1
        tracks = np.empty((len(hits), n_tracks), dtype=cls.tracklet_dtype(trajectory_pts))
        tracks_mask = np.ones(tracks.shape, dtype=bool)
        for i in range(tracks.shape[0]):
            for j in range(tracks.shape[1]):
                mask = ((track_ids[i] == j) & (~track_ids.mask[i])
                        & (~hits['id'].mask[i]))
                if np.count_nonzero(mask) < 2:
                    continue

                # PCA on central hits
                centroid, axis = cls.do_pca(xyz[i], mask)
                r_min, r_max = cls.projected_limits(
                    centroid, axis, xyz[i][mask])
                residual = cls.track_residual(centroid, axis, xyz[i][mask])
                xyp = cls.xyp(axis, centroid)

                # run trajectory approximation algo
                traj = cls.trajectory_approx(centroid, axis, xyz[i][mask], trajectory_residual_mode,
                                             npts=trajectory_pts, dx=trajectory_dx, 
                                             weights=hits[i][mask]['q'])  # (npts, 3)
                d = cls.trajectory_residual(xyz[i][mask], traj, trajectory_residual_mode)  # (npts-1, N)
                min_edge_mask = np.indices(d.shape)[0] != np.expand_dims(np.argmin(d, axis=0), 0)  # (npts-1, N)
                edge_q = ma.sum(ma.array(
                    np.broadcast_to(hits[i][mask]['q'][np.newaxis, :],
                                    min_edge_mask.shape),
                    mask=min_edge_mask, shrink=False), axis=-1)  # (npts-1,)
                edge_res = ma.mean(ma.array(d, mask=min_edge_mask,
                                            shrink=False), axis=-1)  # (npts-1,)

                tracks[i, j]['theta'] = cls.theta(axis)
                tracks[i, j]['phi'] = cls.phi(axis)
                tracks[i, j]['xp'] = xyp[0]
                tracks[i, j]['yp'] = xyp[1]
                tracks[i, j]['nhit'] = np.count_nonzero(mask)
                tracks[i, j]['q'] = np.sum(hits[i][mask]['q'])
                tracks[i, j]['ts_start'] = np.min(hits[i][mask]['ts'])
                tracks[i, j]['ts_end'] = np.max(hits[i][mask]['ts'])
                tracks[i, j]['residual'] = residual
                tracks[i, j]['length'] = np.linalg.norm(r_max - r_min)
                tracks[i, j]['start'] = r_min
                tracks[i, j]['end'] = r_max

                tracks[i, j]['trajectory'] = traj
                tracks[i, j]['trajectory_residual'] = edge_res
                tracks[i, j]['dx'] = np.diff(traj, axis=0)
                tracks[i, j]['dq'] = edge_q
                tracks[i, j]['dn'] = np.sum(~min_edge_mask, axis=-1)

                tracks_mask[i, j] = False

        return ma.array(tracks, mask=tracks_mask, shrink=False)

    def _do_dbscan(self, xyz, mask):
        '''
            :param xyz: ``shape: (N,3)`` array of precomputed 3D distances

            :param mask: ``shape: (N,)`` boolean array of valid positions (``True == valid``)

            :returns: ``shape: (N,)`` array of grouped track ids
        '''
        clustering = self.dbscan.fit(xyz[mask])
        track_ids = np.full(len(mask), -1)
        track_ids[mask] = clustering.labels_
        return track_ids

    def _do_ransac(self, xyz, mask):
        '''
            :param xyz: ``shape: (N,3)`` array of 3D positions

            :param mask: ``shape: (N,)`` boolean array of valid positions (``True == valid``)

            :returns: ``shape: (N,)`` boolean array of colinear positions
        '''
        model_robust, inliers = ransac(xyz[mask], LineModelND,
                                       min_samples=self._ransac_min_samples,
                                       residual_threshold=self._ransac_residual_threshold,
                                       max_trials=self._ransac_max_trials)
        return inliers

    @staticmethod
    def trajectory_approx(centroid, axis, xyz, mode, npts, dx, weights=None):
        '''
            :param centroid: ``shape: (3,)`` pre-calculated centroid of 3D positions

            :param axis: ``shape: (3,)`` pre-calculated PCA of 3D positions

            :param xyz: ``shape: (N, 3)`` array of 3D positions

            :returns: ``shape: (npts, 3)`` array of piecewise-linear approximation
        '''
        # project hits onto PCA axis
        s = np.sum((xyz - centroid[np.newaxis, :]) * axis[np.newaxis, :],
                   axis=-1, keepdims=True)  # (N, 1)
      
        traj = np.empty((npts, 3))  # (M, 3)
        traj_s = np.empty((npts, 1))  # (M, 1)

        start_pt = np.argmin(s, axis=0)
        end_pt = np.argmax(s, axis=0)
 
        traj[0] = TrackletReconstruction.local_mean(xyz, xyz[start_pt], dx, weights=weights)
        traj[1:] = TrackletReconstruction.local_mean(xyz, xyz[end_pt], dx, weights=weights)
        traj_s[0] = s[start_pt]
        traj_s[1:] = s[end_pt]

        for i in range(1, npts - 1):
            # calculate residuals
            d = TrackletReconstruction.trajectory_residual(xyz, traj, mode)  # (M, N)

            # use smallest residual per point
            i_res_min = np.expand_dims(np.argmin(d, axis=0), axis=0)  # (1, N)
            res = np.take_along_axis(d, i_res_min, axis=0)  # (1, N)
            node_d = np.take_along_axis(d, i_res_min, axis=0)  # (1, N)

            # find farthest point
            mask = node_d < dx  # (1, N)
            # important for short tracks
            # the mask is to prevent breaking track segments into pieces smaller than trajectory_dx
            if mask.all() == True:
                break
            max_pt = ma.argmax(ma.array(res, mask=mask, shrink=False), axis=1)  # (1,)

            # update trajectory
            new_pt = TrackletReconstruction.local_mean(xyz, xyz[max_pt].ravel(), dx, weights=weights)  # (3,)
            new_s = np.sum((new_pt - centroid) * axis, axis=-1)  # (1,)
            traj[i] = new_pt
            traj_s[i] = new_s
            
            order = np.argsort(traj_s, axis=0)  # (M, 1)
            traj[:] = np.take_along_axis(traj, order, axis=0)
            traj_s[:] = np.take_along_axis(traj_s, order, axis=0)

        return traj

    @staticmethod
    def local_mean(xyz, pt, dx, weights=None):
        '''
            :param xyz: ``shape: (N, 3)``

            :param pt: ``shape: (3,)``

            :param dx: ``float`` radius to include in mean

            :param weights: ``shape: (N,)`` relative weights for each pt, ``None`` applies same weights

            :returns: ``shape: (M, 3)``
        '''
        # calculate local mean
        r = xyz - np.expand_dims(pt, axis=0)  # (N,3) - (1,3)
        d = np.linalg.norm(r, axis=-1, keepdims=True)  # (N,1)

        mask = np.broadcast_to(d > dx, r.shape)  # (N,3)
        traj = ma.average(ma.array(np.expand_dims(xyz, axis=1), mask=mask, shrink=False),
                          axis=0, weights=weights)  # (3,)
        return traj

    @staticmethod
    def do_pca(xyz, mask):
        '''
            :param xyz: ``shape: (N,3)`` array of 3D positions

            :param mask: ``shape: (N,)`` boolean array of valid positions (``True == valid``)

            :returns: ``tuple`` of ``shape: (3,)``, ``shape: (3,)`` of centroid and central axis
        '''
        centroid = np.mean(xyz[mask], axis=0)
        pca = dcomp.PCA(n_components=1).fit(xyz[mask] - centroid)
        axis = pca.components_[0] / np.linalg.norm(pca.components_[0])

        # break degenerate pca axis direction by fixing y component to be negative
        if axis[1] > 0:
            axis = -axis
        return centroid, axis

    @staticmethod
    def projected_limits(centroid, axis, xyz):
        s = np.dot((xyz - centroid), axis)
        xyz_min, xyz_max = np.amin(xyz, axis=0), np.amax(xyz, axis=0)
        r_max = np.clip(centroid + axis * np.max(s), xyz_min, xyz_max)
        r_min = np.clip(centroid + axis * np.min(s), xyz_min, xyz_max)
        return r_min, r_max

    @staticmethod
    def track_residual(centroid, axis, xyz):
        s = np.dot((xyz - centroid), axis)
        res = np.abs(xyz - (centroid + np.outer(s, axis)))
        return np.mean(res, axis=0)

    @staticmethod
    # mode = 1, shortest distance to the segment ends
    # mode = 2, shortest distance to the tractory
    def trajectory_residual(xyz, traj, mode):
        '''
            :param xyz: ``shape: (N, 3)``, 3D positions

            :param traj: ``shape: (npts, 3)```, trajectory 3D positions

            :returns: distance to nearest trajectory edge ``shape: (npts-1, N)``
        '''
        d0 = np.expand_dims(xyz, axis=0) - np.expand_dims(traj[:-1], axis=1)  # (1, N, 3) - (npts-1, 1, 3)
        d1 = np.expand_dims(xyz, axis=0) - np.expand_dims(traj[1:], axis=1)
        if mode == 1:
            dt = np.minimum(np.linalg.norm(d0, axis=-1), np.linalg.norm(d1, axis=-1))
        elif mode == 2:
            d = np.expand_dims(np.diff(traj, axis=0), axis=1)  # (npts-1, 1, 3)
            with np.errstate(divide='ignore', invalid='ignore'):
                n = d / np.linalg.norm(d, axis=-1, keepdims=True)        
            n[np.isnan(n) | np.isinf(n)] = 0

            dl = np.linalg.norm(d0 * n, axis=-1) # (npts-1, N, 1)
            dt = d0 - np.expand_dims(dl, -1) * n  # (npts-1, N, 3) - (npts-1, N, 1) * (1, 1, 3)
            dt = np.linalg.norm(dt, axis=-1)  # (npts-1, N)

            non_overlap_mask = (dl < 0) | (dl > np.linalg.norm(d, axis=-1))      
            dt[non_overlap_mask] = np.minimum(np.linalg.norm(d0, axis=-1),
                                              np.linalg.norm(d1, axis=-1))[non_overlap_mask]

        return dt

    @staticmethod
    def theta(axis):
        '''
            :param axis: array, ``shape: (3,)``

            :returns: angle of axis w.r.t z-axis
        '''
        return np.arctan2(np.linalg.norm(axis[:2]), axis[-1])

    @staticmethod
    def phi(axis):
        '''
            :param axis: array, ``shape: (3,)``

            :returns: orientation of axis about z-axis
        '''
        return np.arctan2(axis[1], axis[0])

    @staticmethod
    def xyp(axis, centroid):
        '''
            :param axis: array, ``shape: (3,)``

            :param centroid: array, ``shape: (3,)``

            :returns: x,y coordinate where line intersects ``x=0,y=0`` plane
        '''
        if axis[-1] == 0:
            return centroid[:2]
        s = -centroid[-1] / axis[-1]
        return (centroid + axis * s)[:2]
