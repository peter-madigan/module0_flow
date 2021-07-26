import numpy as np
import numpy.ma as ma
from collections import defaultdict
import logging

import sklearn.cluster as cluster
import sklearn.decomposition as dcomp
from skimage.measure import LineModelND, ransac

from h5flow.core import H5FlowStage, resources

class TrackletReconstruction(H5FlowStage):

    default_tracklet_dset_name = 'combined/tracklets'
    default_hits_dset_name = 'charge/hits'
    default_t0_dset_name = 'combined/t0'

    default_dbscan_eps = 2.5
    default_dbscan_min_samples = 5
    default_ransac_min_samples = 2
    default_ransac_residual_threshold = 8
    default_ransac_max_trials = 100

    tracklet_dtype = np.dtype([
            ('id', 'u4'),
            ('theta', 'f8'), ('phi', 'f8'),
            ('xp', 'f8'), ('yp', 'f8'),
            ('nhit', 'i8'), ('q', 'f8'),
            ('ts_start', 'i8'), ('ts_end', 'i8'),
            ('residual', 'f8', (3,)), ('length', 'f8'),
            ('start', 'f8', (4,)), ('end', 'f8', (4,))
        ])

    def __init__(self, **params):
        super(TrackletReconstruction,self).__init__(**params)

        self.tracklet_dset_name = params.get('tracklet_dset_name',self.default_tracklet_dset_name)
        self.hits_dset_name = params.get('hits_dset_name',self.default_hits_dset_name)
        self.t0_dset_name = params.get('t0_dset_name',self.default_t0_dset_name)

        self._dbscan_eps = params.get('dbscan_eps', self.default_dbscan_eps)
        self._dbscan_min_samples = params.get('dbscan_min_samples', self.default_dbscan_min_samples)
        self._ransac_min_samples = params.get('ransac_min_samples', self.default_ransac_min_samples)
        self._ransac_residual_threshold = params.get('ransac_residual_threshold', self.default_ransac_residual_threshold)
        self._ransac_max_trials = params.get('ransac_max_trials', self.default_ransac_max_trials)

        self.pca = dcomp.PCA(n_components=1)
        self.dbscan = cluster.DBSCAN(eps=self._dbscan_eps, min_samples=self._dbscan_min_samples)

    def init(self, source_name):
        self.data_manager.set_attrs(self.tracklet_dset_name,
            classname=self.classname,
            class_version=self.class_version,
            hits_dset=self.hits_dset_name,
            t0_dset=self.t0_dset_name,
            dbscan_eps=self._dbscan_eps,
            dbscan_min_samples=self._dbscan_min_samples,
            ransac_min_samples=self._ransac_min_samples,
            ransac_residual_threshold=self._ransac_residual_threshold,
            ransac_max_trials=self._ransac_max_trials
            )

        self.data_manager.create_dset(self.tracklet_dset_name, self.tracklet_dtype)
        self.data_manager.create_ref(self.tracklet_dset_name, self.hits_dset_name)
        self.data_manager.create_ref(source_name, self.tracklet_dset_name)

    def run(self, source_name, source_slice, cache):
        events = cache[source_name]                     # shape: (N,)
        t0 = cache[self.t0_dset_name]                   # shape: (N,1)
        hits = cache[self.hits_dset_name]               # shape: (N,M)
        hit_idx = cache[self.hits_dset_name+'_index']   # shape: (N,M)

        track_ids = self.find_tracks(hits, t0)
        tracks = self.calc_tracks(hits, t0, track_ids)
        n_tracks = np.count_nonzero(~tracks['id'].mask)
        tracks_mask = ~tracks['id'].mask

        tracks_slice = self.data_manager.reserve_data(self.tracklet_dset_name, n_tracks)
        np.place(tracks['id'], tracks_mask, np.r_[tracks_slice].astype('u4'))
        self.data_manager.write_data(self.tracklet_dset_name, tracks_slice, tracks[tracks_mask])

        # track -> hit ref
        track_ref_id = np.take_along_axis(tracks['id'], track_ids, axis=-1)
        mask = (~track_ref_id.mask) & (track_ids != -1)
        ref = np.c_[track_ref_id[mask], hit_idx[mask]]
        self.data_manager.write_ref(self.tracklet_dset_name, self.hits_dset_name, ref)

        # event -> track ref
        ev_id = np.broadcast_to(np.expand_dims(np.r_[source_slice], axis=-1), tracks.shape)
        ref = np.c_[ev_id[tracks_mask], tracks['id'][tracks_mask]]
        self.data_manager.write_ref(source_name, self.tracklet_dset_name, ref)

    def _hit_xyz(self, hits, t0):
        drift_t = hits['ts'] - t0['ts']
        drift_d = drift_t * (resources['LArData'].v_drift * resources['RunData'].crs_ticks)

        z = resources['Geometry'].get_z_coordinate(hits['iogroup'], hits['iochannel'], drift_d)

        xyz = np.concatenate((
            np.expand_dims(hits['px'], axis=-1),
            np.expand_dims(hits['py'], axis=-1),
            np.expand_dims(z, axis=-1),
            ), axis=-1)
        return xyz

    def find_tracks(self, hits, t0):
        '''
            Extract tracks from a given hits array

            :param hits: masked array ``shape: (N, n)``

            :param t0: masked array ``shape: (N, 1)``

            :returns: mask array ``shape: (N, n)`` of track ids for each hit, a value of -1 means no track is associated with the hit
        '''
        xyz = self._hit_xyz(hits, t0)

        iter_mask = np.ones(hits.shape, dtype=bool)
        iter_mask = iter_mask & (~hits['id'].mask)
        track_id = np.full(hits.shape, -1, dtype='i8')
        for i in range(hits.shape[0]):

            current_track_id = -1
            while True:
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

                    if np.sum(mask) < 2:
                        continue

                    current_track_id += 1
                    track_id[i, mask] = current_track_id
                    iter_mask[i, mask] = False

                if np.all(track_ids == -1) or not np.any(iter_mask[i]):
                    break

        return ma.array(track_id, mask=hits['id'].mask)

    def calc_tracks(self, hits, t0, track_ids):
        xyz = self._hit_xyz(hits, t0)

        n_tracks = track_ids.max() + 1 if np.count_nonzero(~track_ids.mask) \
            else 1
        tracks = np.empty((len(t0), n_tracks), dtype=self.tracklet_dtype)
        tracks_mask = np.ones(tracks.shape, dtype=bool)
        for i in range(tracks.shape[0]):
            for j in range(tracks.shape[1]):
                mask = (track_ids[i] == j) & (~track_ids.mask[i])
                if np.count_nonzero(mask) < 2:
                    continue

                # PCA on central hits
                centroid, axis = self._do_pca(xyz[i], mask)
                r_min, r_max = self._projected_limits(
                    centroid, axis, xyz[i][mask])
                residual = self._track_residual(centroid, axis, xyz[i][mask])
                xyp = self.xyp(axis, centroid)

                tracks[i,j]['theta'] = self.theta(axis)
                tracks[i,j]['phi'] = self.phi(axis)
                tracks[i,j]['xp'] = xyp[0]
                tracks[i,j]['yp'] = xyp[1]
                tracks[i,j]['nhit'] = np.count_nonzero(mask)
                tracks[i,j]['q'] = np.sum(hits[i][mask]['q'])
                tracks[i,j]['ts_start'] = np.min(hits[i][mask]['ts'])
                tracks[i,j]['ts_end'] = np.max(hits[i][mask]['ts'])
                tracks[i,j]['residual'] = residual
                tracks[i,j]['length'] = np.linalg.norm(r_max-r_min)
                tracks[i,j]['start'] = np.append(r_min, tracks[i,j]['ts_start']-t0[i]['ts'])
                tracks[i,j]['end'] = np.append(r_min, tracks[i,j]['ts_end']-t0[i]['ts'])

                tracks_mask[i,j] = False

        return ma.array(tracks, mask=tracks_mask)

    def _do_dbscan(self, xyz, mask):
        '''
            :param xyz: ``shape: (N,3)`` array of 3D positions

            :param mask: ``shape: (N,)`` boolean array of valid positions (``True == valid``)

            :returns: ``shape: (N,)`` array of grouped track ids
        '''
        clustering = self.dbscan.fit(xyz[mask])
        track_ids = np.zeros(len(mask))-1
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

    def _do_pca(self, xyz, mask):
        '''
            :param xyz: ``shape: (N,3)`` array of 3D positions

            :param mask: ``shape: (N,)`` boolean array of valid positions (``True == valid``)

            :returns: ``tuple`` of ``shape: (3,)``, ``shape: (3,)`` of centroid and central axis
        '''
        centroid = np.mean(xyz[mask], axis=0)
        pca = self.pca.fit(xyz[mask] - centroid)
        axis = pca.components_[0] / np.linalg.norm(pca.components_[0])
        return centroid, axis

    def _projected_limits(self, centroid, axis, xyz):
        s = np.dot((xyz - centroid), axis)
        xyz_min, xyz_max = np.amin(xyz, axis=0), np.amax(xyz, axis=0)
        r_max = np.clip(centroid + axis * np.max(s), xyz_min, xyz_max)
        r_min = np.clip(centroid + axis * np.min(s), xyz_min, xyz_max)
        return r_min, r_max

    def _track_residual(self, centroid, axis, xyz):
        s = np.dot((xyz - centroid), axis)
        res = np.abs(xyz - (centroid + np.outer(s, axis)))
        return np.mean(res, axis=0)

    def theta(self, axis):
        return np.arctan2(np.linalg.norm(axis[:2]), axis[-1])

    def phi(self, axis):
        return np.arctan2(axis[1], axis[0])

    def xyp(self, axis, centroid):
        if axis[-1] == 0:
            return centroid[:2]
        s = -centroid[-1] / axis[-1]
        return (centroid + axis * s)[:2]