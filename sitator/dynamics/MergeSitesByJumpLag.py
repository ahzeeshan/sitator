import numpy as np

from scipy.sparse.csgraph import connected_components

from sitator import SiteNetwork, SiteTrajectory
from sitator.dynamics import JumpAnalysis
from sitator.util import PBCCalculator
from sitator.misc import MergeSitesBy

class MergeSitesByJumpLag(MergeSitesBy):
    """Merges sites based on a jump lag threshold.

    :param float avg_res_time_thresh: Threshold below which sites can be merged.
        In frames; sane values depend on timestep.
    """
    def __init__(self,
                 avg_res_time_thresh,
                 stdev_thresh,
                 min_n_jumps,
                 distance_threshold = 2.5,
                 directed = False,
                 connection = 'weak',
                 **kwargs):
        self.distance_threshold = distance_threshold
        self.avg_res_time_thresh = avg_res_time_thresh
        self.stdev_thresh = stdev_thresh
        self.min_n_jumps = min_n_jumps

        self.directed = directed
        self.connection = connection

        super(MergeSitesByJumpLag, self).__init__(**kwargs)


    def _get_merges(self, st):
        # -- Compute jump statistics
        if not st.site_network.has_attribute('jump_lag'):
            ja = JumpAnalysis(verbose = self.verbose)
            ja.run(st)

        jump_lag = st.site_network.jump_lag
        jump_lag_std = st.site_network.jump_lag_std
        n_ij = st.site_network.n_ij

        pbcc = PBCCalculator(st.site_network.structure.cell)

        # -- Build connectivity_matrix
        n_sites_before = st.site_network.n_sites
        centers = st.site_network.centers

        graph = (jump_lag <= self.avg_res_time_thresh) & \
                (jump_lag_std <= self.stdev_thresh) & \
                (n_ij >= self.min_n_jumps)

        for i, j in zip(*np.where(graph)):
            dist = pbcc.distances(centers[i], [centers[j]])
            if dist[0] > self.distance_threshold:
                graph[i, j] = False

        n_components, components = connected_components(graph,
                                                        directed = self.directed,
                                                        connection = self.connection)

        clusters = []

        for component in xrange(n_components):
            clusters.append(np.where(components == component)[0])

        return clusters



        return clusters
