import numpy as np

from sitator import SiteNetwork, SiteTrajectory
from sitator.dynamics import JumpAnalysis
from sitator.util import PBCCalculator
from sitator.misc import MergeSitesBy

class MergeSitesByDynamics(MergeSitesBy):
    """Merges sites using Markov Clustering.

    :param float distance_threshold: Don't merge sites further than this
        in real space.
    :param int iterlimit: Maximum number of Markov Clustering iterations to run
        before throwing an error.
    :param dict markov_parameters: Parameters for underlying Markov Clustering.
        Valid keys are ``'inflation'``, ``'expansion'``, and ``'pruning_threshold'``.
    """
    def __init__(self,
                 distance_threshold = 1.0,
                 iterlimit = 100,
                 markov_parameters = {},
                 **kwargs):
        self.distance_threshold = distance_threshold
        self.iterlimit = iterlimit
        self.markov_parameters = markov_parameters

        super(MergeSitesByDynamics, self).__init__(**kwargs)


    def _get_merges(self, st):
        # -- Compute jump statistics
        if not st.site_network.has_attribute('n_ij'):
            ja = JumpAnalysis(verbose = self.verbose)
            ja.run(st)

        pbcc = PBCCalculator(st.site_network.structure.cell)

        # -- Build connectivity_matrix
        connectivity_matrix = st.site_network.n_ij.copy()
        n_sites_before = st.site_network.n_sites
        assert n_sites_before == connectivity_matrix.shape[0]

        centers_before = st.site_network.centers

        # For diagnostic purposes
        no_diag_graph = connectivity_matrix.astype(dtype = np.float, copy = True)
        np.fill_diagonal(no_diag_graph, np.nan)
        # Rather arbitrary, but this is really just an alarm for if things
        # are really, really wrong
        edge_threshold = np.nanmean(no_diag_graph) + 3 * np.nanstd(no_diag_graph)
        n_alarming_ignored_edges = 0

        # Apply distance threshold
        for i in xrange(n_sites_before):
            dists = pbcc.distances(centers_before[i], centers_before[i + 1:])
            js_too_far = np.where(dists > self.distance_threshold)[0]
            js_too_far += i + 1

            if np.any(connectivity_matrix[i, js_too_far] > edge_threshold) or \
               np.any(connectivity_matrix[js_too_far, i] > edge_threshold):
               n_alarming_ignored_edges += 1

            connectivity_matrix[i, js_too_far] = 0
            connectivity_matrix[js_too_far, i] = 0 # Symmetry

        if self.verbose and n_alarming_ignored_edges > 0:
            print("  At least %i site pairs with high (z-score > 3) fluxes were over the given distance cutoff.\n"
                  "  This may or may not be a problem; but if `distance_threshold` is low, consider raising it." % n_alarming_ignored_edges)

        # -- Do Markov Clustering
        clusters = self._markov_clustering(connectivity_matrix, **self.markov_parameters)

        return clusters

    def _markov_clustering(self,
                           transition_matrix,
                           expansion = 2,
                           inflation = 2,
                           pruning_threshold = 0.001):
        """
        See https://micans.org/mcl/.

        Because we're dealing with matrixes that are stochastic already,
        there's no need to add artificial loop values.

        Implementation inspired by https://github.com/GuyAllard/markov_clustering
        """

        assert transition_matrix.shape[0] == transition_matrix.shape[1]

        m1 = transition_matrix.copy()

        # Normalize (though it should be close already)
        m1 /= np.sum(m1, axis = 0)

        allcols = np.arange(m1.shape[1])

        converged = False
        for i in xrange(self.iterlimit):
            # -- Expansion
            m2 = np.linalg.matrix_power(m1, expansion)
            # -- Inflation
            np.power(m2, inflation, out = m2)
            m2 /= np.sum(m2, axis = 0)
            # -- Prune
            to_prune = m2 < pruning_threshold
            # Exclude the max of every column
            to_prune[np.argmax(m2, axis = 0), allcols] = False
            m2[to_prune] = 0.0
            # -- Check converged
            if np.allclose(m1, m2):
                converged = True
                if self.verbose:
                    print "Markov Clustering converged in %i iterations" % i
                break

            m1[:] = m2

        if not converged:
            raise ValueError("Markov Clustering couldn't converge in %i iterations" % self.iterlimit)

        # -- Get clusters
        attractors = m2.diagonal().nonzero()[0]

        clusters = set()

        for a in attractors:
            cluster = tuple(m2[a].nonzero()[0])
            clusters.add(cluster)

        return list(clusters)
