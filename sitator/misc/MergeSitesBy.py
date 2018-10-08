import numpy as np

from sitator import SiteNetwork, SiteTrajectory
from sitator.util import PBCCalculator

from abc import ABCMeta, abstractmethod

class MergeSitesBy(object):
    """Merges sites using dynamical data.

    An abstract base class.

    :param bool check_types: If True, only sites of the same type are candidates to
        be merged; if false, type information is ignored. Merged sites will only
        be assigned types if this is True.
    :param final_distance_threshold: A final distance threshold for sanity checks
        while merging. Can be `None`, defaults to 2.5.
    :param bool verbose:
    """

    def __init__(self,
                 check_types = True,
                 verbose = True,
                 final_distance_threshold = 2.5):

        self.verbose = verbose
        self.final_distance_threshold = final_distance_threshold
        self.check_types = check_types

    def run(self, st):
        """Takes a SiteTrajectory and returns a SiteTrajectory, including a new SiteNetwork."""

        if self.check_types and st.site_network.site_types is None:
            raise ValueError("Cannot run a check_types=True MergeSitesByDynamics on a SiteTrajectory without type information.")

        pbcc = PBCCalculator(st.site_network.structure.cell)
        site_centers = st.site_network.centers
        if self.check_types:
            site_types = st.site_network.site_types

        # Get what to merge from subclass
        clusters = self._get_merges(st)

        new_n_sites = len(clusters)

        if self.verbose:
            print "After merge there will be %i sites" % new_n_sites

        if self.check_types:
            new_types = np.empty(shape = new_n_sites, dtype = np.int)

        # -- Merge Sites
        new_centers = np.empty(shape = (new_n_sites, 3), dtype = st.site_network.centers.dtype)
        translation = np.empty(shape = st.site_network.n_sites, dtype = np.int)
        translation.fill(-1)

        for newsite in xrange(new_n_sites):
            mask = list(clusters[newsite])
            # Update translation table
            if np.any(translation[mask] != -1):
                # We've assigned a different cluster for this before... weird
                # degeneracy
                raise ValueError("Markov clustering tried to merge site(s) into more than one new site")
            translation[mask] = newsite

            to_merge = site_centers[mask]

            # Check distances
            if not self.final_distance_threshold is None:
                dists = pbcc.distances(to_merge[0], to_merge[1:])
                assert np.all(dists < self.final_distance_threshold), \
                            "Tried to merge sites more than final_distance_threshold = %f apart. Modify merge settings or raise `final_distance_threshold`." % (self.final_distance_threshold)

            # New site center
            new_centers[newsite] = pbcc.average(to_merge)
            if self.check_types:
                assert np.all(site_types[mask] == site_types[mask][0])
                new_types[newsite] = site_types[mask][0]

        newsn = st.site_network.copy()
        newsn.centers = new_centers
        if self.check_types:
            newsn.site_types = new_types

        newtraj = translation[st._traj]
        newtraj[st._traj == SiteTrajectory.SITE_UNKNOWN] = SiteTrajectory.SITE_UNKNOWN

        # It doesn't make sense to propagate confidence information through a
        # transform that might completely invalidate it
        newst = SiteTrajectory(newsn, newtraj, confidences = None)

        if not st.real_trajectory is None:
            newst.set_real_traj(st.real_trajectory)

        return newst

    @abstractmethod
    def _get_merges(self, st):
        """Get what sites to merge.

        Should return a list of tuples. Each tuple lists a set of sites to be
        merged into one site. The length of that list is the new number of sites.
        """
        pass
