import warnings

import numpy as np

import numbers

import sys
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    if 'terminal' in ipy_str:
        from tqdm import tqdm
except:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable


class DotProdClassifier(object):
    def __init__(self,
                 threshold = 0.9,
                 max_converge_iters = 10,
                 min_samples = 1):
        """
        :param float threshold: Similarity threshold for joining a cluster.
            In cos-of-angle-between-vectors (i.e. 1 is exactly the same, 0 is orthogonal)
        :param int max_converge_iters: Maximum number of iterations. If the algorithm hasn't converged
            by then, it will exit with a warning.
        :param int|float min_samples: filter out clusters with low sample counts.
            If an int, filters out clusters with fewer samples than this.
            If a float, filters out clusters with fewer than floor(min_samples * n_assigned_samples)
                samples assigned to them.
        """
        self._threshold = threshold
        self._max_iters = max_converge_iters
        self._min_samples = min_samples
        self._cluster_centers = None
        self._cluster_counts = None
        self._featuredim = None

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @property
    def cluster_counts(self):
        return self._cluster_counts

    @property
    def n_clusters(self):
        return len(self._cluster_counts)

    def fit_predict(self, X, verbose = True, predict_threshold = None, return_info = False):
        """ Fit the data vectors X and return their cluster labels.
        """

        assert len(X.shape) == 2, "Training data must be 2D."

        if self._featuredim is None:
            self._featuredim = X.shape[1]
        else:
            raise RuntimeError("DotProdClassifier cannot be fitted twice!")

        if predict_threshold is None:
            predict_threshold = self._threshold

        # Essentially hierarchical clustering that stops when no cluster *centers*
        #  are more similar than the threshold.

        labels = np.empty(shape = len(X), dtype = np.int)
        labels.fill(-1)

        # Start with each sample as a cluster
        old_centers = X
        old_n_assigned = [1] * len(X)
        old_members = [[i] for i in xrange(len(X))]

        # -- Classification loop

        # Maximum number of iterations
        last_n_sites = -1
        did_converge = False

        for iteration in xrange(self._max_iters):
            # This iterations centers
            cluster_centers = list()
            n_assigned_to = list()
            members = list()

            for i, vec in enumerate(tqdm(old_centers, desc = "Iteration %i" % iteration)):

                assigned_to = -1
                assigned_cosang = 0.0

                if len(cluster_centers) > 0:
                    diffs = np.sum(vec * cluster_centers, axis = 1)
                    diffs /= np.linalg.norm(vec) * np.linalg.norm(cluster_centers, axis = 1)

                    assigned_to = np.argmax(diffs)
                    assigned_cosang = diffs[assigned_to]

                    if assigned_cosang < self._threshold:
                        assigned_cosang = 0.0
                        assigned_to = -1

                # If couldn't assign, start a new cluster
                if assigned_to == -1:
                    # New cluster!
                    cluster_centers.append(vec)
                    n_assigned_to.append(old_n_assigned[i])
                    members.append(list())
                    members[-1].extend(old_members[i])
                    assigned_to = len(cluster_centers) - 1
                    # By definition, confidence is 1.0
                    assigned_cosang = 1.0
                else:
                    # Update average center vector of assigned cluster
                    cluster_centers[assigned_to] *= n_assigned_to[assigned_to]
                    cluster_centers[assigned_to] += vec
                    #cluster_centers[assigned_to] = np.maximum(cluster_centers[assigned_to], vec)
                    n_assigned_to[assigned_to] += old_n_assigned[i]
                    members[assigned_to].extend(old_members[i])
                    cluster_centers[assigned_to] /= n_assigned_to[assigned_to]

            old_centers = cluster_centers
            old_n_assigned = n_assigned_to
            old_members = members

            assert [len(m) for m in members] == n_assigned_to, "%s\n%s" % (members, n_assigned_to)

            n_sites = len(n_assigned_to)

            # Check converged
            if last_n_sites == n_sites:
                did_converge = True
                break

            last_n_sites = n_sites

        if not did_converge:
            warnings.warn("Clustering for site type %i did NOT converge after %i iterations" % (site_type, max_converge_iters))

        self._cluster_centers = np.asarray(cluster_centers)

        # Run a predict now:
        labels, confs = self.predict(X, return_confidences = True, verbose = verbose, threshold = predict_threshold)

        total_n_assigned = np.sum(labels >= 0)

        # -- filter out low counts
        if not self._min_samples is None:
            self._cluster_counts = np.bincount(labels[labels >= 0])

            assert len(self._cluster_counts) == len(self._cluster_centers)

            min_samples = None
            if isinstance(self._min_samples, numbers.Integral):
                min_samples = self._min_samples
            elif isinstance(self._min_samples, numbers.Real):
                min_samples = int(np.floor(self._min_samples * total_n_assigned))
            else:
                raise ValueError("Invalid value `%s` for min_samples; must be integral or float." % self._min_samples)

            count_mask = self._cluster_counts >= min_samples

            self._cluster_centers = self._cluster_centers[count_mask]
            self._cluster_counts = self._cluster_counts[count_mask]

            if len(self._cluster_centers) == 0:
                # Then we removed everything...
                raise ValueError("`min_samples` too large; all %i clusters under threshold." % len(count_mask))

            if verbose:
                print "DotProdClassifier: %i/%i assignment counts below threshold %s (%s); %i clusters remain." % \
                    (np.sum(~count_mask), len(count_mask), self._min_samples, min_samples, len(self._cluster_counts))

            # Do another predict -- this could be more efficient, but who cares?
            labels, confs = self.predict(X, return_confidences = True, verbose = verbose, threshold = predict_threshold)

        if return_info:
            info = {
                'clusters_below_min_samples' : np.sum(~count_mask)
            }
            return labels, confs, info
        else:
            return labels, confs

    def predict(self, X, return_confidences = False, threshold = None, verbose = True, ignore_zeros = True):
        """Return a predicted cluster label for vectors X.

        :param float threshold: alternate threshold. Defaults to None, when self.threshold
            is used.

        :returns: an array of labels. -1 indicates no assignment.
        :returns: an array of confidences in assignments. Normalzied
            values from 0 (no confidence, no label) to 1 (identical to cluster center).
        """

        assert len(X.shape) == 2, "Data must be 2D."

        if not X.shape[1] == (self._featuredim):
            raise TypeError("x has wrong dimension %s; should be (%i)" % (x.shape, self._featuredim))

        labels = np.empty(shape = len(X), dtype = np.int)

        if threshold is None:
            threshold = self._threshold

        confidences = None
        if return_confidences:
            confidences = np.empty(shape = len(X), dtype = np.float)

        zeros_count = 0

        center_norms = np.linalg.norm(self._cluster_centers, axis = 1)

        for i, x in enumerate(tqdm(X, desc = "Sample")):

            if np.all(x == 0):
                if ignore_zeros:
                    labels[i] = -1
                    zeros_count += 1
                    continue
                else:
                    raise ValueError("Data %i is all zeros!" % i)

            diffs = np.sum(x * self._cluster_centers, axis = 1)
            diffs /= np.linalg.norm(x) * center_norms

            assigned_to = np.argmax(diffs)
            assignment_confidence = diffs[assigned_to]

            if assignment_confidence < threshold:
                assigned_to = -1
                assignment_confidence = 0.0

            labels[i] = assigned_to
            confidences[i] = assignment_confidence

        if verbose and zeros_count > 0:
            print "Encountered %i zero vectors during prediction" % zeros_count

        if return_confidences:
            return labels, confidences
        else:
            return labels
