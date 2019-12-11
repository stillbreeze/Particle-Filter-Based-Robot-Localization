import numpy as np
import pdb
import random
class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        X_bar_resampled = []
        M = len(X_bar)
        wt = X_bar[:,3]
        r = random.uniform(0, 1.0/M)
        wt /= wt.sum()
        c = wt[0]
        i = 0
        for m in range(M):
            u = r + (m)*(1.0/M)
            while u>c:
                i = i +1
                c = c + wt[i]
            X_bar_resampled.append(X_bar[i])
        X_bar_resampled = np.asarray(X_bar_resampled)

        return X_bar_resampled

if __name__ == "__main__":
    pass