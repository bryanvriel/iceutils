#-*- coding: utf-8 -*-

import functools
import numpy as np
from ..matutils import dmultl
from cvxopt import solvers, matrix, sparse, spmatrix, log, div, blas
from sklearn.linear_model import orthogonal_mp_gram, RANSACRegressor
import sys
solvers.options['show_progress'] = False

from ..constants import NULL_REG

def select_solver(solver_type, reg_indices=None, rw_iter=1, regMat=None, robust=False,
                  penalty=1.0, n_nonzero_coefs=10):
    """
    Factory for instantiating a linear regression solver with the correct options
    and returning it.
    """
    # Form regularization indices from regularization matrix (inverse covariance)
    if reg_indices is None and regMat is not None:
        cov = np.diag(regMat)
        reg_indices = (cov < NULL_REG).nonzero()[0]
    elif reg_indices is None:
        reg_indices = []


    # Instantiate a solver
    if solver_type == 'lasso':
        solver = LassoRegression(reg_indices, penalty, regMat=regMat, rw_iter=rw_iter)
    elif solver_type == 'ridge':
        solver = RidgeRegression(reg_indices, penalty, regMat=regMat)
    elif solver_type == 'omp':
        solver = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, regMat=regMat)
    elif solver_type == 'lsqr':
        solver = LinearRegression(robust=robust)

    # Done
    return solver


class LinearRegression:
    """
    Base class for all linear regression solvers. Implements a simple linear
    least squares.
    """

    def __init__(self, robust=False, pinv=False, **kwargs):
        """
        Initialize the LinearRegression class.
        """
        self.robust = robust
        if robust:
            self.ransac = RANSACRegressor(min_samples=10)

        # Cache operator for computing array inverses
        if pinv:
            self.inv_func = functools.partial(np.linalg.pinv, rcond=1.0e-10)
        else:
            self.inv_func = np.linalg.inv

    def invert(self, G, d, wgt=None):
        """
        Simple wrapper around numpy.linalg.lstsq.

        Parameters
        ----------
        G: (M,N) np.ndarray
            Input design matrix.
        d: (M,) np.ndarray
            Input data.
        wgt: (M,) np.ndarray, optional
            Optional weights for the data.

        Returns
        -------
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        # Prepare least squares data
        if wgt is not None:
            GtG = np.dot(G.T, dmultl(wgt**2, G))
            Gtd = np.dot(G.T, wgt**2 * d)
        else:
            GtG = np.dot(G.T, G)
            Gtd = np.dot(G.T, d)
        iGtG = self.inv_func(GtG)

        # Perform inversion
        if self.robust:
            m = self.ransac.fit(GtG, Gtd)
            outlier_mask = np.logical_not(self.ransac.inlier_mask_)
            d[outlier_mask] = np.nan
        else:
            m = np.dot(iGtG, Gtd)
        return m, iGtG


class RidgeRegression(LinearRegression):
    """
    Simple ridge regression (L2-regularization on amplitudes).
    """

    def __init__(self, reg_indices, penalty, regMat=None, **kwargs):
        """
        Initialize the RidgeRegression class and store regularization indices
        and regularization parameter.

        Parameters
        ----------
        reg_indices: np.ndarray
            Regularization indices.
        penalty: float
            Regularization parameter.
        """
        super().__init__(**kwargs)
        self.reg_indices = reg_indices
        self.penalty = penalty
        self.regMat = regMat

        return


    def invert(self, G, d, wgt=None):
        """
        Perform inversion.
        Simple wrapper around numpy.linalg.lstsq.

        Parameters
        ----------
        G: (M,N) np.ndarray
            Input design matrix.
        d: (M,) np.ndarray
            Input data.
        wgt: (M,) np.ndarray, optional
            Optional weights for the data.

        Returns
        -------
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """

        # Cache the regularization matrix or compute it
        regMat = self.regMat
        if regMat is None:
            regMat = np.zeros((G.shape[1], G.shape[1]))
            regMat[self.reg_indices,self.reg_indices] = self.penalty
       
        # Perform inversion 
        if wgt is not None:
            GtG = np.dot(G.T, dmultl(wgt**2, G))
            Gtd = np.dot(G.T, wgt**2 * d)
        else:
            GtG = np.dot(G.T, G)
            Gtd = np.dot(G.T, d)
        iGtG = self.inv_func(GtG + regMat)
        m = np.dot(iGtG, Gtd)
        return m, iGtG


class LassoRegression(LinearRegression):
    """
    Linear regression with an L1-norm regularization function.
    """

    def __init__(self, reg_indices, penalty, reweightingMethod='log',
                 rw_iter=5, regMat=None, estimate_uncertainty=False,
                 **kwargs):
        """
        Initialize the LassoRegression class and store regularization indices,
        regularization parameter, and re-weighting method.

        Parameters
        ----------
        reg_indices: np.ndarray
            Regularization indices.
        penalty: float
            Regularization parameter.
        reweightingMethod: str, {'log', 'inverse', 'isquare'}, optional
            Specify the reweighting method. Default: log.
        rw_iter: int, optional
            Number of re-weighting operations. Default: 5.
        """
        super().__init__(**kwargs)
        self.reg_indices = np.array(reg_indices) 
        self.penalty = penalty
        self.eps = 1.0e-4
        self.rwiter = rw_iter
        self.estimate_uncertainty = estimate_uncertainty
        self.regMat = regMat
        if 'log' in reweightingMethod:
            self.weightingFunc = self.logWeighting
        elif 'inverse' in reweightingMethod:
            self.weightingFunc = self.inverseWeighting
        elif 'isquare' in reweightingMethod:
            self.weightingFunc = self.inverseSquareWeighting
        else:
            raise NotImplementedError('unsupported weighting method')

        return


    def invert(self, G, d, wgt=None):
        """
        Perform inversion.

        Parameters
        ----------
        G: (M,N) np.ndarray
            Input design matrix.
        d: (M,) np.ndarray
            Input data.
        wgt: (M,) np.ndarray, optional
            Optional weights for the data.

        Returns
        -------
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        arrflag = isinstance(self.penalty, np.ndarray)
        weightingFunc = self.weightingFunc

        # If weight array provided, pre-multiply design matrix and data
        if wgt is not None:
            G = dmultl(wgt, G)
            d = wgt * d

        # If a regularization matrix (prior covariance matrix) has been provided
        # convert G -> GtG and d -> Gtd
        if self.regMat is not None:
            d = np.dot(G.T, d)
            G = np.dot(G.T, G) + self.regMat

        # Convert Numpy arrays to CVXOPT matrices
        A = matrix(G.T.tolist())
        b = matrix(d.tolist())
        m, n = A.size
        reg_indices_n = (self.reg_indices + n).tolist()

        # Fill q (will modify for re-weighting)
        q = matrix(0.0, (2*n,1))
        q[:n] = -A.T * b
        q[reg_indices_n] = self.penalty

        # Fill h
        h = matrix(0.0, (2*n,1))

        # Fill P
        P = matrix(0.0, (2*n,2*n))
        P[:n,:n] = A.T * A
        # Add small constant to diagonal for numerical conditioning
        P[list(range(n)),list(range(n))] += 1.0e-8

        # Fill G
        G = matrix(0.0, (2*n,2*n))
        eye = spmatrix(1.0, range(n), range(n))
        G[:n,:n] = eye
        G[:n,n:] = -1.0 * eye
        G[n:,:n] = -1.0 * eye
        G[n:,n:] = -1.0 * eye
        G = sparse(G)

        # Perform re-weighting by calling solvers.coneqp()
        for iters in range(self.rwiter):
            soln = solvers.coneqp(P, q, G=G, h=h)
            status, x = soln['status'], soln['x'][:n]
            if status != 'optimal':
                x = np.nan * np.ones((n,))
                break
            xspl = x[self.reg_indices.tolist()]
            wnew = weightingFunc(xspl)
            if arrflag: # if outputting array, use only 1 re-weight iteration
                q[reg_indices_n] = wnew
            else:
                q[reg_indices_n] = self.penalty * wnew
        x = np.array(x).squeeze()
        q = np.array(q[n:]).squeeze()

        # Estimate uncertainty or set to identity
        if self.estimate_uncertainty:
            # Get indices for variance reduction
            best_ind = self._selectBestBasis(np.array(A), x, d)
            Gsub = np.array(A)[:,best_ind]
            nsub = Gsub.shape[1]
            # Compute new linear algebra arrays
            if wgt is not None:
                GtG = np.dot(Gsub.T, dmultl(wgt**2, Gsub))
                Gtd = np.dot(Gsub.T, wgt**2 * d)
            else:
                GtG = np.dot(Gsub.T, Gsub)
                Gtd = np.dot(Gsub.T, d)
            # Do sub-set least squares
            iGtG = self.inv_func(GtG + 0.01*np.eye(nsub))
            m = np.dot(iGtG, Gtd)
            # Place in original locations
            x = np.zeros(n)
            Cm = np.zeros((n,n))
            x[best_ind] = m
            Cm[best_ind,best_ind] = np.diag(iGtG)
        else:
            Cm = np.eye(n)

        return x, Cm


    def logWeighting(self, x):
        """
        Log re-weighting function used in sparse optimization.

        Parameters
        ----------
        x: np.ndarray
            Array of parameters for weighting.
        """
        ncoeff = x.size[0]
        return log(div(blas.asum(x) + ncoeff*self.eps, abs(x) + self.eps))


    def inverseWeighting(self, x):
        """
        Re-weighting function used in Candes, et al. (2009).

        Parameters
        ----------
        x: np.ndarray
            Array of parameters for weighting.
        """
        return div(1.0, abs(x) + self.eps)

    
    def inverseSquareWeighting(self, x):
        """
        Similar to Candes re-weighting but using inverse square relationship.

        Parameters
        ----------
        x: np.ndarray
            Array of parameters for weighting.
        """
        return div(1.0, x**2 + self.eps**2)


    def _selectBestBasis(self, G, m, d, normalize=False, varThresh=0.95):
        """
        Given a sparse solution, this routine chooses the elements that give the most variance 
        reduction. 
        """
        # Cache indices
        all_ind = np.arange(len(m), dtype=self.reg_indices.dtype)
        steady_ind = np.setxor1d(all_ind, self.reg_indices)
        reg_ind = self.reg_indices

        # First remove the steady-state terms
        refVariance = np.std(d)**2
        dat = d - np.dot(G[:,steady_ind], m[steady_ind])
        variance = np.std(dat)**2
        varianceReduction = 1.0 - variance / refVariance

        # Sort the transient components of m from highest to lowest
        sortIndices = np.argsort(np.abs(m[reg_ind]))[::-1]
        sortIndices = reg_ind[sortIndices]

        # Loop over components and compute variance reduction
        bestIndices = steady_ind.tolist()
        cnt = 0
        ref_var_reduction = varianceReduction
        delta_reduction = 100.0
        while varianceReduction < varThresh:

            # Get the model fit for this component
            index = sortIndices[cnt]
            fit = np.dot(G[:,index], m[index])

            # Remove from data
            dat -= fit
            variance = np.std(dat)**2
            varianceReduction = 1.0 - variance / refVariance

            # Check if we're not getting any better
            delta_reduction = varianceReduction - ref_var_reduction
            if delta_reduction < 1.0e-6:
                break
            ref_var_reduction = varianceReduction

            bestIndices.append(index)
            cnt += 1

            #print(varianceReduction)
            #print(index, m[index])

        return bestIndices


class OrthogonalMatchingPursuit(LinearRegression):


    def __init__(self, n_nonzero_coefs=10, regMat=None): 
        # Cache some properties
        self.n_nonzero_coefs = n_nonzero_coefs
        self.regMat = regMat 


    def invert(self, G, d, wgt=None):
        """
        Perform inversion.

        Parameters
        ----------
        G: (M,N) np.ndarray
            Input design matrix.
        d: (M,) np.ndarray
            Input data.
        wgt: (M,) np.ndarray, optional
            Optional weights for the data.

        Returns
        -------
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        # If weight array provided, pre-multiply design matrix and data
        if wgt is not None:
            G = dmultl(wgt, G)
            d = wgt * d

        # Compute Gram matrix (X.T*X) and product (X.T*y)
        if self.regMat is not None:
            XtX = np.dot(G.T, G) + self.regMat
        else:
            XtX = np.dot(G.T, G)
        Xty = np.dot(G.T, d)

        # Solve
        m = orthogonal_mp_gram(XtX, Xty, n_nonzero_coefs=self.n_nonzero_coefs, copy_Xy=False)
        return m, np.eye(len(m))


# end of file
