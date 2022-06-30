#-*- coding: utf-8 -*-

import warnings
import functools
import numpy as np
from ..matutils import dmultl
from cvxopt import solvers, matrix, sparse, spmatrix, log, div, blas
from sklearn.linear_model import orthogonal_mp_gram, RANSACRegressor, \
                                 LinearRegression as sk_LinearRegression
import sys
solvers.options['show_progress'] = False

from ..constants import NULL_REG, FAIL, SUCCESS

def select_solver(solver_type, reg_indices=None, rw_iter=1, regMat=None, robust=False,
                  penalty=1.0, n_nonzero_coefs=10, n_min=20):
    """
    Factory for instantiating a linear regression solver with the correct options
    and returning it.

    Parameters
    ----------
    solver_type: str,
        Name of solver from ('lasso', 'ridge', 'omp', 'ransac', 'lsqr').
    reg_indices: array_like, optional
        Integer indices for elements to be regularized. Default: None.
    rw_iter: int, optional
        Number of re-weighting iterations for Lasso solver. Default: 1.
    regMat: ndarray, optional
        Extra prior covariance matrix. Default: None.
    robust: bool, optional
        Use RANSAC regressor for lsqr solver. Default: False.
    penalty: float, optional
        Regularization penalty for lasso and ridge solvers. Default: 1.0.
    n_nonzero_coefs: int, optional
        Number of non-zero regularized elements for omp solver. Default: 10.
    n_min: int, optional
        Minimum number of valid data to perform inversion. Default: 20.

    Returns
    -------
    solver: LinearRegression
        Instantiated solver.
    """
    # Form regularization indices from regularization matrix (inverse covariance)
    if reg_indices is None and regMat is not None:
        cov = np.diag(regMat)
        reg_indices = (cov < NULL_REG).nonzero()[0]
    elif reg_indices is None:
        reg_indices = []

    # Instantiate a solver
    if solver_type == 'lasso':
        solver = LassoRegression(reg_indices, penalty, regMat=regMat, rw_iter=rw_iter,
                                 robust=robust, n_min=n_min)
    elif solver_type == 'ridge':
        solver = RidgeRegression(reg_indices, penalty, regMat=regMat, n_min=n_min)
    elif solver_type == 'omp':
        solver = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, regMat=regMat,
                                           n_min=n_min)
    elif solver_type == 'ransac':
        solver = LinearRegression(robust=True, n_min=n_min)
    elif solver_type == 'lsqr':
        solver = LinearRegression(robust=robust, n_min=n_min)

    # Done
    return solver


class LinearRegression:
    """
    Base class for all linear regression solvers. Implements a simple linear
    least squares.
    """

    def __init__(self, robust=False, pinv=False, n_min=20, **kwargs):
        """
        Initialize the LinearRegression class.
        """
        self.robust = robust
        self.n_min = n_min
        if robust:
            estimator = sk_LinearRegression(fit_intercept=False)
            self.ransac = RANSACRegressor(estimator=estimator, min_samples=10)

        # Cache operator for computing array inverses
        if pinv:
            self.inv_func = functools.partial(np.linalg.pinv, rcond=1.0e-10)
        else:
            self.inv_func = np.linalg.inv

    def invert(self, G, d, wgt=None, mask=None):
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
        mask: np.ndarray, optional
            Array of indices to use for inversion. Default: None.

        Returns
        -------
        status: int
            Integer flag for failure or success.
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        # Indices for finite data
        if mask is None:
            mask = np.isfinite(d).nonzero()[0]
        if mask.size < self.n_min:
            warnings.warn('Not enough data for inversion. Returning None.')
            return FAIL, None, None
        Gf, df, wgt = self.apply_mask(mask, G, d, wgt=wgt)

        # If doing an inversion with RANSAC, use directly in order to maintain
        # a design matrix with M data poitns
        if self.robust:
            self.ransac.fit(Gf, df)
            m = self.ransac.estimator_.coef_
            G_in = Gf[self.ransac.inlier_mask_, :]
            GtG = np.dot(G_in.T, G_in)
            return SUCCESS, m, self.inv_func(GtG)
        
        # Prepare least squares data
        if wgt is not None:
            GtG = np.dot(Gf.T, dmultl(wgt**2, Gf))
            Gtd = np.dot(Gf.T, wgt**2 * df)
        else:
            GtG = np.dot(Gf.T, Gf)
            Gtd = np.dot(Gf.T, df)
        iGtG = self.inv_func(GtG)

        # Perform inversion
        m = np.dot(iGtG, Gtd)
        return SUCCESS, m, iGtG

    @staticmethod
    def apply_mask(mask, G, d, wgt=None):
        """
        Convenience function for returning subset of least squares specified by a logical mask.
        """
        if wgt is not None:
            return G[mask], d[mask], wgt[mask]
        else:
            return G[mask], d[mask], None
        

class RidgeRegression(LinearRegression):
    """
    Simple ridge regression (L2-regularization on amplitudes).
    """

    def __init__(self, reg_indices, penalty, robust=False, regMat=None, **kwargs):
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


    def invert(self, G, d, wgt=None, mask=None):
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
        mask: np.ndarray, optional
            Array of indices to use for inversion. Default: None.

        Returns
        -------
        status: int
            Integer flag for failure or success.
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        # Indices for finite data
        if mask is None:
            mask = np.isfinite(d).nonzero()[0]
        if mask.size < self.n_min:
            warnings.warn('Not enough data for inversion. Returning None.')
            return FAIL, None, None
        Gf, df, wgt = self.apply_mask(mask, G, d, wgt=wgt)

        # Cache the regularization matrix or compute it
        regMat = self.regMat
        if regMat is None:
            regMat = np.zeros((G.shape[1], G.shape[1]))
            regMat[self.reg_indices,self.reg_indices] = self.penalty
       
        # Perform inversion 
        if wgt is not None:
            GtG = np.dot(Gf.T, dmultl(wgt**2, Gf))
            Gtd = np.dot(Gf.T, wgt**2 * df)
        else:
            GtG = np.dot(Gf.T, Gf)
            Gtd = np.dot(Gf.T, df)
        if self.robust:
            m = self.ransac.fit(GtG, Gtd)
        else:
            iGtG = self.inv_func(GtG + regMat)
            m = np.dot(iGtG, Gtd)
        return SUCCESS, m, iGtG


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


    def invert(self, G, d, wgt=None, mask=None):
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
        mask: np.ndarray, optional
            Array of indices to use for inversion. Default: None.

        Returns
        -------
        status: int
            Integer flag for failure or success.
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        # Indices for finite data
        if mask is None:
            mask = np.isfinite(d).nonzero()[0]
        if mask.size < self.n_min:
            warnings.warn('Not enough data for inversion. Returning None.')
            return FAIL, None, None
        Gf, df, wgt = self.apply_mask(mask, G, d, wgt=wgt)

        arrflag = isinstance(self.penalty, np.ndarray)
        weightingFunc = self.weightingFunc

        # Cache design matrix and data vector
        G_input = Gf.copy()
        d_input = df.copy()

        # If weight array provided, pre-multiply design matrix and data
        if wgt is not None:
            Gf = dmultl(wgt, Gf)
            df = wgt * df

        # If a regularization matrix (prior covariance matrix) has been provided
        # convert G -> GtG and d -> Gtd (Gram products)
        if self.regMat is not None:
            df = np.dot(Gf.T, df)
            Gf = np.dot(Gf.T, Gf) + self.regMat

        # Convert Numpy arrays to CVXOPT matrices
        A = matrix(Gf.T.tolist())
        b = matrix(df.tolist())
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
            best_ind = self._selectBestBasis(G_input, x, d_input)
            Gsub = G_input[:, best_ind]
            dsub = d_input
            # Compute new linear algebra arrays
            if wgt is not None:
                Gsub = dmultl(wgt, Gsub)
                dsub = d_input * wgt
            if self.regMat is not None:
                regMat = self.regMat[best_ind,:][:,best_ind]
                GtG = np.dot(Gsub.T, Gsub) + regMat
            else:
                GtG = np.dot(Gsub.T, Gsub)
            Gtd = np.dot(Gsub.T, dsub)
            # Inverse and least squares
            iGtG = self.inv_func(GtG)
            m = np.dot(iGtG, Gtd)
            # Place in original locations
            x = np.zeros(n)
            Cm = np.zeros((n,n))
            x[best_ind] = m
            row, col = np.meshgrid(best_ind, best_ind)
            Cm[row, col] = iGtG
        else:
            Cm = np.eye(n)

        return SUCCESS, x, Cm


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

        # Extract dictionary elements to traverse (regularized elements)
        G_reg = G[:, reg_ind]
        m_reg = m[reg_ind]

        # Sort the transient components of m from highest to lowest
        sortIndices = np.argsort(np.abs(m_reg))[::-1]

        # Loop over components and compute variance reduction
        bestIndices = steady_ind.tolist()
        cnt = 0
        ref_var_reduction = varianceReduction
        delta_reduction = 100.0
        while cnt < len(reg_ind):

            # Get the model fit for this component
            index = sortIndices[cnt]
            fit = np.dot(G_reg[:,index], m_reg[index])

            # Remove from data
            dat -= fit
            variance = np.std(dat)**2
            varianceReduction = 1.0 - variance / refVariance
            bestIndices.append(reg_ind[index])

            #print(varianceReduction)

            # Check if we've met threshold
            if varianceReduction >= varThresh:
                print('Breaking due to threshold')
                break
            cnt += 1
            
        return bestIndices


class OrthogonalMatchingPursuit(LinearRegression):


    def __init__(self, n_nonzero_coefs=10, regMat=None, **kwargs): 
        super().__init__(**kwargs)
        # Cache some properties
        self.n_nonzero_coefs = n_nonzero_coefs
        self.regMat = regMat 


    def invert(self, G, d, wgt=None, mask=None):
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
        mask: np.ndarray, optional
            Array of indices to use for inversion. Default: None.

        Returns
        -------
        status: int
            Integer flag for failure or success.
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        # Indices for finite data
        if mask is None:
            mask = np.isfinite(d).nonzero()[0]
        if mask.size < self.n_min:
            warnings.warn('Not enough data for inversion. Returning None.')
            return FAIL, None, None
        Gf, df, wgt = self.apply_mask(mask, G, d, wgt=wgt)

        # If weight array provided, pre-multiply design matrix and data
        if wgt is not None:
            Gf = dmultl(wgt, Gf)
            df = wgt * df

        # Compute Gram matrix (X.T*X) and product (X.T*y)
        if self.regMat is not None:
            XtX = np.dot(Gf.T, Gf) + self.regMat
        else:
            XtX = np.dot(Gf.T, Gf)
        Xty = np.dot(Gf.T, df)

        # Solve
        m = orthogonal_mp_gram(XtX, Xty, n_nonzero_coefs=self.n_nonzero_coefs, copy_Xy=False)
        return SUCCESS, m, np.eye(len(m))


# end of file
