####################################
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
####################################
import time
import itertools
from typing import List, Optional, Sequence
from numpy.typing import NDArray
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize


def rmse(x,y):
    return np.sqrt(np.mean((x-y)**2))

################################################################################################################
# kernels: cubic spline kernel, categorical kernel, and linear kernel

# cubic spline kernel
class SSkernel:
    def __init__(self):
        pass

    def K(self, X: NDArray, X2: NDArray = None) -> tf.Tensor:
        B1X = X - 0.5
        B2X = (X**2 - X + 1.0/6)*0.5
        if X2 is None:
            X2 = X
            B1X2 = B1X
            B2X2 = B2X
        else:
            B1X2 = X2 - 0.5
            B2X2 = (X2 ** 2 - X2 + 1/6) * 0.5
        tmp = tf.math.abs(X - tf.transpose(X2))
        B4X = tmp**4 - 2*tmp**3 + tmp**2 - 1/30
        part1 = tf.matmul(B1X, B1X2, transpose_b=True) + tf.matmul(B2X, B2X2, transpose_b=True)
        k = part1 - 1.0/24 * B4X
        return k

    def K_diag(self, X: NDArray) -> tf.Tensor:
        B1X = X - 0.5
        B2X = X ** 2 - X + 1.0 / 6
        k = np.linalg.matmul(B1X, B1X.T) + np.linalg.matmul(B2X, B2X.T) + 1.0/720
        return tf.linalg.diag_part(k)


# categorical kernel
class CategoricalKernel:
    def __init__(self):
       pass

    def K(self, X:NDArray, X2:NDArray = None) -> tf.Tensor:
        if X2 is None:
            X2 = X
        X = tf.constant(X, dtype=tf.float64)
        X2 = tf.constant(X2, dtype=tf.float64)
        G = len(np.unique(X))
        K = (G-1)/G * tf.cast(X == tf.transpose(X2), dtype=tf.float64) - 1/G * tf.cast(X != tf.transpose(X2), dtype=tf.float64)
        return K

    def K_diag(self, X:NDArray):
        return tf.linalg.diag_part(self.K(X))


# linear kernel
class LinearKernel:
    def __init__(self):
        pass

    def K(self, X: NDArray, X2: NDArray = None) -> tf.Tensor:
        B1X = X - 0.5
        if X2 is None:
            B1X2 = B1X
        else:
            B1X2 = X2 - 0.5
        k = tf.matmul(B1X, B1X2, transpose_b=True)
        return k

    def K_diag(self, X:NDArray) -> tf.Tensor:
        B1X = X - 0.5
        k = tf.matmul(B1X, B1X, transpose_b=True)
        return tf.linalg.diag_part(k)

################################################################################################################


class GPANOVAModel:
    def __init__(
            self,
            X: NDArray,
            Y: NDArray,
            max_interaction_depth: int = 2,
            active_dims: Optional[List[List[int]]] = None,
            categorical_variables: Optional[List] = None,
            linear_variables: Optional[List] = None,
    ):
        """
        :param X: n*p design matrix
        :param Y: n*1 response vector
        :param max_interaction_depth: maximum order of interactions, default to 2 (second-order interactions)
        :param active_dims: pre-assigned active indexes, default to the all components in FANOVA whose order<=max_interaction_depth
        :param categorical_variables: dims for categorical variables
        :param linear_variables: dims for linear variables
        """
        # initialize model parameters
        self.categorical_variables = categorical_variables
        self.max_interaction_depth = max_interaction_depth
        self.categorical_variables = [] if categorical_variables is None else categorical_variables
        self.linear_variables = [] if linear_variables is None else linear_variables
        # The categorical variables and linear variables can not be overlapped.
        if set(self.categorical_variables) & set(self.linear_variables):
            raise ValueError(f"Overlap found in categorical and linear variables.")

        # scale data
        self.N, self.p = X.shape
        self.X, self.Y = X, Y
        self.Y_mean, self.Y_std = np.mean(Y), np.std(Y)
        self.Y_scaled = (self.Y - self.Y_mean) / self.Y_std
        # initialization
        self.kernel_matrices = []
        for dim in range(self.p):
            if dim in self.categorical_variables:
                K = CategoricalKernel().K(X=self.X[:, dim].reshape(-1, 1), X2=None)
            elif dim in self.linear_variables:
                K = LinearKernel().K(X=self.X[:, dim].reshape(-1, 1), X2=None)
            else:
                K = SSkernel().K(X=self.X[:, dim].reshape(-1, 1), X2=None)
            self.kernel_matrices.append(K)

        # initial selected components and the corresponding kernel variances
        self.selected_dims = []  # all dims including active and inactive indexes with order <= max_interaction_depth
        # the dims are stored in the order like [[0], ..., [p-1],
        #                                       [i, j], j=i+1, ..., p-1, i=0,..., p-2]
        # where the dims are denoted as 0, 1, ..., p-1 for convenience in python
        for ii in range(self.max_interaction_depth + 1):
            if ii > 0:
                tmp = [
                    list(tup) for tup in itertools.combinations(range(self.p), ii)
                ]
                self.selected_dims = self.selected_dims + tmp
        self.active_dims = self.selected_dims.copy() if active_dims is None else active_dims
        self.mu, self.sigma2, self.tau = 0.0, 1.0, 0.001
        self.individual_variances = [1.0] * self.p  # variances for main effects
        self.interaction_variances = [1.0] * (len(self.selected_dims) - self.p)  # variance for interactions

    def compute_covariance(self, kernel_matrices: Sequence[tf.Tensor]) -> tf.Tensor:
        """
        :param kernel_matrices: Sequence of one-dimensional kernel matrices
            [K_1(X1, X2), K_2(X1, X2), ..., K_p(X1, X2)].
        returns:
        A covariance matrix:
            Kn = Σ r_j * K_j + Σ r_ij * (K_i * K_j) + ...
        where interactions are computed over active dimensions.
        """
        variances = self.individual_variances + self.interaction_variances
        # Precompute lookup: active_dim -> variance index
        dim_to_var_index = {tuple(dim): i for i, dim in enumerate(self.selected_dims)}
        additive_terms = []
        for active_dim in self.active_dims:
            var_idx = dim_to_var_index[tuple(active_dim)]
            term = variances[var_idx] * tf.reduce_prod(
                [kernel_matrices[i] for i in active_dim], axis=0
            )
            additive_terms.append(term)

        return tf.add_n(additive_terms) if additive_terms else tf.zeros_like(kernel_matrices[0])

    def update_mu_sigma2(self):
        # Fix r, update (mu, sigma2) by using the MLE：
        # mu = 1^T(Kn + tau*In)^{-1} yn / 1n^T (Kn + tau In)^{-1} 1n
        # sigma2 = (1/n) (yn - mu * 1n)^T (Kn + tau*In)^{-1} (yn - mu* 1n)
        K = self.compute_covariance(self.kernel_matrices)
        Q = tf.add(K, (self.tau + 1e-12) * tf.eye(self.N, dtype=K.dtype))

        # Cholesky factor
        L = tf.linalg.cholesky(Q)
        # Prepare RHS for both solves in one batch: [ones_vec, Y_scaled]
        rhs = tf.concat([
            tf.ones((self.N, 1), dtype=Q.dtype),
            self.Y_scaled
        ], axis=1)
        V = tf.linalg.triangular_solve(L, rhs)  # shape (N, 2), single triangular solve (forward substitution)
        v, v1 = V[:, 0:1], V[:, 1:2] # Extract the two solved vectors
        # Compute the MLEs
        numerator = tf.matmul(v, v1, transpose_a=True)  # scalar [[x]]
        denominator = tf.matmul(v, v, transpose_a=True)  # scalar [[y]]
        self.mu = tf.squeeze(numerator / denominator).numpy().item()
        r_solve = tf.linalg.triangular_solve(L, self.Y_scaled - self.mu)
        quad_form = tf.matmul(r_solve, r_solve, transpose_a=True)
        self.sigma2 = (tf.squeeze(quad_form) / tf.cast(self.N, Q.dtype)).numpy().item()

    def loglikelihood(self, tau):
        K = self.compute_covariance(self.kernel_matrices)  # (N,N)
        # Q = gpflow.utilities.add_noise_cov(K, likelihood_variance=(tau + 1e-8) * np.ones((N, ))) # Add jitter for stability
        Q = tf.add(K, tau * tf.eye(self.N, dtype=K.dtype))
        L = tf.linalg.cholesky(Q)
        r_solve = tf.linalg.triangular_solve(L, self.Y_scaled - self.mu)
        quad = tf.matmul(r_solve, r_solve, transpose_a=True)
        # log |Q| = 2 * sum(log(diag(L)))
        logdetQ = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        # objective: (N/2) log sigma2 + log|Q| + (1/(2 sigma2)) * quad
        #loss = (tf.cast(self.N, K.dtype) / 2.0) * tf.math.log(self.sigma2) \
        #       + logdetQ \
        #       + (0.5 / self.sigma2) * tf.squeeze(quad)
        obj = tf.cast(self.N, K.dtype) * tf.math.log(quad) + logdetQ

        return obj

    def update_tau(self, init_tau):
        """
        Args:
            init_tau: the initial value of tau does not affect the convergence
        Returns:
            the MLE estimate of tau via minimizing the profiled log-likelihood
        """
        dtype = tf.float64
        # local helper to compute objective with jitter if desired
        # @tf.function
        def tf_objective_and_grad(tau_np):
            tau = tf.convert_to_tensor(tau_np, dtype=dtype)
            with tf.GradientTape() as tape:
                tape.watch(tau)
                obj = self.loglikelihood(tau)
            grad = tape.gradient(obj, tau)
            return obj.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        # wrapper for scipy
        def obj_fn(tau_np):
            val, grad = tf_objective_and_grad(tau_np)
            return val, grad

            # run L-BFGS-B with positivity bound
        res = minimize(obj_fn, x0=[init_tau], method="L-BFGS-B", jac=True, bounds=[(1e-4, None)])

        # convert to Python floats
        tau_opt = float(res.x[0])
        obj_val = self.loglikelihood(tf.constant(tau_opt, dtype=dtype))

        # store on self as plain Python floats (as you requested previously)
        self.tau = tau_opt
        self.profile_loss = float(obj_val.numpy().item())
        self.sigma2_noise = self.tau * self.sigma2

    def update_f(self):
        K = self.compute_covariance(self.kernel_matrices)  # (N,N)
        Q = tf.add(K, (self.tau + 1e-12) * tf.eye(self.N, dtype=K.dtype))
        L = tf.linalg.cholesky(Q)
        v = tf.linalg.solve(L, self.Y_scaled - self.mu)
        self.alpha = tf.linalg.solve(tf.transpose(L), v)

    def Step1(self, init_tau):
        """
        Step 1: update mu, sigma^2, tau (then sigma^2_epsilon), and f (and f_V).
        """
        # if tune_sigma2:
        self.update_mu_sigma2()
        self.update_tau(init_tau=init_tau)
        # self.update_f()

    def Step2(self, omega=None, r_threshold=1e-8, cut_off_r=False):
        """
        Args:
            omega: the regularization parameter, 10/N_train is the default.
            r_threshold: a cut-off value, let r = 0 if r<r_threshold.
        Returns:
            the H-likelihood predictor for each (nonzero) r.
        """
        self.update_f()
        if omega is None:
            omega = 10 / self.N  # default omega
        variances = self.individual_variances + self.interaction_variances
        nonzero_components = []
        for active_dim in self.active_dims:
            iComponent_location = self.selected_dims.index(active_dim)
            k_mats = [K for K in [self.kernel_matrices[l] for l in active_dim]]
            cov = variances[iComponent_location] * tf.reduce_prod(k_mats, axis=0)
            fKf = tf.matmul(self.alpha, tf.matmul(cov, self.alpha), transpose_a=True) * variances[iComponent_location]
            variance_i = 0.25 * (2 - (self.N + 2) * omega) + tf.sqrt(
                    (2 - (self.N + 2) * omega) ** 2 + 8 * omega * fKf) * 0.25
            if cut_off_r:
                variances[iComponent_location] = variance_i.__float__() if variance_i.__float__() > r_threshold  else 0.0
            else:
                variances[iComponent_location] = variance_i.__float__()

            if variance_i.__float__() > 0.0:
                    nonzero_components.append(iComponent_location)

        self.individual_variances = [variances[i] for i in range(self.p)]
        self.interaction_variances = [variances[i] for i in range(self.p, len(self.selected_dims))]
        self.active_dims = [self.selected_dims[i] for i in nonzero_components]

    def fit(self, omega=None, init_tau=None, gamma=1e-4, r_threshold=1e-6, verbose=False):
        # initialize
        iter = 1
        # not real initial values, just for the while loop.
        mu_0, mu_1 = 0, 0.1
        sigma2_0, sigma2_1 = 0, 1
        tau_0, tau_1 = 0, 0.001

        # real initialization
        self.mu = 0.0
        self.tau = init_tau if init_tau is not None else 0.001
        r_0 = np.copy([0]*len(self.individual_variances + self.interaction_variances))
        r_1 = np.copy(self.individual_variances + self.interaction_variances)
        t_start = time.time()
        cut_off_r = False
        while max([
            rmse(mu_1, mu_0)/(rmse(mu_0, 0) + 1e-6),
            rmse(sigma2_1, sigma2_0)/(rmse(sigma2_0, 0) + 1e-6),
            rmse(tau_1, tau_0)/(rmse(tau_0, 0) + 1e-6),
            rmse(r_1, r_0)/(rmse(r_0, 0)+ 1e-6)
             ]) > gamma or iter == 1:

            self.Step1(init_tau=self.tau)
            self.Step2(omega=omega, r_threshold=r_threshold, cut_off_r=cut_off_r)
            if iter > 10:
                cut_off_r = True

            if verbose:
                print(f"{iter}-th updating: mu = {self.mu}, sigma2 = {self.sigma2}, tau = {self.tau}" + "\n")
                print(f"{iter}-th updating: individual variances = {self.individual_variances}" + "\n")
                print(f"{iter}-th updating: interaction variances = {self.interaction_variances}" + "\n")
                max_change_of_param = max([rmse(mu_1, mu_0)/(rmse(mu_0, 0) + 1e-6), rmse(sigma2_1, sigma2_0)/(rmse(sigma2_0, 0) + 1e-6),
                    rmse(tau_1, tau_0)/(rmse(tau_0, 0) + 1e-6),
                    rmse(r_1, r_0)/(rmse(r_0, 0)+ 1e-6)])
                print(f"{iter}-th iterative maximum change of parameters (scaled) = {max_change_of_param}" + "\n")

            mu_0, sigma2_0, tau_0, r_0 = mu_1, sigma2_1, tau_1, r_1
            mu_1, sigma2_1, tau_1 = self.mu, self.sigma2, self.tau
            r_1 = np.copy(self.individual_variances + self.interaction_variances)
            iter += 1
        fitting_time_seconds = time.time() - t_start
        print(f"GPANOVA (Algorithm 1) training took {fitting_time_seconds:.3f} seconds.")

    def smoother_matrix(self, K: tf.Tensor, tau: float):
        """
        Compute the smoother matrix: fn = A(lambda1)yn
        Args:
            K: K = Σ r_j * K_j + Σ r_ij * (K_i * K_j) + ... for nonzero r_j and r_ij, the covariance matrix
            loglambda1: log(lambda1), lambda1, a tuning parameter

        Returns:
            The Smoother matrix A(lambda1)
        """
        n = K.shape[0]
        dtype = K.dtype
        tau = tf.convert_to_tensor(tau, dtype=dtype)
        # Step 1: Compute the smoothing matrix fn = A(lambda) yn
        Q = tf.add(K, tau * tf.eye(n, dtype=dtype))
        Q_inv = tf.linalg.inv(Q)
        J = tf.ones((n, 1), dtype=dtype)
        alpha = tf.matmul(J, Q_inv, transpose_a=True) / tf.squeeze(tf.matmul(J, tf.matmul(Q_inv, J), transpose_a=True))
        A = tf.matmul(tf.matmul(K, Q_inv), tf.eye(n, dtype=dtype) - tf.matmul(J, alpha)) + tf.matmul(J, alpha)
        return A

    def predict(self, Xnew: NDArray):
        """
        :param Xnew: inputs to predict the response on
        :return: predicted response on input Xnew
        """
        kernel_matrices = []
        for dim in range(self.p):
            if dim in self.categorical_variables:
                K = CategoricalKernel().K(X=self.X[:, dim].reshape(-1, 1), X2=Xnew[:, dim].reshape(-1, 1))
            elif dim in self.linear_variables:
                K = LinearKernel().K(X=self.X[:, dim].reshape(-1, 1), X2=Xnew[:, dim].reshape(-1, 1))
            else:
                K = SSkernel().K(X=self.X[:, dim].reshape(-1, 1), X2=Xnew[:, dim].reshape(-1, 1))
            kernel_matrices.append(K)

        k_mat_new = self.compute_covariance(kernel_matrices)
        f_pred = self.mu + tf.linalg.matmul(k_mat_new, self.alpha, transpose_a=True).numpy()[:, 0]
        return self.Y_mean + self.Y_std * f_pred

    def predict_by_component(self, Xnew: NDArray):
        variances = self.individual_variances + self.interaction_variances
        # Precompute lookup: active_dim -> variance index
        dim_to_var_index = {tuple(dim): i for i, dim in enumerate(self.selected_dims)}
        kernel_matrices = []
        for dim in range(self.p):
            if dim in self.categorical_variables:
                K = CategoricalKernel().K(X=self.X[:, dim].reshape(-1, 1), X2=Xnew[:, dim].reshape(-1, 1))
            elif dim in self.linear_variables:
                K = LinearKernel().K(X=self.X[:, dim].reshape(-1, 1), X2=Xnew[:, dim].reshape(-1, 1))
            else:
                K = SSkernel().K(X=self.X[:, dim].reshape(-1, 1), X2=Xnew[:, dim].reshape(-1, 1))
            kernel_matrices.append(K)

        n_samples = Xnew.shape[0]
        n_components = len(self.selected_dims)
        ncurves = np.zeros((n_samples, n_components))
        # additive_terms = []
        for dim in self.selected_dims:
            var_idx = dim_to_var_index[tuple(dim)]
            term = variances[var_idx] * tf.reduce_prod(
                [kernel_matrices[i] for i in dim], axis=0
            )
            #additive_terms.append(term)
            f_pred = tf.linalg.matmul(term, self.alpha, transpose_a=True)
            ncurves[:, var_idx] = tf.reshape(f_pred, [-1]).numpy()
        return self.Y_std * ncurves

    def selected_main_effects(self):
        return [i for i in self.active_dims if len(i) == 1]

    def selected_interactions(self):
        return [i for i in self.active_dims if len(i) > 1]

