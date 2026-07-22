#-
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import itertools
from typing import List, Optional, Sequence
from numpy.typing import NDArray
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar
from functools import partial
from sklearn.linear_model import Lasso
import warnings


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


class SSANOVAModel:
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
        :param Y: n*1 response
        :param max_interaction_depth: maximum order of interactions, default to 2 (second-order interactions)
        :param active_dims: pre-assigned active indexes, default to the all components in FANOVA whose order<=max_interaction_depth
        :param categorical_variables: dims for categorical variables
        :param linear_variables: dims for linear variables
        """
        # scale data
        self.X, self.Y = X, Y
        self.Y_mean, self.Y_std = np.mean(Y), np.std(Y)
        self.Y_scaled = (self.Y - self.Y_mean) / self.Y_std

        # initialization
        self.N, self.p = X.shape
        self.max_interaction_depth = max_interaction_depth
        self.categorical_variables = [] if categorical_variables is None else categorical_variables
        self.linear_variables = [] if linear_variables is None else linear_variables
        # The categorical variables and linear variables can not be overlapped.
        if set(self.categorical_variables) & set(self.linear_variables):
            raise ValueError(f"Overlap found in categorical and linear variables.")

        self.kernel_matrices = []
        for dim in range(self.p):
            if dim in self.categorical_variables:
                K = CategoricalKernel().K(X=self.X[:, dim].reshape(-1, 1), X2=None)
            elif dim in self.linear_variables:
                K = LinearKernel().K(X=self.X[:, dim].reshape(-1, 1), X2=None)
            else:
                K = SSkernel().K(X=self.X[:, dim].reshape(-1, 1), X2=None)
            self.kernel_matrices.append(K)
        self.selected_dims = []  # all dims including active and inactive indexes with order <= max_interaction_depth
        for ii in range(self.max_interaction_depth + 1):
            if ii > 0:
                tmp = [
                    list(tup) for tup in itertools.combinations(range(self.p), ii)
                ]
                self.selected_dims = self.selected_dims + tmp
        self.active_dims = self.selected_dims.copy() if active_dims is None else active_dims
        self.individual_variances, self.interaction_variances = None, None

        self.mu, self.c = 0.0, None
        self.lambda1 = 0.001 / self.N
        self.lambda2 = None


    def compute_covariance(self, kernel_matrices: Sequence[tf.Tensor], r: NDArray) -> tf.Tensor:
        """
        Args:
            kernel_matrices:  Sequence of one-dimensional kernel matrices [K_1, K_2, ..., K_p].
            r: kernel variances [r_1, ..., r_p, r_ij, j=i+1, ..., p, i=1, ..., p-1]
        Returns:
            A covariance matrix:
            K = Σ r_j * K_j + Σ r_ij * (K_i * K_j) + ... for nonzero r_j and r_ij.
        """
        # Precompute lookup: active_dim -> variance index
        dim_to_var_index = {tuple(dim): i for i, dim in enumerate(self.selected_dims)}
        additive_terms = []
        for active_dim in self.active_dims:
            var_idx = dim_to_var_index[tuple(active_dim)]
            term = r[var_idx] * tf.reduce_prod(
                [kernel_matrices[i] for i in active_dim], axis=0
            )
            additive_terms.append(term)
        return tf.add_n(additive_terms) if additive_terms else tf.zeros_like(kernel_matrices[0])

    def smoother_matrix(self, K: tf.Tensor, loglambda1: float):
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
        lambda1 = tf.exp(tf.convert_to_tensor(loglambda1, dtype=dtype))
        # Step 1: Compute the smoothing matrix fn = A(lambda) yn
        Q = tf.add(K, n * lambda1 * tf.eye(n, dtype=dtype))
        Q_inv = tf.linalg.inv(Q)
        J = tf.ones((n, 1), dtype=dtype)
        alpha = tf.matmul(J, Q_inv, transpose_a=True) / tf.squeeze(tf.matmul(J, tf.matmul(Q_inv, J), transpose_a=True))
        A = tf.matmul(tf.matmul(K, Q_inv), tf.eye(n, dtype=dtype) - tf.matmul(J, alpha)) + tf.matmul(J, alpha)
        return A

    def update_mu_and_c(self, lambda1):
        """
            For fixed r and lambda1, solve
                [K + n*lambda_1 I   1] [c] = [y]
                [1^T               0] [mu]   [0]
            Args:
                lambda1: float or tf scalar, smoothing parameter
            Returns:
                mu:  Python float (converted from TF scalar)
                c:   (n,1) tf.Tensor (same dtype as K)
            """
        r = np.array(self.individual_variances + self.interaction_variances)
        K = self.compute_covariance(self.kernel_matrices, r) # (N,N)
        Q = tf.add(K, tf.cast(self.N * lambda1, K.dtype) * tf.eye(self.N, dtype=K.dtype))
        L = tf.linalg.cholesky(Q)
        # Solve L v = ones  and L w = y  (forward solves)
        rhs = tf.concat([
            tf.ones((self.N, 1), dtype=Q.dtype),
            self.Y_scaled
        ], axis=1)

        V = tf.linalg.triangular_solve(L, rhs)      # shape (N, 2), single triangular solve (forward substitution)
        v1, vy = V[:, 0:1], V[:, 1:2]   # Extract the two solved vectors
        # Compute mu = (1^T Q^{-1} Y) / (1^T Q^{-1} 1)
        numerator = tf.matmul(v1, vy, transpose_a=True)  # scalar [[x]]
        denominator = tf.matmul(v1, v1, transpose_a=True)  # scalar [[y]]
        mu = tf.squeeze(numerator / denominator).numpy().item()
        vc = tf.linalg.solve(L, self.Y_scaled - mu)
        c = tf.linalg.solve(tf.transpose(L), vc)
        return mu, c

    def tuning_lambda1(self, K: tf.Tensor, cv: str = 'gcv', bounds=(1e-6, 0.2)):
        def objective(loglambda1:float, cv: str):
            if cv == 'bic':
                fun =  self.bic_score(K, loglambda1).numpy()
            elif cv == 'gml':
                fun = self.gml_score(K, loglambda1).numpy()
            else:
                fun = self.gcv_score(K, loglambda1).numpy()
            return fun

        objective_opt = partial(objective, cv=cv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Omit the warning message.
            res = minimize_scalar(objective_opt, bounds=(np.log(bounds[0]/self.N), np.log(bounds[1]/self.N)), method='bounded')
        loglambda1_opt = res.x
        lambda1_opt = np.exp(loglambda1_opt)
        return lambda1_opt

    def Gram(self, kernel_matrices, active_dims, c) -> tf.Tensor:
        """
        Compute the Gram matrix of the kernel matrices.
        Args:
            kernel_matrices: a list [K_1, ..., K_p]
            active_dims: only consider the q active dims for reducing the computation burden
            c: coefficients: f= mu + k^T * c
        Returns:
            the n* q Gram matrix: G = [Kj c, ..., K_ij c], i.e., its column is given by K_j c
        """
        additive_kernels = [
            tf.reduce_prod([kernel_matrices[i] for i in active_dim], axis=0)
            for active_dim in active_dims
        ]
        G_cols = [K @ c for K in additive_kernels]  # Compute columns G_j = K_j @ alpha
        G = tf.concat(G_cols, axis=1)  # Concatenate into (n,q)
        return G

    def solve_weighted_lasso_sklearn(self, G, nonzero_r_tilde, lambda2, delta, omega):
        """
        Fit nonnegative Lasso using sklearn.Lasso with alpha (sklearn alpha).
        alpha corresponds to sklearn's alpha (objective (1/(2n))||.||^2 + alpha ||r||_1).
        Returns numpy array r (p,).
        """
        n, q = G.shape
        G = np.asarray(G)  # shape (n,p)
        z = self.Y_scaled - self.mu - .5 * n * self.lambda1 * self.c.numpy()
        z = np.asarray(z).reshape(-1, )  # shape (n,)
        w = 2 / omega + ((n + 2) * omega - 2) / omega * np.array(
            [1 / delta if 0 <= r <= delta else 1 / r for r in nonzero_r_tilde])
        w = w.reshape(-1, ).astype(float)
        X = G / w[np.newaxis, :]  # broadcasting: (n,p) / (p,) -> (n,p)
        # p = G.shape[1]
        alpha = float(lambda2) / 2.0
        model = Lasso(alpha=float(alpha), fit_intercept=False, positive=True,
                      max_iter=200, tol=1e-4, warm_start=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Omit the warning message.
            model.fit(X, z)
        beta_hat = model.coef_
        r_hat = beta_hat / w
        return r_hat

    def tuning_lambda2(self, cv, delta, omega, lambda1, lambda2_grid_number=20):
        """
        Grid-search over log(lambda2), evaluate approximate GCV using df = #nonzero(r).
        G: (n,p) numpy array
        Returns: dict with best_lambda2, best_loglam, best_r, best_gcv, rss, df
        """
        # r_tilde = np.ones(len(self.selected_dims), )

        # if omega > 2/(self.N+2):
        #     lam_opt = np.power(self.N, -1.8) / np.log(self.N)
        #     loglam_grid = np.linspace(np.log(.001 * lam_opt), np.log(1 *lam_opt), 20)
        # else:
        #     loglam_grid = np.linspace(np.log(1e-7), np.log(1e-3), 20)

        # loglam_grid = np.linspace(self.log_lambda2_min, self.log_lambda2_max, 20)
        # loglam_grid = np.linspace(np.log(1e-7), np.log(1e-3), 30)

        active_dims = self.active_dims
        G = self.Gram(kernel_matrices=self.kernel_matrices, active_dims=active_dims, c=self.c)
        r_tilde = np.array(self.individual_variances + self.interaction_variances)
        non_zero_idx = np.array([self.selected_dims.index(active_dim) for active_dim in active_dims])
        nonzero_r_tilde = r_tilde[non_zero_idx]
        current_step_size = (self.log_lambda2_max - self.log_lambda2_min) / (lambda2_grid_number - 1)

        best = None
        best_info = dict(lambda2=self.lambda2, r=r_tilde, cv_value=np.inf)

        for loglam in self.log_lambda2_grid:
            lambda2 = float(np.exp(loglam))
            nonzero_r_tilde = self.solve_weighted_lasso_sklearn(G=G, nonzero_r_tilde=nonzero_r_tilde,lambda2=lambda2,delta=delta,omega=omega)
            K = self.compute_covariance(self.kernel_matrices, r_tilde)
            if cv == 'gml':
                fun = self.gcv_score(K, np.log(lambda1))
            elif cv == 'bic':
                fun = self.bic_score(K, np.log(lambda1))
            else:
                fun = self.gcv_score(K, np.log(lambda1))
            r_tilde[non_zero_idx] = nonzero_r_tilde  # warm start next grid point
            if best is None or fun < best:
                best = fun
                best_info = dict(lambda2=lambda2, r=r_tilde, cv_value=fun)

        # # 获取当前最优解的对数位置
        # best_log_val = np.log(best_info['lambda2'])
        # # 计算当前网格的步长 (对数尺度下的距离)
        # # 将下一次的搜索区间缩小到当前最优解的左右各 'zoom_factor' 个步长范围内
        # # 设为 1.5 是为了确保新的区间能覆盖相邻的网格点，防止陷入由于网格太稀疏导致的假最优点
        # # current_step_size = current_step_size / 2
        # zoom_factor = 16
        # log_lambda2_min = best_log_val - (current_step_size * zoom_factor)
        # log_lambda2_max = best_log_val + (current_step_size * zoom_factor)
        # # 边界保护：防止搜索区间向外扩展得太离谱（例如不允许 lambda2 大于 10^3）
        # log_lambda2_min = max(self.log_lambda2_min,  log_lambda2_min)
        # log_lambda2_max = min(self.log_lambda2_max, log_lambda2_max)
        # self.log_lambda2_grid = np.linspace(log_lambda2_min, log_lambda2_max, lambda2_grid_number)
        return best_info

    def step1(self, tuning_lambda1=True, lambda1_cv='gml'):
        """
        Solve for (mu, c) for a fixed r, while the smoothing parameter lambda1 can be selected via minimizing GML
        """
        if tuning_lambda1:
            r = np.array(self.individual_variances + self.interaction_variances).copy()
            K = self.compute_covariance(self.kernel_matrices, r)
            self.lambda1 = self.tuning_lambda1(K=K, cv=lambda1_cv)
        mu, c = self.update_mu_and_c(lambda1=self.lambda1)
        return mu, c

    def step2(self, delta, omega, tuning_lambda2=True, lambda2_cv='gcv', lambda2_grid_number=20):
        """
        Solve for r for a fixed (mu, c), while tuning M via GCV / BIC / GML
        """
        if tuning_lambda2:
            best_info = self.tuning_lambda2(cv=lambda2_cv, delta=delta, omega=omega, lambda1=self.lambda1, lambda2_grid_number=lambda2_grid_number)
            self.lambda2 = best_info['lambda2']
            r_best = best_info['r']
        else:
            G = self.Gram(kernel_matrices=self.kernel_matrices, active_dims=self.active_dims, c=self.c)
            r_tilde = np.array(self.individual_variances + self.interaction_variances)
            r_best = r_tilde.copy()
            non_zero_idx = np.array([self.selected_dims.index(active_dim) for active_dim in self.active_dims])
            nonzero_r_tilde = r_tilde[non_zero_idx]
            nonzero_r_tilde = self.solve_weighted_lasso_sklearn(G=G, nonzero_r_tilde=nonzero_r_tilde, lambda2=self.lambda2,
                                                                delta=delta, omega=omega)
            r_best[non_zero_idx] = nonzero_r_tilde
        return r_best

    def fit(self,
            delta=0.001,
            omega=None,
            gamma=1e-4,
            max_iter=200,
            r_thres=1e-6,
            cut_off_step=10,
            verbose=False,
            lambda1_cv='gml',
            lambda2_cv = 'gcv',
            lambda2_max=None,
            lambda2_grid_number=20
            ):
        """
        The iterative fitting procedure.

        Args:
            delta: the linearization parameter.
            omega: the regularization parameter.
            gamma: the stopping-rule parameter.
            max_iter: maximum number of iterations.
            r_thres: a thresholding value to reduce the active dims during iterations.
            cut_off_step: threshold r when iter > cut_off_steps.
            verbose: whether to print the results_0.1 during iterations.
            lambda1_cv: the criterion for tuning lambda1 by GCV/BIC/GML ('gcv'/'bic'/'gml').
            lambda2_cv: the criterion for tuning lambda2 by GCV/BIC/GML ('gcv'/'bic'/'gml').
            lambda2_max: interval (lambda2_min, lambda2_max) for tuning lambda2
            lambda2_min: interval (lambda2_min, lambda2_max) for tuning lambda2
            lambda2_grid_number: the number of even grids in (lambda2_min, lambda2_max)
        Returns:
            The estimated (mu, c) and r.
        """
        if omega is None:
            omega = 10 / self.N
        mu1 = 0.0
        c0, c1 = np.zeros((self.N, 1)), np.ones((self.N, 1))
        self.individual_variances = [1.0] * self.p  # variances for main effects
        self.interaction_variances = [1.0] * (len(self.selected_dims) - self.p)  # variance for interactions
        r1 = np.copy(self.individual_variances + self.interaction_variances)
        t_start = time.time() # time recorder
        iter = 1
        max_change_of_param = np.inf
        tuning_lambda1 = True
        tuning_lambda2 = True

        if lambda2_max is None:
            lambda2_max = 0.01/self.N
        lambda2_min = lambda2_max * 1e-3

        self.log_lambda2_max = np.log(lambda2_max)
        self.log_lambda2_min = np.log(lambda2_min)
        self.log_lambda2_grid = np.linspace(self.log_lambda2_min, self.log_lambda2_max, lambda2_grid_number)
        while max_change_of_param > gamma and iter <= max_iter:
            self.mu, self.c = self.step1(tuning_lambda1=tuning_lambda1, lambda1_cv=lambda1_cv)
            r_best = self.step2(delta=delta, omega=omega, tuning_lambda2=tuning_lambda2, lambda2_cv=lambda2_cv, lambda2_grid_number=lambda2_grid_number)

            if iter>cut_off_step:
                # tuning_lambda1 = False
                # tuning_lambda2 = False
                r_best = np.where(r_best > r_thres, r_best, 0.0)
                self.active_dims = [self.selected_dims[i] for i in range(len(self.selected_dims)) if
                                    float(r_best[i]) > r_thres]

            self.individual_variances = [float(r_best[i]) for i in range(self.p)]
            if self.max_interaction_depth > 1:
                self.interaction_variances = [float(r_best[i]) for i in range(self.p, len(self.selected_dims))]

            mu0, c0, r0 = mu1, c1.copy(), r1.copy()
            mu1, c1 = self.mu, np.copy(self.c.numpy())
            r0 = r1.copy()
            r1 = np.copy(self.individual_variances + self.interaction_variances)

            # change of parameter
            mu_change = np.sqrt(np.mean((mu1 - mu0)**2))/ (np.sqrt(np.mean(mu0**2)) + 1e-6)
            c_change = np.sqrt(np.mean((c1 - c0)**2)) / (np.sqrt(np.mean(c0**2)) + 1e-6)
            r_change = np.sqrt(np.mean((r1 - r0)**2)) / (np.sqrt(np.mean(r0**2)) + 1e-6)
            max_change_of_param = max(mu_change, c_change, r_change)

            if verbose:
                print(f"{iter}-th updating: individual variances = {self.individual_variances}." + "\n")
                if self.max_interaction_depth > 1:
                    print(f"{iter}-th updating: interaction variances = {self.interaction_variances}." + "\n")
                print(f"{iter}- the best n*lambda1 = {self.N * self.lambda1}, best lambda2 = {self.lambda2}." + "\n")
                print(f"{iter}-th iterative maximum change of parameters (scaled) = {max_change_of_param}." + "\n")

            iter += 1

        fitting_time_seconds = time.time() - t_start
        print(f"SSANOVA(Algorithm 2) done: time = {fitting_time_seconds:.3f} seconds, change of parameters = {max_change_of_param}. \n")

    def predict(self, Xnew: NDArray):
        """
        Args:
            Xnew: inputs to predict the response on
        Returns:
            predicted response on input Xnew
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
        r = np.array(self.individual_variances + self.interaction_variances)
        k_mat_new = self.compute_covariance(kernel_matrices, r)
        f_pred = self.mu + tf.linalg.matmul(k_mat_new, self.c, transpose_a=True).numpy()[:, 0]
        return self.Y_mean + self.Y_std * f_pred

    def selected_main_effects(self):
        return [i for i in self.active_dims if len(i) == 1]

    def selected_interactions(self):
        return [i for i in self.active_dims if len(i) > 1]


    def predict_by_component(self, Xnew: NDArray):
        r = self.individual_variances + self.interaction_variances
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
        for dim in self.selected_dims:
            var_idx = dim_to_var_index[tuple(dim)]
            term = r[var_idx] * tf.reduce_prod(
                [kernel_matrices[i] for i in dim], axis=0
            )
            f_pred = tf.linalg.matmul(term, self.c, transpose_a=True)
            ncurves[:, var_idx] = tf.reshape(f_pred, [-1]).numpy()
        return self.Y_std * ncurves

    def gcv_score(self, K: tf.Tensor, loglambda1):
        """
                Compute the GCV score (TF scalar) for a given log(lambda1).
        """
        # Step 1: Compute A(lambda1)
        A = self.smoother_matrix(K, loglambda1)
        # Step 2: Compute I - A(lambda1)
        n = K.shape[0]
        dtype = K.dtype
        I_minus_A = tf.eye(n, dtype=dtype) - A
        # Step 3: Compute the numerator: (1/n) *  || (I - A(lambda1)) * y||^2
        y = tf.convert_to_tensor(self.Y_scaled, dtype=K.dtype)
        numerator = tf.squeeze(tf.matmul(tf.matmul(I_minus_A, y), tf.matmul(I_minus_A, y), transpose_a=True)) / n
        # Step 4: Compute the denominator: tr(I - A(lambda1))
        denominator = (tf.linalg.trace(I_minus_A) / n) ** 2
        # Step 5: Compute the GML score
        gcv = numerator / denominator
        return gcv

    def gml_score(self, K: tf.Tensor, loglambda1) -> tf.Tensor:
        """
                Compute the GML score (TF scalar) for a given log(lambda1).
        """
        n = K.shape[0]  # Number of data points
        dtype = K.dtype
        # Step 1: Compute the smoothing matrix fn = A(lambda) yn
        A = self.smoother_matrix(K, loglambda1)
        # Step 2: Compute I - A(lambda)
        I_minus_A = tf.eye(n, dtype=dtype) - A
        # Step 3: Compute the numerator: (1/n) * y^T (I - A(lambda)) y
        y = tf.convert_to_tensor(self.Y_scaled, dtype=K.dtype)
        numerator = tf.squeeze(tf.matmul(y, tf.matmul(I_minus_A, y), transpose_a=True)) / n
        # Step 4: Compute the denominator: (det^(+) (I - A(lambda)))^(1/n_1) # n_1 is number of nonzero eigenvalues
        eigenvalues = tf.linalg.eigvalsh(I_minus_A)  # Eigenvalues of (I - A)
        non_zero_eigenvalues = tf.boolean_mask(eigenvalues, eigenvalues > 1e-12)  # Avoid zero eigenvalues
        eigenvalue_product = tf.reduce_prod(non_zero_eigenvalues)  # Product of non-zero eigenvalues
        denominator = tf.pow(eigenvalue_product, 1.0 / non_zero_eigenvalues.shape[0])
        # Step 5: Compute the GML score
        gml = numerator / denominator
        return gml

    def bic_score(self, K: tf.Tensor, loglambda1):
        """
                Compute the BIC score (TF scalar) for a given log(lambda1).
        """
        # Step 1: Compute A(lambda1)
        A = self.smoother_matrix(K, loglambda1)
        # Step 2: Compute I - A(lambda1)
        n = K.shape[0]
        dtype = K.dtype
        I_minus_A = tf.eye(n, dtype=dtype) - A
        # Step 3: Compute the numerator: (1/n) *  || (I - A(lambda1)) * y||^2
        y = tf.convert_to_tensor(self.Y_scaled, dtype=K.dtype)
        numerator = tf.squeeze(tf.matmul(tf.matmul(I_minus_A, y), tf.matmul(I_minus_A, y), transpose_a=True)) / n
        # Step 4: Compute the df: tr(A(lambda1))
        df = tf.linalg.trace(A)
        # Step 5: Compute the BIC score
        bic = n * np.log(numerator) + df * np.log(n)
        return bic
