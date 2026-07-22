import numpy as np
import pickle
import random
import os
from simulation.utils import make_orthogonal_uniform_measure, hypercube_design_uniform, hypercube_design_normal


def make_orthogonal_test_functions(orthogonal=False, lower=0, upper=1):
    test_functions = {'g1': lambda t: t,
                      'g2': lambda t: (2*t - 1) ** 2,
                      'g3': lambda t: np.sin(2*np.pi*t)/(2 - np.sin(2*np.pi*t)),
                      'g4': lambda t: 0.1*np.sin(2*np.pi*t) + 0.2*np.cos(2*np.pi*t) + 0.3*np.sin(2*np.pi*t)**2 + 0.4*np.cos(2*np.pi*t)**3 + 0.5*np.sin(2*np.pi*t)**3,
                      }

    if not orthogonal:
        test_functions_orthogonalized = test_functions
    else:
        test_functions_orthogonalized = {}
        for name in test_functions.keys():
            test_functions_orthogonalized[name] = make_orthogonal_uniform_measure(
                test_functions[name], lower=lower, upper=upper)

    return test_functions_orthogonalized


def file_name(cov, param):
    if cov == 'CompSymm':
        return '{0}_t_{1}_100_replications.pkl'.format(
             cov, param)
    if cov == 'AR(1)':
        return '{0}_rho_{1}_100_replications.pkl'.format(
            cov, param)


# User must specify the root path.
root_path = '/Users/yjzeng/Desktop/FANOVA/github_code/'

# The path 'root_path + 'simulation/additive/data/' ' is used to store the simulation data.
# Please ensure the dictionary '/data' exists.
isExists = os.path.exists(root_path+'simulation/additive/data/')
if not isExists:
    os.makedirs(root_path +'simulation/additive/data/')


if __name__ == '__main__':
    seed_value = 2025
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    block_functions = make_orthogonal_test_functions()
    N_train = 100
    N_test = 1000
    p = 10
    noise_sd = np.sqrt(3.03)
    covs = ['CompSymm', 'AR(1)']
    ts = [0, 1, 3]
    rhos = [-0.5, 0, 0.5]
    for cov in covs:
        if cov == 'CompSymm':
            for t in ts:
                X_train_100_replicated = []
                Y_train_100_replicated = []
                X_test_100_replicated = []
                Y_test_100_replicated = []
                for i in range(100):
                    W_train = hypercube_design_uniform(N_train, p, 0, 1)
                    W_test = hypercube_design_uniform(N_test, p, 0, 1)
                    U_train = hypercube_design_uniform(N_train, 1, 0, 1)
                    U_test = hypercube_design_uniform(N_test, 1, 0, 1)
                    X_train = (W_train + t * U_train) / (1 + t)
                    X_test = (W_test + t * U_test) / (1 + t)
                    Y_train_noiseless = 5 * block_functions['g1'](X_train[:, 0]) + 3 * block_functions['g2'](
                        X_train[:, 1]) + 4 * block_functions['g3'](X_train[:, 2]) + 6 * block_functions['g4'](X_train[:, 3])
                    Y_test = 5 * block_functions['g1'](X_test[:, 0]) + 3 * block_functions['g2'](X_test[:, 1]) + 4 * \
                             block_functions['g3'](X_test[:, 2]) + 6 * block_functions['g4'](X_test[:, 3])
                    Y_train = Y_train_noiseless + noise_sd * np.random.normal(size=(N_train,))

                    X_train_100_replicated.append(X_train)
                    X_test_100_replicated.append(X_test)
                    Y_train_100_replicated.append(Y_train)
                    Y_test_100_replicated.append(Y_test)

                pickle.dump(X_train_100_replicated, open(root_path + 'simulation/additive/data/X_train_' + file_name(cov, t), 'wb'))
                pickle.dump(X_test_100_replicated, open(root_path + 'simulation/additive/data/X_test_' + file_name(cov, t), 'wb'))
                pickle.dump(Y_train_100_replicated, open(root_path + 'simulation/additive/data/Y_train_' + file_name(cov, t), 'wb'))
                pickle.dump(Y_test_100_replicated, open(root_path + 'simulation/additive/data/Y_test_' + file_name(cov, t), 'wb'))


        if cov == 'AR(1)':
            for rho in rhos:
                X_train_100_replicated = []
                Y_train_100_replicated = []
                X_test_100_replicated = []
                Y_test_100_replicated = []
                for i in range(100):
                    W_train = hypercube_design_normal(N_train, p, 0, 1)
                    W_test = hypercube_design_normal(N_test, p, 0, 1)
                    X_train_orig = np.zeros((N_train, p))
                    X_test_orig = np.zeros((N_test, p))
                    for j in range(p):
                        if j == 0:
                            X_train_orig[:, j] = W_train[:, j]
                            X_test_orig[:, j] = W_test[:, j]
                        else:
                            X_train_orig[:, j] = rho * X_train_orig[:, j - 1] + np.sqrt(1 - rho ** 2) * W_train[:, j]
                            X_test_orig[:, j] = rho * X_test_orig[:, j - 1] + np.sqrt(1 - rho ** 2) * W_test[:, j]
                    X_train = (np.clip(X_train_orig, -2.5, 2.5) + 2.5) / 5
                    X_test = (np.clip(X_test_orig, -2.5, 2.5) + 2.5) / 5
                    Y_train_noiseless = 5 * block_functions['g1'](X_train[:, 0]) + 3 * block_functions['g2'](
                        X_train[:, 1]) + 4 * block_functions['g3'](X_train[:, 2]) + 6 * block_functions['g4'](X_train[:, 3])
                    Y_test = 5 * block_functions['g1'](X_test[:, 0]) + 3 * block_functions['g2'](X_test[:, 1]) + 4 * \
                             block_functions['g3'](X_test[:, 2]) + 6 * block_functions['g4'](X_test[:, 3])
                    Y_train = Y_train_noiseless + noise_sd * np.random.normal(size=(N_train,))

                    X_train_100_replicated.append(X_train)
                    X_test_100_replicated.append(X_test)
                    Y_train_100_replicated.append(Y_train)
                    Y_test_100_replicated.append(Y_test)

                pickle.dump(X_train_100_replicated, open(root_path + 'simulation/additive/data/X_train_' + file_name(cov, rho), 'wb'))
                pickle.dump(X_test_100_replicated, open(root_path + 'simulation/additive/data/X_test_' + file_name(cov, rho), 'wb'))
                pickle.dump(Y_train_100_replicated, open(root_path + 'simulation/additive/data/Y_train_' + file_name(cov, rho), 'wb'))
                pickle.dump(Y_test_100_replicated, open(root_path + 'simulation/additive/data/Y_test_' + file_name(cov, rho), 'wb'))


