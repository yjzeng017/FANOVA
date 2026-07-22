import numpy as np
import pickle
import random
import os
from simulation.utils import make_orthogonal_uniform_measure, hypercube_design_uniform


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


def file_name(N_train, p):
        return 'Uniform_N_train_{0}_p_{1}_100_replications.pkl'.format(N_train, p)



# User must specify the root path.
root_path = '/Users/yjzeng/Desktop/FANOVA/github_code/'

# The path 'root_path + 'simulation/interaction/data/' ' is used to store the simulation data.
# Please ensure the dictionary '/data' exists.
isExists = os.path.exists(root_path+'simulation/interaction/data/')
if not isExists:
    os.makedirs(root_path +'simulation/interaction/data/')



n_replicates = 100


if __name__ == '__main__':
    seed_value = 2025
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    block_functions = make_orthogonal_test_functions()
    N_test = 1000
    noise_sd = 0.2546
    for p in [10]:
        for N_train in  [100, 200, 400]:
            X_train_100_replicated = []
            Y_train_100_replicated = []
            X_test_100_replicated = []
            Y_test_100_replicated = []
            for i in range(n_replicates):
                X_train = hypercube_design_uniform(N_train, p, 0, 1)
                X_test = hypercube_design_uniform(N_test, p, 0, 1)
                Y_train_noiseless = block_functions['g1'](X_train[:, 0]) + block_functions['g2'](
                    X_train[:, 1]) + block_functions['g3'](X_train[:, 2]) + block_functions['g4'](X_train[:, 3]) + \
                                    block_functions['g3'](X_train[:, 0] * X_train[:, 1]) + \
                                    block_functions['g2']((X_train[:, 0] + X_train[:, 2])/2) + block_functions['g1'](X_train[:, 2] * X_train[:, 3])

                Y_test = block_functions['g1'](X_test[:, 0]) + block_functions['g2'](X_test[:, 1]) + \
                         block_functions['g3'](X_test[:, 2]) + block_functions['g4'](X_test[:, 3]) + \
                         block_functions['g3'](X_test[:, 0] * X_test[:, 1]) + \
                         block_functions['g2']((X_test[:, 0] + X_test[:, 2]) / 2) + block_functions['g1'](
                    X_test[:, 2] * X_test[:, 3])


                Y_train = Y_train_noiseless + noise_sd * np.random.normal(size=(N_train,))

                X_train_100_replicated.append(X_train)
                X_test_100_replicated.append(X_test)
                Y_train_100_replicated.append(Y_train)
                Y_test_100_replicated.append(Y_test)


            pickle.dump(X_train_100_replicated, open(root_path + 'simulation/interaction/data/X_train_' + file_name(N_train, p), 'wb'))
            pickle.dump(X_test_100_replicated, open(root_path + 'simulation/interaction/data/X_test_' + file_name(N_train, p), 'wb'))
            pickle.dump(Y_train_100_replicated, open(root_path + 'simulation/interaction/data/Y_train_' + file_name(N_train, p), 'wb'))
            pickle.dump(Y_test_100_replicated, open(root_path + 'simulation/interaction/data/Y_test_' + file_name(N_train, p), 'wb'))

