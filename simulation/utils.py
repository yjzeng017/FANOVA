import numpy as np
import scipy.integrate as integrate


def hypercube_design_uniform(N, M, lower=0, upper=1):
    return np.random.uniform(lower, upper, size=(N,M))


def hypercube_design_normal(N, M, loc=0, scale=1):
    return np.random.normal(loc, scale, size=(N,M))


def synthetic_file_path(root, N_train, N_test, p):
    return '{0}/example_1_train_size_{1}_test_size_{2}_p_{3}.pkl'.format(
        root, N_train, N_test, p)


def make_orthogonal_uniform_measure(f, lower=0, upper=1):
    unif_measure = 1 / (upper - lower)
    f_mean = integrate.quad(f, lower, upper)[0] * unif_measure
    f_var = integrate.quad(lambda x: (f(x) - f_mean) ** 2, lower, upper)[0] * unif_measure
    return lambda x: (f(x) - f_mean) / np.sqrt(f_var)



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