import numpy as np
from FANOVAModels.GPANOVA import GPANOVAModel
from FANOVAModels.SSANOVA import SSANOVAModel
import pickle
import time
from functools import partial
import multiprocessing
from tqdm import tqdm
import contextlib
import io



def file_name(cov, param):
    if cov == 'CompSymm':
        return '{0}_t_{1}_100_replications.pkl'.format(
             cov, param)
    if cov == 'AR(1)':
        return '{0}_rho_{1}_100_replications.pkl'.format(
            cov, param)


# User must specify the root path.
# The simulation data is used to stored in "root_path + 'simulation/additive/data/' ".


root_path = '/Users/yjzeng/Desktop/FANOVA/github_code/'




def one_fit(i, method, X_train, Y_train, X_test, Y_test):
    X = X_train[i]
    Y = Y_train[i]
    Xt = X_test[i]
    Yt = Y_test[i]
    N_train = X.shape[0]
    p = X.shape[1]
    t_start = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        if method == 'GPANOVA':
            model = GPANOVAModel(
                X=X,
                Y=Y.reshape(N_train, 1),
                max_interaction_depth=1,
            )
            model.fit(omega=10/(N_train+2))
        else:
            model = SSANOVAModel(
                X=X,
                Y=Y.reshape(N_train, 1),
                max_interaction_depth=1,
            )
            model.fit(
                delta=.001/np.log(N_train),
                omega=10/(N_train+2), lambda2_max=np.pow(N_train, -1.8)/np.log(N_train), max_iter=200,verbose=True)

    ISE = np.mean((Yt - model.predict(Xt)) ** 2)
    runtime = time.time() - t_start
    variable_freq = [1 if [i] in model.selected_main_effects() else 0 for i in range(p)]
    model_size= [1 if (i+1) == len(model.selected_main_effects()) else 0 for i in range(p)]
    # print(f"{method} fitting - {i}th replication finished.")
    return {
        'ISE': ISE,
        'model_size': model_size,
        'variable_freq':variable_freq,
        'runtime': runtime
    }


n_cores = 8
n_replications = 100 # User can run less replicated simulations for testing.


if __name__ == '__main__':
    p = 10
    # N_train = 100
    # N_test = 1000
    covs = ["CompSymm", "AR(1)"]
    ts = [0, 1, 3]
    rhos = [-0.5, 0, 0.5]
    f = open(root_path + 'simulation/additive/proposed_result.txt', 'w')
    f.write("##########################################################\n")
    f.write(" Simulation: additive model \n")
    f.write("##########################################################\n")
    f.write('\n')

    for cov in covs:
        for params in zip(ts, rhos):
            if cov == 'CompSymm':
                param = params[0]
            else:
                param = params[1]

            X_train = pickle.load(open(root_path+ 'simulation/additive/data/X_train_' + file_name(cov, param), 'rb'))
            X_test = pickle.load(open(root_path + 'simulation/additive/data/X_test_' + file_name(cov, param), 'rb'))
            Y_train = pickle.load(open(root_path + 'simulation/additive/data/Y_train_' + file_name(cov, param), 'rb'))
            Y_test = pickle.load(open(root_path + 'simulation/additive/data/Y_test_' + file_name(cov, param), 'rb'))
            for method in ['GPANOVA', 'SSANOVA']:
                print(f"Method = {method}, cov={cov}, t or rho ={param}")
                partial_setting = partial(one_fit, method=method, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
                with multiprocessing.Pool(processes=n_cores) as pool:
                   result = list(tqdm(pool.imap_unordered(partial_setting, range(n_replications)), total=n_replications))

                ISE = np.array([[value for key, value in c.items() if key == 'ISE'] for c in result])
                runtime = np.array([[value for key, value in c.items() if key == 'runtime'] for c in result])
                variable_freq = np.array([[value for key, value in c.items() if key == 'variable_freq'] for c in result])
                model_size = np.array([[value for key, value in c.items() if key == 'model_size'] for c in result])

                f.write("covariate structure: {0}, t or rho: {1} \n".format(cov, param))
                f.write("----------------------------------------------" + '\n')
                f.write('Method: ' + method + '\n')
                f.write('Mean ISE: ' + str('%.3f' % np.array(ISE).mean()) + '(' + str('%.3f' % (
                        np.array(ISE).std()/np.sqrt(n_replications))) + ')' + '   Median ISE: ' + str('%.3f' % np.nanmedian(ISE)) +'\n')
                f.write('Varible ' + ' & '.join(str(i) for i in range(1, p+1))+ '\n')
                f.write('Frequency '  + ' & '.join(str(i) for i in np.array(variable_freq).sum(axis=0).flatten()) + '\n')
                f.write('Model size ' + ' & '.join(str(i) for i in range(1, p + 1)) + '\n')
                f.write('Frequency ' + ' & '.join(str(i) for i in np.array(model_size).sum(axis=0).flatten()) + '\n')
                f.write("----------------------------------------------" + '\n')
                f.write('\n')
    f.close()
