import time
import numpy as np
import pickle
import multiprocessing
from functools import partial
from tqdm import tqdm
from FANOVAModels.GPANOVA import GPANOVAModel
from FANOVAModels.SSANOVA import SSANOVAModel
import contextlib
import io

def file_name(N_train, p):
    return 'Uniform_N_train_{0}_p_{1}_100_replications.pkl'.format(N_train, p)




# User must specify the root path.
# The simulation data is used to stored in " root_path + 'simulation/interaction/data/' ".

root_path = '/Users/yjzeng/Desktop/FANOVA/github_code/'


def one_fit(i, method, X_train, Y_train, X_test, Y_test):
    X = X_train[i]
    Y = Y_train[i]
    Xt = X_test[i]
    Yt = Y_test[i]
    N_train = X.shape[0]
    t_start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        if method == 'GPANOVA':
            model = GPANOVAModel(
                X=X,
                Y=Y.reshape(N_train, 1),
                max_interaction_depth=2,
            )
            model.fit(omega=10/N_train)
        else:
            model = SSANOVAModel(X=X,
                                 Y=Y.reshape(N_train, 1),
                                 max_interaction_depth=2,
                               )
            model.fit(delta=.001/np.log(N_train), omega=10/N_train)
    ISE = np.mean((Yt - model.predict(Xt)) ** 2)
    runtime = time.time() - t_start
    selected_effects = model.selected_main_effects() + model.selected_interactions()
    model_size = len(selected_effects)
    effect_freq = [1 if dim in selected_effects else 0 for dim in model.selected_dims]
    true_effects = [[0], [1], [2], [3], [0, 1], [0, 2], [2, 3]]

    true_set = set(tuple(sorted(effect)) for effect in true_effects)
    selected_set = set(tuple(sorted(effect)) for effect in selected_effects)
    total_possible_effects = 55

    TP = len(true_set.intersection(selected_set))
    FP = len(selected_set.difference(true_set))
    FN = len(true_set.difference(selected_set))

    P = len(true_set)
    N = total_possible_effects - P

    # TN = N - FP

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    FDR = 1.0 - Precision  # (False Discovery Rate)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # (True Positive Rate) / Recall

    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 0.0
    if (Precision + Recall) > 0:
        f1_score = 2 * (Precision * Recall) / (Precision + Recall)

    FPR = FP / N if N > 0 else 0.0  #  (False Positive Rate)
    FNR = FN / P if P > 0 else 0.0  # (False Negative Rate)

    return {
        'ISE':ISE,
        'runtime': round(runtime, 3),
        'model_size': model_size,
        'effect_freq': effect_freq,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        'FPR': FPR,
        'FNR': FNR,
        'FDR': FDR,
        'TPR': TPR,
        "Precision": Precision,
        "Recall": Recall,
        "F1": f1_score
    }



repetitions = 100 # User could run fewer simulations for testing.
n_cores = 8


if __name__ == '__main__':
    # cores = 5
    f = open(root_path + 'simulation/interaction/proposed_results.txt', 'w')
    f.write("##########################################################\n")
    f.write(" Simulation: interaction model \n")
    f.write("##########################################################\n")
    f.write('\n')

    for p in [10]:
        for N_train in [100, 200, 400]:
            X_train = pickle.load(open(root_path + 'simulation/interaction/data/X_train_' + file_name(N_train, p), 'rb'))
            X_test = pickle.load(open(root_path + 'simulation/interaction/data//X_test_' + file_name(N_train, p), 'rb'))
            Y_train = pickle.load(open(root_path + 'simulation/interaction/data//Y_train_' + file_name(N_train, p), 'rb'))
            Y_test = pickle.load(open(root_path + 'simulation/interaction/data//Y_test_' + file_name(N_train, p), 'rb'))
            for method in ['GPANOVA', 'SSANOVA']:
                print(f'p={p}, N_train={N_train}, method={method}')
                partial_setting = partial(one_fit, method=method, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
                with multiprocessing.Pool(processes=n_cores) as pool:
                   result= list(tqdm(pool.imap_unordered(partial_setting, range(repetitions)), total=repetitions))

                ISE = np.array([[value for key, value in c.items() if key == 'ISE'] for c in result])
                FPR = np.array([[value for key, value in c.items() if key == 'FPR'] for c in result])
                FNR = np.array([[value for key, value in c.items() if key == 'FNR'] for c in result])
                Precision = np.array([[value for key, value in c.items() if key == 'Precision'] for c in result])
                Recall = np.array([[value for key, value in c.items() if key == 'Recall'] for c in result])
                F1 = np.array([[value for key, value in c.items() if key == 'F1'] for c in result])

                runtime = np.array([[value for key, value in c.items() if key == 'runtime'] for c in result])
                effect_freq = np.array([[value for key, value in c.items() if key == 'effect_freq'] for c in result])
                model_size = np.array(
                    [[value for key, value in c.items() if key == 'model_size'] for c in result]).flatten()

                f.write("Uniform design, N_train = {0}, p = {1} \n".format(N_train, p))
                f.write("----------------------------------------------" + '\n')
                f.write('Method: ' + method + '\n')
                f.write('ISE: ' + str('%.3f' % ISE.mean()) + '(' + str('%.3f' % (ISE.std()/np.sqrt(repetitions))) + ')' + '\n')
                f.write('Runtime: ' + str('%.3f' % runtime.mean()) + '(' + str('%.3f' % (runtime.std()/np.sqrt(repetitions))) + ')' + '\n')
                f.write('Mode size: ' + str('%.3f' % model_size.mean()) + '(' + str('%.3f' % (model_size.std()/np.sqrt(repetitions))) + ')' + '\n')
                f.write(' -The frequency of selected effects- \n')
                f.write(' & '.join(str('%.3f' % i) for i in effect_freq.mean(axis=0).flatten()) + "\n")
                f.write('& FPR & FNR & Precision & Recall & F1 score \n')
                f.write(' & ' + str('%.3f' % FPR.mean()) + ' & ' + str('%.3f' % FPR.mean()) + ' & ' + str('%.3f' % Precision.mean()) + ' & ' + str('%.3f' % Recall.mean()) + ' & ' + str('%.3f' % F1.mean()) + '\n')

                f.write("----------------------------------------------" + '\n')
                f.write('\n')
    f.close()
