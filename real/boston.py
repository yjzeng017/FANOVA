#-
import io
import os
import numpy as np
import pandas as pd
import pickle
from functools import partial
from FANOVAModels.GPANOVA import GPANOVAModel
from FANOVAModels.SSANOVA import SSANOVAModel
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import contextlib
from sklearn.preprocessing import MinMaxScaler


# User should specify the path.
root_path = '/Users/yjzeng/Desktop/FANOVA/github_code/'
# The real datasets were stored in: root_path + 'real/datasets/'
data_path = root_path + 'real/datasets/'
# The splits were stored in: root_path + 'real/splits/'
splits_path = root_path + 'real/splits/'
# The results will be stored in: root_path + 'real/results/', make sure this dictionary has been created.
result_path = root_path + 'real/results/'
isExists = os.path.exists(result_path)
if not isExists:
    os.makedirs(result_path)


splits = pickle.load(open(splits_path + "boston_splits.pkl", "rb"))
df = pd.read_csv(data_path +'boston.data', sep=',',header=0)
df = df.drop(columns=['town', 'tract', 'medv', 'chas', "lon", "lat",])
df = df.reset_index(drop=True)

exclude_cols = ['cmedv']
cols_to_scale = [col for col in df.columns if col not in exclude_cols]
scaler = MinMaxScaler(feature_range=(0, 1))
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
# Define the features by dropping both target variables
X_scaled = np.array(df.drop(columns=['cmedv']))
Y = np.array(df['cmedv'])


def one_fit(j, method, omega):
    train_idx, test_idx  = splits[j]['train'], splits[j]['test']
    X_train = X_scaled[train_idx, :]
    Y_train = Y[train_idx]
    X_test = X_scaled[test_idx, :]
    Y_test = Y[test_idx]
    N_train, p = X_train.shape
    with contextlib.redirect_stdout(io.StringIO()):
        if method == 'GPANOVA':
            model = GPANOVAModel(
                X=X_train,
                Y=Y_train.reshape(N_train, 1),
                max_interaction_depth=2)
            model.fit(omega=omega/(N_train+2), verbose=True)
        else:
            model = SSANOVAModel(
                X=X_train,
                Y=Y_train.reshape(N_train, 1),
                max_interaction_depth=2,
            )
            model.fit(delta=0.001/np.log(N_train), omega=omega/(N_train+2), max_iter=100, verbose=True)
        # evaluation
        Y_test_pred = model.predict(X_test)
        ISE = np.mean((Y_test - Y_test_pred) ** 2)
        selected_effects = model.selected_main_effects() + model.selected_interactions()
        model_size = len(selected_effects)
        effect_freq = [1 if dim in selected_effects else 0 for dim in model.selected_dims]
        ncurves = model.predict_by_component(X_test)
        f = ncurves.sum(axis=1)
        effect_corr = ncurves.T @ f / np.dot(f, f)
        sobol_indice = np.var(ncurves, axis=0) / np.std(f) **2
    return {
        'ISE': ISE,
        'effect_freq': effect_freq,
        'model_size': model_size,
        'effect_corr': effect_corr,
        'sobol_indice': sobol_indice,
    }



# Do 10-fold cross-validation ten times, then len(splits)=100.
# Users can set a smaller value for rep when testing.
rep = len(splits)
omega_list = np.arange(2, 11, 1)
n_cores=8


if __name__ == "__main__":
    for method in ['GPANOVA', 'SSANOVA']:
        f = open(root_path + f'real/boston_{method}_results.txt', 'w')
        f.write("========================================================\n")
        f.write(
            "boston data: interaction model evaluated based on 10-fold CV for {0} repetitions".format(rep / 10) + "\n")
        f.write("========================================================\n")
        for omega in omega_list:
            print(f"Method = {method}, omega={omega}/(n+2)")
            train_with_fixed_method = partial(one_fit, method=method, omega=omega)
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                results = list(tqdm(executor.map(train_with_fixed_method, range(rep)), total=rep))
            pickle.dump(results, open(result_path + f"boston_{method}_w_{omega}_10cv_results.pkl", 'wb'))
            ISE = np.array([[value for key, value in c.items() if key == 'ISE'] for c in results])
            effect_freq = np.array([[value for key, value in c.items() if key == 'effect_freq'] for c in results])
            model_size = np.array([[value for key, value in c.items() if key == 'model_size'] for c in results]).flatten()
            max_size, min_size, mean_size, median_size, std_size = model_size.max(), model_size.min(), model_size.mean(), np.nanmedian(model_size), model_size.std()
            effect_corr = np.array([[value for key, value in c.items() if key == 'effect_corr'] for c in results])
            sobol_indice = np.array([[value for key, value in c.items() if key == 'sobol_indice'] for c in results])

            f.write(f"Method = {method}, omega={omega}/(N+2)\n")
            f.write('Risk:' + '& Ave.ISE' + '& Med.ISE' + '& Sd.ISE' + '\n')
            f.write('& ' + str(round(ISE.mean(), 3)) + ' & ' + str(round(np.nanmedian(ISE), 3)) + ' & ' + str(
                round(ISE.std() / np.sqrt(rep), 3)) + '\n')
            f.write(' & Ave.Size' + ' & Med.Size' + ' & Sd.Size' +  ' & Range'+ '\n')
            f.write(' & ' + str(round(mean_size, 3)) + ' & ' + str(round(median_size, 3)) + '& ' + str(
                round(std_size / np.sqrt(rep), 3)) + ' & [' + str(min_size) + ',' + str(max_size) + ']' + '\n')

            f.write(' -The frequency of effects- \n')
            f.write(', '.join(str('%.3f' % i) for i in effect_freq.mean(axis=0).flatten()) + "\n")
            f.write(' -The correlation of components to the total function- \n')
            f.write(', '.join(str('%.3f' % i) for i in effect_corr.mean(axis=0).flatten()) + "\n")
            f.write(' -The sobol indice of components to the total function- \n')
            f.write(', '.join(str('%.3f' % i) for i in sobol_indice.mean(axis=0).flatten()) + "\n")
            f.write("\n")
    f.close()




