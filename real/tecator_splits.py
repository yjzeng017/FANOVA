# import numpy as np
import os
import pickle
from sklearn.model_selection import KFold



# User should specify the path.
root_path = "/Users/yjzeng/Desktop/FANOVA/github_code/"
# The splits will be stored in: root_path + 'real/splits/', make sure this dictionary has been created.
split_path = root_path + 'real/splits/'
isExists = os.path.exists(split_path)
if not isExists:
    os.makedirs(split_path)


replication = 10
n = 215

if __name__ == "__main__":
    fold_indices = []
    for i in range(replication):
        kf = KFold(n_splits=10, shuffle=True, random_state=i+1)
        for train_idx, test_idx in kf.split(range(n)):
            fold_indices.append({
                'train': train_idx,
                'test': test_idx
            })

    pickle.dump(fold_indices, open(split_path + "tecator_splits.pkl", 'wb'))