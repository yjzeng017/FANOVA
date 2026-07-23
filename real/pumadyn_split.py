import os
import pickle
from sklearn.model_selection import train_test_split


# User should specify the path.
root_path = "/Users/yjzeng/Desktop/FANOVA/github_code/"
# The splits will be stored in: root_path + 'real/splits/', make sure this dictionary has been created.
split_path = root_path + 'real/splits/'
isExists = os.path.exists(split_path)
if not isExists:
    os.makedirs(split_path)


replication = 50
test_size = 0.9
n = 8192


if __name__ == "__main__":
    fold_indices = []
    for i in range(replication):
        split = train_test_split(range(n), test_size=test_size,random_state=i+1)
        fold_indices.append({
            'train': split[0],
            'test': split[1]
        })
    pickle.dump(fold_indices, open(split_path + "pumadyn_splits.pkl", 'wb'))
