from main import tree_grow, tree_grow_b, tree_pred, tree_pred_b
import numpy as np
import pandas as pd

train = pd.read_csv("eclipse-metrics-packages-2.0.csv", delimiter=";")
test = pd.read_csv("eclipse-metrics-packages-3.0.csv", delimiter=";")

feats = ['pre',
 'ACD_avg',
 'ACD_max',
 'ACD_sum',
 'FOUT_avg',
 'FOUT_max',
 'FOUT_sum',
 'MLOC_avg',
 'MLOC_max',
 'MLOC_sum',
 'NBD_avg',
 'NBD_max',
 'NBD_sum',
 'NOCU',
 'NOF_avg',
 'NOF_max',
 'NOF_sum',
 'NOI_avg',
 'NOI_max',
 'NOI_sum',
 'NOM_avg',
 'NOM_max',
 'NOM_sum',
 'NOT_avg',
 'NOT_max',
 'NOT_sum',
 'NSF_avg',
 'NSF_max',
 'NSF_sum',
 'NSM_avg',
 'NSM_max',
 'NSM_sum',
 'PAR_avg',
 'PAR_max',
 'PAR_sum',
 'TLOC_avg',
 'TLOC_max',
 'TLOC_sum',
 'VG_avg',
 'VG_max',
 'VG_sum']
train_x = train[feats]
train_y = train["post"]
test_x = test[feats]
test_y = test["post"]

train_x_np = train_x.to_numpy()
test_x_np = test_x.to_numpy()

train_y_np = train_y.to_numpy()
train_y_np = np.where(train_y_np > 0, 1, 0)
test_y_np = test_y.to_numpy()
test_y_np = np.where(test_y_np > 0, 1, 0)

bagging_tree = tree_grow_b(train_x_np, train_y_np, 15, 5, 41, 100)

test_y_np_pred_bt = tree_pred_b(test_x_np, bagging_tree)