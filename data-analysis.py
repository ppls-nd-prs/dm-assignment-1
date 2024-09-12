from main import tree_grow, tree_grow_b, tree_pred, tree_pred_b
import pandas as pd

train = pd.read_csv("eclipse-metrics-packages-2.0.csv", delimiter=";")
print(train.columns)