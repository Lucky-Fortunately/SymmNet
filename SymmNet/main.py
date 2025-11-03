import warnings
from SymmNet import SymmNet
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn

dataset = "Criteo"
if dataset == "Avazu":
    col_names = ['click'] + [str(i) for i in range(1, 27)]
    data = pd.read_csv("./datasets/Avazu/sample_avazu.csv", header=None, names=col_names)
    label = 'click'
    target = [label]
    dense_features = [str(i) for i in range(1, 17)]
    sparse_features = [str(i) for i in range(17, 27)]

elif dataset == "Criteo":
    col_names = ['lable'] + [str(i) for i in range(1, 40)]
    data = pd.read_csv("./datasets/Criteo/sample_criteo.csv", header=None, names=col_names)
    label = 'lable'
    target = [label]
    dense_features = [str(i) for i in range(1, 14)]
    sparse_features = [str(i) for i in range(14, 40)]

feature_names = sparse_features + dense_features
feat_sizes1={ feat:1 for feat in dense_features}
feat_sizes2 = {feat: data[feat].max() + 1 for feat in sparse_features}
feat_sizes={}
feat_sizes.update(feat_sizes1)
feat_sizes.update(feat_sizes2)
train, test = train_test_split(data, test_size=0.2, random_state=42)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
if dataset == "Avazu":
    model = SymmNet(feat_sizes,
                   sparse_feature_columns=sparse_features,
                   dense_feature_columns=dense_features,
                   device=device,
                   mlp1_hidden_units=[32, 32, 32],
                   mlp2_hidden_units=[ 256, 256, 256],
                   embedding_size=8,
                   lr=0.001,
                   head=4)
elif dataset == "Criteo":
    model = SymmNet(feat_sizes,
                   sparse_feature_columns=sparse_features,
                   dense_feature_columns=dense_features,
                   device=device,
                   mlp1_hidden_units=[32, 32, 32],
                   mlp2_hidden_units=[ 256, 256, 256],
                   embedding_size=4,
                   lr=0.0001,
                   head=4)

test_auc, test_logloss = model.fit(train_model_input, train[target].values, test_model_input, test[target].values, epochs=20, verbose=1, early_stop_round=3)
print(f"test AUC: {test_auc:.4f}, LogLoss: {test_logloss:.4f}")
