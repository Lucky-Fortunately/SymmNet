import warnings
from MultiHeadAttention import MultiHeadAttention
warnings.filterwarnings('ignore')
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import time
from tqdm import tqdm
from collections import OrderedDict
class SymmNet(nn.Module):
    def __init__(self, feat_sizes, sparse_feature_columns, dense_feature_columns,
                 mlp1_hidden_units=[32, 32, 32], mlp2_hidden_units=[128, 128, 128, 128],
                 dnn_dropout=0.3, embedding_size=4, l2_reg_linear=0.0001,
                 l2_reg_embedding=0.0001, l2_reg_dnn=0.0001, init_std=0.0001,
                 device='cpu', head=2, lr=0.001):
        super(SymmNet, self).__init__()
        self.feat_sizes = feat_sizes
        self.device = device
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.embedding_size = embedding_size
        self.l2_reg_linear = l2_reg_linear
        self.feature_sizes = len(feat_sizes)
        self.mlp1_hidden_units = mlp1_hidden_units
        self.mlp2_hidden_units = mlp2_hidden_units
        self.dnn_dropout = dnn_dropout
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_dnn = l2_reg_dnn
        self.lr = lr
        self.head = head
        self.feature_index = self.build_input_features(self.feat_sizes)
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.weight = nn.Parameter(torch.Tensor(len(self.dense_feature_columns), 1))
        torch.nn.init.normal_(self.weight, mean=0, std=0.0001)
        self.embedding_dict = self.create_embedding_matrix(self.sparse_feature_columns, feat_sizes, self.embedding_size,
                                                           sparse=False, device=self.device)
        self.dropout = nn.Dropout(dnn_dropout)
        self.dnn_input_size = self.embedding_size * len(self.sparse_feature_columns) + len(self.dense_feature_columns)
        self.sparse_input_size = self.embedding_size * len(self.sparse_feature_columns)
        self.dense_input_size = len(self.dense_feature_columns)
        self.mha1 = MultiHeadAttention(heads=head, d_model=self.dnn_input_size)
        hidden_mlp1_units = [self.dnn_input_size] + self.mlp1_hidden_units
        hidden_mlp2_units = [self.dnn_input_size] + self.mlp2_hidden_units
        self.mlp1 = self._build_mlp(hidden_mlp1_units)
        self.mlp2 = self._build_mlp(hidden_mlp2_units)
        for module in [self.mlp1, self.mlp2]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param, mean=0, std=init_std)
        mlp_output_cat_dim = self.mlp1_hidden_units[-1] + self.mlp2_hidden_units[-1]
        self.mha2 = MultiHeadAttention(heads=head, d_model=mlp_output_cat_dim)
        self.final_linear = nn.Linear(mlp_output_cat_dim, 1)
        self.to(device)

    def _build_mlp(self, hidden_units):
        layers = []
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dnn_dropout))
        return nn.Sequential(*layers)

    def forward(self, X):
        X = X.to(self.device)
        sparse_embedding = [self.embedding_dict[feat](
            X[:, self.feature_index[feat][0]:self.feature_index[feat][1]].long().to(self.device))
            for feat in self.sparse_feature_columns]
        dense_inputs = [X[:, self.feature_index[feat][0]:self.feature_index[feat][1]].to(self.device)
                        for feat in self.dense_feature_columns]
        batch_size = X.shape[0]
        sparse_input = torch.cat(sparse_embedding, dim=1).view(batch_size, -1)
        dense_input = torch.cat(dense_inputs, dim=1)
        dnn_input = torch.cat([sparse_input, dense_input], dim=-1)  # shape: (batch_size, input_dim)
        dnn_input_reshaped = dnn_input.unsqueeze(1)
        attn_output = self.mha1(dnn_input_reshaped, dnn_input_reshaped, dnn_input_reshaped)
        attn_output = attn_output.squeeze(1)
        mlp1_output = self.mlp1(attn_output)
        mlp2_output = self.mlp2(attn_output)
        mlp_concat = torch.cat([mlp1_output, mlp2_output], dim=-1)  # (batch_size, 2 * dnn_hidden_units[-1])
        mlp_concat_reshaped = mlp_concat.unsqueeze(1)  # (batch_size, 1, 2 * dnn_hidden_units[-1])
        final_attn_output = self.mha2(mlp_concat_reshaped, mlp_concat_reshaped, mlp_concat_reshaped)
        final_attn_output = final_attn_output.squeeze(1)  # (batch_size, 2 * dnn_hidden_units[-1])
        logits = self.final_linear(final_attn_output)
        y_pred = torch.sigmoid(logits)
        return y_pred

    def fit(self, train_input, y_label, test_input, y_test, batch_size=128, epochs=15, verbose=5, early_stop_round=3):
        feature_names = list(self.feature_index.keys())
        test_features = np.column_stack([test_input[feat] for feat in feature_names])
        test_features, val_features, y_test, y_val = train_test_split(
            test_features, y_test, test_size=0.5, random_state=42
        )
        val_input = {}
        for i, feat in enumerate(feature_names):
            val_input[feat] = val_features[:, i]
        final_test_input = {}
        for i, feat in enumerate(feature_names):
            final_test_input[feat] = test_features[:, i]

        train_x = [train_input[feature] for feature in self.feature_index]
        for i in range(len(train_x)):
            if len(train_x[i].shape) == 1:
                train_x[i] = np.expand_dims(train_x[i], axis=1)

        train_tensor_data = TensorDataset(
            torch.from_numpy(np.concatenate(train_x, axis=-1)),
            torch.from_numpy(y_label)
        )
        train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=batch_size)

        val_x = [val_input[feature] for feature in self.feature_index]
        for i in range(len(val_x)):
            if len(val_x[i].shape) == 1:
                val_x[i] = np.expand_dims(val_x[i], axis=1)

        val_tensor_data = TensorDataset(
            torch.from_numpy(np.concatenate(val_x, axis=-1)),
            torch.from_numpy(y_val)
        )
        val_loader = DataLoader(dataset=val_tensor_data, shuffle=False, batch_size=batch_size)

        test_x = [final_test_input[feature] for feature in self.feature_index]
        for i in range(len(test_x)):
            if len(test_x[i].shape) == 1:
                test_x[i] = np.expand_dims(test_x[i], axis=1)

        test_tensor_data = TensorDataset(
            torch.from_numpy(np.concatenate(test_x, axis=-1)),
            torch.from_numpy(y_test)
        )
        test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=batch_size)
        model = self.train()
        loss_func = F.binary_cross_entropy
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // 128 + 1
        best_auc = 0
        no_improve_count = 0
        best_model_state = None

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_start_time = time.time()
            model.train()
            total_loss_epoch = 0.0
            train_preds, train_targets = [], []
            for index, (x_train, y_train) in enumerate(train_loader):
                x = x_train.to(self.device).float()
                y = y_train.to(self.device).float()
                y_pred = model(x).squeeze()
                loss = loss_func(y_pred, y.squeeze())
                loss = loss + self.l2_reg_linear * self.get_L2_Norm()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss_epoch += loss.item()
                train_preds.append(y_pred.detach().cpu().numpy())
                train_targets.append(y.squeeze().detach().cpu().numpy())


            avg_train_loss = total_loss_epoch / steps_per_epoch

            model.eval()
            val_preds, val_targets = [], []
            val_loss = 0.0

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device).float()
                    y_val = y_val.to(self.device).float()
                    y_pred_val = model(x_val).squeeze()
                    val_loss += loss_func(y_pred_val, y_val.squeeze()).item()
                    val_preds.append(y_pred_val.cpu().numpy())
                    val_targets.append(y_val.squeeze().cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_auc = roc_auc_score(val_targets, val_preds)
            avg_val_loss = val_loss / len(val_loader)
            epoch_duration = time.time() - epoch_start_time

            if epoch % verbose == 0 or epoch == epochs - 1:
                print(f"\nEpoch {epoch + 1}/{epochs} | Time: {epoch_duration:.2f}s")
                print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                no_improve_count = 0
                best_model_state = self.state_dict().copy()
            else:
                no_improve_count += 1
            if no_improve_count >= early_stop_round:
                break
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        model.eval()
        test_preds, test_targets = [], []
        test_loss = 0.0

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device).float()
                y_test = y_test.to(self.device).float()

                y_pred_test = model(x_test).squeeze()
                test_loss += loss_func(y_pred_test, y_test.squeeze()).item()

                test_preds.append(y_pred_test.cpu().numpy())
                test_targets.append(y_test.squeeze().cpu().numpy())

        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_auc = roc_auc_score(test_targets, test_preds)
        test_logloss = log_loss(test_targets, test_preds)
        avg_test_loss = test_loss / len(test_loader)
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test LogLoss: {test_logloss:.4f}")

        return test_auc, test_logloss

    def val_auc_logloss(self, val_input, y_val, batch_size=5000, use_double=False, print_result=True):
        pred_ans = self.predict(val_input, batch_size)
        val_logloss = log_loss(y_val, pred_ans)
        val_auc = roc_auc_score(y_val, pred_ans)
        if print_result:
            print("test LogLoss is %.4f test AUC is %.4f" % (val_logloss, val_auc))
        return val_auc, val_logloss

    def get_L2_Norm(self):
        loss = torch.zeros((1,), device=self.device)
        loss = loss + torch.norm(self.weight)
        for t in self.embedding_dict.parameters():
            loss = loss + torch.norm(t)
        return loss

    def build_input_features(self, feat_sizes):
        features = OrderedDict()
        start = 0
        for feat in feat_sizes:
            feat_name = feat
            if feat_name in features:
                continue
            features[feat_name] = (start, start + 1)
            start += 1
        return features

    def create_embedding_matrix(self, sparse_feature_columns, feat_sizes, embedding_size, init_std=0.0001, sparse=False,
                                device='cpu'):
        embedding_dict = nn.ModuleDict(
            {feat: nn.Embedding(feat_sizes[feat], embedding_size, sparse=False)
             for feat in sparse_feature_columns}
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        return embedding_dict.to(device)


