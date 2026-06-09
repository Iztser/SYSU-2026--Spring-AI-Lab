import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 丢弃Id列，分离特征和目标值
    X = data.drop(['Id', 'SalePrice'], axis=1)
    y = data['SalePrice'].values.reshape(-1, 1)

    # 缺失值填充
    zero_fill_cols = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2',
                      'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea',
                      'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                      'ScreenPorch', 'PoolArea', 'MiscVal']
    for col in zero_fill_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    # 有序特征映射
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    bsmt_expo_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
    fin_type_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
    func_map = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'NA': 0}
    garage_fin_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}

    ordinal_maps = {
        'ExterQual': qual_map, 'ExterCond': qual_map,
        'BsmtQual': qual_map, 'BsmtCond': qual_map,
        'HeatingQC': qual_map, 'KitchenQual': qual_map,
        'FireplaceQu': qual_map, 'GarageQual': qual_map, 'GarageCond': qual_map,
        'PoolQC': qual_map,
        'BsmtExposure': bsmt_expo_map,
        'BsmtFinType1': fin_type_map, 'BsmtFinType2': fin_type_map,
        'Functional': func_map,
        'GarageFinish': garage_fin_map
    }
    for col, mapping in ordinal_maps.items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(0)

    # 处理数值型特征：填充0
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    # 处理文本型特征：填充缺失并进行 One-hot 编码
    categorical_cols = X.select_dtypes(include=[object]).columns
    remaning_cats = [c for c in categorical_cols if c not in ordinal_maps.keys()]
    X = pd.get_dummies(X, columns=remaning_cats)

    # C. 数据标准化
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 先划分，后标准化 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 特征标准化：用训练集拟合，再转换验证集
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)

    # 目标值标准化：用训练集拟合，再转换验证集
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # 返回标准化后的数据，以及 scaler_y（用于后续反标准化）
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_y

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.001, batch_size=64):
    # 转换为张量
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    # 创建数据加载器
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()  # 使用 MSE 作为主要损失

    train_history = []
    val_history = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        # 每个 epoch 结束后进行验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            train_loss = np.mean(batch_losses)
            
            train_history.append(train_loss)
            val_history.append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    return train_history, val_history

def plot_results(train_losses, val_losses, y_true_scaled, y_pred_scaled, scaler_y):
    # 反标准化回真实价格单位
    y_true = scaler_y.inverse_transform(y_true_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    plt.figure(figsize=(14, 5))

    # 左图：Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 右图：预测 vs 实际
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
    
    # 画出 y=x 基准线
    ideal_line = [y_true.min(), y_true.max()]
    plt.plot(ideal_line, ideal_line, color='red', lw=2, label='y=x')
    
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. 主程序入口
# ==========================================
def main():
    # A. 数据准备
    X_train, X_val, y_train, y_val, scaler_y = load_and_preprocess_data('train.csv')
    
    # B. 模型实例化
    input_dim = X_train.shape[1]
    model = MLP(input_dim)
    
    # C. 模型训练
    train_history, val_history = train_model(model, X_train, y_train, X_val, y_val, epochs=200)
    
    # D. 获取最终预测值进行绘图
    model.eval()
    with torch.no_grad():
        final_val_pred = model(torch.FloatTensor(X_val)).numpy()
    
    # E. 结果展示
    plot_results(train_history, val_history, y_val, final_val_pred, scaler_y)

if __name__ == "__main__":
    main()