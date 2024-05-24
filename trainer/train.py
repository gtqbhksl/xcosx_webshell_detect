import keras.optimizers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd

# 加载CSV文件
df = pd.read_csv('train.csv')  # 替换为你的CSV文件路径

# 分离特征和标签
X = df.iloc[:, :-1].values  # 所有行，除了最后一列
y = df.iloc[:, -1].values  # 最后一列

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 定义神经网络模型
def create_model(hidden_layers, neurons_per_layer, learning_rate, decay, momentum, batch_size):
    model = Sequential()
    for i, neurons in enumerate(neurons_per_layer):
        model.add(Dense(neurons, input_dim=X_scaled.shape[1] if i == 0 else None, activation='relu'))
        if i < len(neurons_per_layer) - 1:
            model.add(Dropout(0.5))  # 添加dropout防止过拟合
    model.add(Dense(1, activation='sigmoid'))  # 输出层
    # 编译模型，使用SGD优化器并传入超参数
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=learning_rate, decay=decay, momentum=momentum),
                  metrics=['accuracy'])
    return model


# 定义每层可能的神经元数量
neurons_options = range(3, 8)

# 定义隐藏层数量的范围
hidden_layers_range = range(3, 8)

# 定义学习率、衰减、动量和批量大小的范围
learning_rate_range = [0.01,0.05,0.08,0.1]
decay_range = [0.01,0.03,0.05,0.08,0.1]
momentum_range = [0.05,0.08,0.12,0.2]
batch_size_range = [8,16, 32, 50]

# 创建所有可能的组合
param_grid = {
    'hidden_layers': [(n,) for n in hidden_layers_range],
    'neurons_per_layer': [(n,) for n in neurons_options],
    'learning_rate': learning_rate_range,
    'decay': decay_range,
    'momentum': momentum_range,
    'batch_size': batch_size_range
}

# 创建Keras分类器
classifier = KerasClassifier(build_fn=create_model, verbose=0)

# 创建网格搜索对象
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)

# 训练模型
grid.fit(X_scaled, y)

# 打印最佳参数和对应的准确率和召回率
print(f'Best parameters found: {grid.best_params_}')
best_model = grid.best_estimator_
y_pred = best_model.predict(X_scaled)
print(classification_report(y, y_pred))