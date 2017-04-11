# -*- coding: utf-8 -*-
# 反向传播算法
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = np.matrix(iris.data)
y = iris.target

n_values = np.max(y) + 1
y = np.eye(n_values)[y]
y = np.matrix(y)


def sigmoid(x, deriv=False):
    y = 1.0 / (1 + np.exp(-x))
    if (deriv == True):
        return np.multiply(x, 1 - x)
    return y


# 样本数
sample_num = len(X)
# 样本特征数
sample_len = 4
# 输出层单元数
out_num = 3
# 隐含层1单元数
hid_num_1 = 10
# 隐含层2单元数
hid_num_2 = 10

m = len(X)
# 输入层到隐含层1的权重
Theta1 = np.matrix(np.ones(hid_num_1 * (sample_len + 1)).reshape(hid_num_1, sample_len + 1))  # 10*5
# 隐含层1到隐含层2的权重
Theta2 = np.matrix(np.ones(hid_num_2 * (hid_num_1 + 1)).reshape(hid_num_2, hid_num_1 + 1))  # 10*11
# 隐含层2到输出层的权重
Theta3 = np.matrix(np.ones(out_num * (hid_num_2 + 1)).reshape(out_num, hid_num_2 + 1))  # 3*11
# 正向传播 求得假设函数与y间的误差

while True:
    a_1 = np.insert(X, 0, values=np.ones(len(X)), axis=1)
    a_2 = sigmoid(a_1 * Theta1.T)
    a_2 = np.insert(a_2, 0, np.ones(len(a_2)), axis=1)
    a_3 = sigmoid(a_2 * Theta2.T)
    a_3 = np.insert(a_3, 0, np.ones(len(a_3)), axis=1)
    a_4 = sigmoid(a_3 * Theta3.T)
    # 输出层误差
    error_4 = a_4 - y

    # 计算前一层误差
    error_3 = np.multiply(error_4 * Theta3, sigmoid(a_3, True))
    a_2 = a_2[:, 1:]
    error_2 = np.multiply(error_3 * Theta2.T, sigmoid(a_2, True))

    # 计算梯度下降
    delta3 = a_3.T * error_4
    delta2 = a_2.T * error_3
    delta1 = a_1.T * error_2

    Theta1 -= 0.01 * delta1.T
    Theta2 -= 0.01 * delta2
    Theta3 -= 0.01 * delta3.T

    z_1 = np.matrix([1, 6.7, 3., 5., 1.7])
    z_2 = np.insert(sigmoid(z_1 * Theta1.T), 0, values=np.ones(len(sigmoid(z_1 * Theta1.T))), axis=1)
    z_3 = np.insert(sigmoid(z_2 * Theta2.T), 0, values=np.ones(len(sigmoid(z_2 * Theta2.T))), axis=1)
    z_4 = sigmoid(z_3 * Theta3.T)
    print(z_4)
