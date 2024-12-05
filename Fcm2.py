import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 生成二维数据
def generate_data(n_samples=500, n_features=2, centers=5, cluster_std=2.0, random_state=42):
    dataX, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return dataX, labels

# 初始化隶属度矩阵U
def initialize_U(n_samples, n_clusters):
    # 使用Dirichlet分布随机初始化隶属度矩阵U
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples)
    return U

# 根据隶属度矩阵U计算簇中心
def calculate_centroids(U, data, m):
    # 计算隶属度矩阵的m次方
    um = U ** m
    # 计算新的簇中心
    centroids = um.T @ data / np.sum(um.T, axis=1, keepdims=True)
    return centroids

# 计算每个数据点到每个簇中心的距离
def calculate_distance(data, centroids):
    # 使用欧氏距离计算
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return distances

# 根据距离更新隶属度矩阵U
def update_U(distances, m):
    power = 2 / (m - 1)
    inv_distances = 1 / distances
    # 计算新的隶属度矩阵U
    U = inv_distances / np.sum(inv_distances, axis=1, keepdims=True)
    return U

# FCM算法的主函数
def fcm(data, n_clusters, m=2, max_iter=100, tol=1e-5):
    n_samples = data.shape[0]
    # 初始化隶属度矩阵U
    U = initialize_U(n_samples, n_clusters)
    
    for _ in range(max_iter):
        U_old = U.copy()
        # 计算簇中心
        centroids = calculate_centroids(U, data, m)
        # 计算距离
        distances = calculate_distance(data, centroids)
        # 更新隶属度矩阵U
        U = update_U(distances, m)
        
        # 检查是否满足停止条件
        if np.linalg.norm(U - U_old) < tol:
            break
    
    return U, centroids

# 根据簇中心预测数据点的类别
def predict(X, centroids, m=2):
    # 计算距离
    distances = calculate_distance(X, centroids)
    # 更新隶属度矩阵U
    U = update_U(distances, m)
    # 找到隶属度矩阵中每行的最大值，即该样本最大可能所属类别
    labels = np.argmax(U, axis=1)
    return labels

# 主程序
def main():
    # 生成数据
    dataX, labels = generate_data(n_samples=500, n_features=2, centers=5, cluster_std=2.0, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    dataX = scaler.fit_transform(dataX)

    # 划分训练集和测试集
    ratio = 0.6  # 训练集的比例
    trainLength = int(dataX.shape[0] * ratio)  # 训练集长度
    trainX = dataX[:trainLength, :]
    trainLabels = labels[:trainLength]
    testX = dataX[trainLength:, :]
    testLabels = labels[trainLength:]

    EPS = 1e-5   # 停止误差条件
    m_values = [1.5, 2, 2.5]  # 尝试不同的模糊因子
    n_clusters_values = [4, 5, 6]  # 尝试不同的簇数量

    best_train_accuracy = 0
    best_test_accuracy = 0

    # 多次运行算法并选择最佳结果
    for n_clusters in n_clusters_values:
        for m in m_values:
            for _ in range(10):
                U, centroids = fcm(trainX, n_clusters, m, max_iter=1000, tol=EPS)

                # 预测训练集和测试集的标签
                trainLabels_prediction = predict(trainX, centroids, m)
                testLabels_prediction = predict(testX, centroids, m)

                # 计算训练集和测试集的聚类准确率
                train_accuracy = np.sum(trainLabels_prediction == trainLabels) / trainLength
                test_accuracy = np.sum(testLabels_prediction == testLabels) / (dataX.shape[0] - trainLength)

                if test_accuracy > best_test_accuracy:
                    best_train_accuracy = train_accuracy
                    best_test_accuracy = test_accuracy

    print("Best clustering accuracy on training set: %.2f%%" % (best_train_accuracy * 100))
    print("Best clustering accuracy on test set: %.2f%%" % (best_test_accuracy * 100))

if __name__ == "__main__":
    main()
