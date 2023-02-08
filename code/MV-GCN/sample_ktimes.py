# --*- coding:UTF-8 -*-
# 这个文件先实现K=2时，节点的dropout
# 采样N次次，每次做K次采样，将N次的结果传进去，不需要求均值，传进去以后会做normalize
# Load data
from utils import *
import random
import copy
from scipy import sparse


def Sample_graph_ktimes(adj, features, N=10, K=3, init_prob = 0.5, random_flag = 0):
    N = N
    K = K
    # adj = adj
    # features = features
    # print(adj)
    # print(features)

    # init_prob = 1.0 / K
    init_prob = init_prob
    # 先得到所有节点个数，然后做节点的dropout
    node_num = adj.shape[0]
    # node_num = 5

    # 共采样N次，对N次采样结果相加再Normalize，防止采样结果与期望偏差较大
    adj_sum = []
    feature_sum = []
    Sample_node_sum = []

    adj = adj.todense()
    features = features.todense()

    for t in range(N):
        flag_node = np.zeros([node_num])  # 标志一个节点历史上是否被采样过
        Sample_node = []  # 记录所有次采样的结果

        # sample节点，先得到每个节点的被采样概率，生成一个0-1之间的随机数，如果这个数小于prob，则这个节点被采样到,根本之前轮次的采样结果更新下一次采样
        for k in range(K):
            sample_node_tmp = np.zeros([node_num])  # 记录当前次采样的结果，被采到为1

            if (random_flag == 1 or k == 0):
                prob_node = np.full([node_num], init_prob)  # 记录每个节点这当前次的采样概率
                # 设置每个节点的采样概率，一种为每次都完全random的方式，(random = 1)
                # 一种为根据历史采样结果，调整当前次的采样概率的方式
            elif (random_flag == 0 and k > 0):
                # 如果一个节点在历史上从没有被采样到过，则采样概率增加为1/K*k
                recover_adj_prob = np.sum(sum_adj - sum_sample_adj, axis=1) / np.sum(sum_adj, axis=1)
                for i in range(node_num):
                    # tmp_prob1 = init_prob + (1 - init_prob) / (K - 1) * (k - flag_node[i])
                    tmp_prob2 = init_prob + (1 - init_prob) / (K - 1) * (k - flag_node[i]) * recover_adj_prob[i]
                    # prob_node[i] = (tmp_prob1 + tmp_prob2) / 2
                    prob_node[i] = tmp_prob2

            tmp_adj = copy.deepcopy(adj)

            for i in range(node_num):
                x = random.random()
                if x <= prob_node[i]:
                    flag_node[i] += 1
                    sample_node_tmp[i] = 1
                else:
                    tmp_adj[i, :] = 0
                    tmp_adj[:, i] = 0

            if (k == 0):
                sum_sample_adj = tmp_adj
                sum_adj = adj
            else:
                sum_sample_adj = sum_sample_adj + tmp_adj
                sum_adj = sum_adj + adj

            Sample_node.append(sample_node_tmp)

        sparse_sample_adj = []
        sparse_sample_feature = []

        # 有了K轮的采样结果，这个采样结果会影响邻接矩阵，如果直接把对应的节点在邻接矩阵和特征矩阵置0就可以了？
        # 对adj的行和列，及feature中对应的行置0，每次采样下都有一个adj和X，
        for k in Sample_node:
            tmp_adj = copy.deepcopy(adj)
            tmp_feature = copy.deepcopy(features)
            # 为1就是被采样到了，为0就是没有被采样到
            for j in range(node_num):
                if k[j] == 0:
                    tmp_adj[j, :] = 0
                    tmp_adj[:, j] = 0
                    tmp_feature[j, :] = 0

            sparse_sample_adj.append(sparse.csr_matrix(tmp_adj))
            sparse_sample_feature.append(sparse.csr_matrix(tmp_feature))

        adj_sum.append(sparse_sample_adj) # sparse_sample_adj是K维的，每个是被采样的稀疏的邻接矩阵，adj_sum是N维的，每个是一次sparse_sample_adj
        feature_sum.append(sparse_sample_feature.copy())
        Sample_node_sum.append(Sample_node) # Sample_node是K维的，每个是标志当前次每个节点是否被采样，Sample_node_sum是N维的，每个是一次Sample_node

    test_adj = []
    test_features = []
    test_sample_node = []
    # 把所有次采样的结果求平均，送到模型中，训练和测试都是在这个平均采样上进行
    for i in range(K):
        tmp = adj_sum[0][i]
        tmp_feature = feature_sum[0][i]
        tmp_node = Sample_node_sum[0][i]
        for j in range(1, N):
            tmp = tmp + adj_sum[j][i]
            tmp_feature += feature_sum[j][i]
            tmp_node += Sample_node_sum[j][i]
        test_adj.append(tmp)
        test_features.append(tmp_feature)
        test_sample_node.append(tmp_node)
        # tmp_node是把block i的N次采样的结果接起来了，指示在这N次采样下，每个节点被采到的次数，test_sample_node还是K维的

    return test_adj, test_features, test_sample_node

    # 得到K次采样后的邻接矩阵和特征矩阵后，直接将原来的邻接矩阵和特征矩阵替换
    # 输入的placeholder会增加，要求为K个邻接矩阵和特征矩阵，算完以后要增加一个拼接层

    # 定义K超参数，在placeholder中定义K个放邻接矩阵和稀疏特征矩阵的位置
    # 每个定义两个层，传入对应的邻接矩阵和稀疏矩阵
    # 把所有采样的结果拼接起来，然后再接一个MLP层


# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
# print([adj])
# test_adj, test_features, test_sample_node = Sample_graph_ktimes(adj, features, N=10, K=2, init_prob = 0.6)
# print(test_adj)
# print(test_features)
# print(test_sample_node)