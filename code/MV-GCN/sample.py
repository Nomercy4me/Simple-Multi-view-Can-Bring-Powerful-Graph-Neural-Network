# --*- coding:UTF-8 -*-
# 这个文件先实现K=2时，节点的dropout
# Load data
from utils import *
import random
import copy
from scipy import sparse


def Sample_graph(adj, features, K=3, init_prob = 0.5, random_flag = 0):
    K = K
    adj = adj
    features = features
    # print(adj)
    # print(features)

    # init_prob = 1.0 / K
    init_prob = init_prob
    # 先得到所有节点个数，然后做节点的dropout
    # node_num = adj.shape[0]

    node_num = adj.shape[0]
    # node_num = 5
    flag_node = np.zeros([node_num])  # 标志一个节点历史上是否被采样过
    Sample_node = []  # 记录所有次采样的结果

    adj = adj.todense()
    features = features.todense()

    # sample节点，先得到每个节点的被采样概率，生成一个0-1之间的随机数，如果这个数小于prob，则这个节点被采样到,根本之前轮次的采样结果更新下一次采样
    for k in range(K):
        sample_node_tmp = np.zeros([node_num])  # 记录当前次采样的结果

        if(random_flag == 1 or k ==0):
            prob_node = np.full([node_num], init_prob)  # 记录每个节点这当前次的采样概率


        #设置每个节点的采样概率，一种为每次都完全random的方式，(random = 1)
        # 一种为根据历史采样结果，调整当前次的采样概率的方式
        elif(random_flag == 0 and k>0):
            # 计算如果恢复K次邻接矩阵的和，需要再采到哪些节点
            recover_adj_prob = np.sum(sum_adj - sum_sample_adj, axis=1) / np.sum(sum_adj, axis=1)
            # print("***********************")
            # print(np.sum(sum_adj - sum_sample_adj, axis=1))
            # print(np.sum(sum_adj, axis=1))
            # print(recover_adj_prob)
            # print("***********************")
            for i in range(node_num):
                # 如果一个节点在历史上从没有被采样到过，则采样概率增加为1/K*k
                # if flag_node[i] == 0:
                #     prob_node[i] = init_prob + k * 1 / K
                tmp_prob1 = init_prob + (1-init_prob)/(K-1)*(k-flag_node[i])
                tmp_prob2 = init_prob + (1-init_prob)/(K-1)*(k-flag_node[i])*recover_adj_prob[i]
                prob_node[i] = (tmp_prob1 + tmp_prob2) /2
            # print(prob_node)

        tmp_adj = copy.deepcopy(adj)

        for i in range(node_num):
            x = random.random()
            if x <= prob_node[i]:
                flag_node[i] += 1
                sample_node_tmp[i] = 1
            else:
                tmp_adj[i, :] = 0
                tmp_adj[:, i] = 0

        if(k == 0):
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

    return sparse_sample_adj, sparse_sample_feature, Sample_node
    # Sample_node是K维的，每个是节点个数维度，标志当前次每个节点是否被采样


    # 得到K次采样后的邻接矩阵和特征矩阵后，直接将原来的邻接矩阵和特征矩阵替换
    # 输入的placeholder会增加，要求为K个邻接矩阵和特征矩阵，算完以后要增加一个拼接层

    # 定义K超参数，在placeholder中定义K个放邻接矩阵和稀疏特征矩阵的位置
    # 每个定义两个层，传入对应的邻接矩阵和稀疏矩阵
    # 把所有采样的结果拼接起来，然后再接一个MLP层


# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
# print([adj])
# test_adj, test_features, test_sample_node = Sample_graph(adj, features, K=2, init_prob = 0.6)
# print(test_adj)
# print(test_features)
# print(test_sample_node)