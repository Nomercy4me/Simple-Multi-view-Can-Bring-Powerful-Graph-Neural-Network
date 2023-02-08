# --*- coding:UTF-8 -*-
# 这个文件先实现K=2时，边的dropout，因为采样边是无偏的
# 先加上self loop再采样边
# 采样N次，每次做K次采样，将N次的结果传进去，不需要求均值，传进去以后会做normalize
# 参考Dropedge的实现，遍历连边
# Load data
from utils import *
import copy
from scipy import sparse


def Sample_edge_ktimes(adj,features, N=10, K=3, init_prob = 0.5, random_flag = 0):
    N = N
    K = K
    adj = adj+sp.eye(adj.shape[0])
    init_prob = init_prob
    # 先得到所有节点个数，然后做节点的dropout
    node_num = adj.shape[0]

    # 共采样N次，对N次采样结果相加再Normalize，防止采样结果与期望偏差较大
    # adj = adj.todense()

    # 返回经过N次采样的k个block个的邻接矩阵
    # 先对这个邻接矩阵加上self loop

    adj_sample = {}

    for t in range(N):
        flag_node = np.zeros([node_num, node_num])  # 标志一个连边历史上是否被采样过

        # sample节点，先得到每个节点的被采样概率，生成一个0-1之间的随机数，如果这个数小于prob，则这个节点被采样到,根本之前轮次的采样结果更新下一次采样
        for k in range(K):
            if (random_flag == 1 or k == 0):
                prob_node = np.full([node_num, node_num], init_prob)  # 记录每个节点这当前次的采样概率
                # 设置每个节点的采样概率，一种为每次都完全random的方式，(random = 1)
                # 一种为根据历史采样结果，调整当前次的采样概率的方式
            elif (random_flag == 0 and k > 0):
                # 如果一个节点在历史上从没有被采样到过，则采样概率增加为1/K*k
                recover_adj_prob = (sum_adj - sum_sample_adj) / sum_adj
                prob_node = np.full([node_num, node_num], init_prob)
                tmp_ones = np.ones([node_num,node_num])
                tmp_K_matrix = np.full([node_num, node_num], K*1.0)
                tmp_sk_matrix = np.full([node_num, node_num], k)
                tmp_prob1 = prob_node + (tmp_ones - prob_node)/(tmp_K_matrix - tmp_ones)*(tmp_sk_matrix-flag_node)
                tmp_prob2 = prob_node + (tmp_ones - prob_node) / (tmp_K_matrix - tmp_ones) * (tmp_sk_matrix - flag_node)*recover_adj_prob
                prob_node = tmp_prob1 + tmp_prob2
                # for i in range(node_num):
                #     for j in range(node_num):
                #         tmp_prob1 = init_prob + (1 - init_prob) / (K - 1) * (k - flag_node[i,j])
                #         tmp_prob2 = init_prob + (1 - init_prob) / (K - 1) * (k - flag_node[i,j]) * recover_adj_prob[i,j]
                #         prob_node[i,j] = (tmp_prob1 + tmp_prob2) / 2

            tmp_adj = copy.deepcopy(adj)

            x = np.random.rand(node_num*node_num)
            x = np.reshape(x,[node_num,node_num])
            index_flag = (x<=prob_node)
            index_flag = index_flag.astype(int)
            flag_node += index_flag
            tmp_adj[index_flag == 0] = 0.0

            # for i in range(node_num):
            #     for j in range(node_num):
            #         print(i,j)
            #         x = random.random()
            #         if x <= prob_node[i,j]:
            #             flag_node[i,j] += 1
            #         else:
            #             tmp_adj[i, j] = 0
            #             tmp_adj[j, i] = 0

            if (k == 0):
                sum_sample_adj = tmp_adj
                sum_adj = adj
            else:
                sum_sample_adj = sum_sample_adj + tmp_adj
                sum_adj = sum_adj + adj

            adj_sample[str(t)+","+str(k)] = tmp_adj

    test_adj = []
    test_features = []
    for i in range(K):
        tmp = sparse.csr_matrix(adj_sample[str(0)+","+str(i)])
        for j in range(1, N):
            tmp = tmp + sparse.csr_matrix(adj_sample[str(j)+","+str(i)])
        test_adj.append(tmp)
        test_features.append(features)

        # tmp_node是把block i的N次采样的结果接起来了，指示在这N次采样下，每个节点被采到的次数，test_sample_node还是K维的

    return test_adj, test_features

    # 得到K次采样后的邻接矩阵和特征矩阵后，直接将原来的邻接矩阵和特征矩阵替换
    # 输入的placeholder会增加，要求为K个邻接矩阵和特征矩阵，算完以后要增加一个拼接层

    # 定义K超参数，在placeholder中定义K个放邻接矩阵和稀疏特征矩阵的位置
    # 每个定义两个层，传入对应的邻接矩阵和稀疏矩阵
    # 把所有采样的结果拼接起来，然后再接一个MLP层


# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
# test_adj,test_features = Sample_edge_ktimes(adj,features, N=3, K=2, init_prob = 0.6)