import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))

    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # 标准数据划分是采用（前类别*20）作为训练样本，接着500个作为验证样本，整个数据的最后1000个作为测试样本
    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + 500)

    # 为了防止因为数据太少MLP引入的参数太多带的效果下降，我们引入更多数据：整个数据的最后1000个作为测试样本，再倒数500个作为验证样本，其余作为训练样本
    # print("using more datas")
    idx_test = test_idx_range.tolist()
    start = idx_test[0]
    idx_val = range(start-500,start)
    idx_train = range(start-500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


from ogb.nodeproppred import NodePropPredDataset

def load_new_data():
    dataset = NodePropPredDataset(name = "ogbn-arxiv")
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0]
    y_train = np.zeros(label.shape)
    y_train[train_idx, :] = label[train_idx, :]
    y_val = np.zeros(label.shape)
    y_val[valid_idx, :] = label[valid_idx, :]
    y_test = np.zeros(label.shape)
    y_test[test_idx, :] = label[test_idx, :]

    for j in graph:
        print(j,type(graph[j]))
        if j == 'edge_index':
            row = graph[j][0]
            col = graph[j][1]
            data = [1] * len(row)
            adj = csr_matrix((data, (row, col)), shape=(169343, 169343))
        if j == 'node_feat':
            features = graph[j]

        if j == 'num_nodes':
            num_nodes = graph[j]
    print(features)
    return adj, features, y_train, y_val, y_test, train_idx, valid_idx, test_idx, num_nodes

load_new_data()

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
print(y_train)

# 把arxiv数据处理成cora的形式