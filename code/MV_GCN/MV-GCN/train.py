# --*- coding:UTF-8 -*-
from __future__ import division
from __future__ import print_function
from ogb.nodeproppred import NodePropPredDataset
import warnings

warnings.filterwarnings('ignore')

import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
from utils import *
from models import GCN, MLP
# from model_multi_layer import GCN, MLP
from sample import Sample_graph
from sample_ktimes import Sample_graph_ktimes

# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.8, 'Dropout rate (1 - keep probability).') # 越大被dropout的概率越高
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
# 如果是用不同采样的结果相加，那么hidden2应该和输出的维度相同
flags.DEFINE_integer('hidden2', 6, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 6, 'Number of units in hidden layer 1.') #
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

#跟模型结构相关的超参数
flags.DEFINE_integer('K_sample_num', 5, 'Number of sampling.')
flags.DEFINE_float('init_prob', 0.1, 'Value for init sample prob.')

flags.DEFINE_integer('mlp_flag', 1, 'The flag of MLP after GCN.')
flags.DEFINE_integer('mlp_weight_share', 1, 'The flag of MLP weight share.') # 取1时所有MLP共享权重

flags.DEFINE_integer('average_sample', 1, 'The flag of average_sample only before training.') # 取1时只在训练前做一次采样，否则每轮训练都做一次采样
flags.DEFINE_integer('N_average_sample_num', 10, 'The Number of average sample num before training.') # 采样是来自于多次采样的均值，还是来自于1次采样
flags.DEFINE_integer('random_flag', 0, 'The flag of random sample.') # 取1时在做随机采样


# 和Resnet相关的超参数
flags.DEFINE_float('alpha', 0.5, 'Value for MLP representation.') # MLP结果的权重
flags.DEFINE_float('beta', 0.5, 'Value for GCN representation.') # GCN结果的权重
flags.DEFINE_integer('resnet', 1, 'The flag of resNet.') # 标志是否有ResNet


print("Start Optimization!")

print("K: ",FLAGS.K_sample_num, "hidden1: ",FLAGS.hidden1, "hidden2: ",FLAGS.hidden2,
      "hidden3: ",FLAGS.hidden3,"lr: ",FLAGS.learning_rate, "init_prob: ",FLAGS.init_prob,
      "mlp_flag", FLAGS.mlp_flag, "mlp_weight_share", FLAGS.mlp_weight_share,
      "average_sample", FLAGS.average_sample, "N_average_sample_num", FLAGS.N_average_sample_num,
      "dropout", FLAGS.dropout, "random_flag", FLAGS.random_flag, "weight_decay", FLAGS.weight_decay, "dataset",FLAGS.dataset, "resnet",FLAGS.resnet,"alpha",FLAGS.alpha,"beta",FLAGS.beta)

training_num = 5

# Load data
if FLAGS.dataset != 'arxiv':
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
    # 在采样之前做feature的normalize，feature是按行normalize，对节点采样不影响feature的normalize，所以采样之前做就可以
    # Some preprocessing
    features = preprocess_features(features)
else:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, num_nodes = load_big_data()


# smple data
if(FLAGS.average_sample == 1):
    sparse_sample_adj, sparse_sample_feature, Sample_node = Sample_graph_ktimes(adj, features, N=FLAGS.N_average_sample_num, K=FLAGS.K_sample_num, init_prob = FLAGS.init_prob,random_flag=FLAGS.random_flag)
    for i in range(len(sparse_sample_feature)):
        sparse_sample_feature[i] = to_tuple(preprocess_features(sparse_sample_feature[i]))

    support = [preprocess_adj(sparse_sample_adj[i], Sample_node[i]) for i in range(len(sparse_sample_adj))]

tuple_features = to_tuple(features)

if FLAGS.model == 'gcn':
    # support = [preprocess_adj(i) for i in sparse_sample_adj]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    # support 和features都是数组，每个元素是一次采样下的adj和feature
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(FLAGS.K_sample_num)],
    'features': [tf.sparse_placeholder(tf.float32) for _ in range(FLAGS.K_sample_num)],
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': [tf.placeholder(tf.int32) for _ in range(FLAGS.K_sample_num)]  # helper variable for sparse dropout
}
#, shape=tf.constant(features[2], dtype=tf.int64)
# Create model

model = model_func(placeholders, input_dim=tuple_features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(sparse_sample_feature, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(sparse_sample_feature, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)




# 在随机seed 下每个模型训练5次，取均值作为最终结果
output_test_training = []
last_test_training = []

for this_num in range(0, training_num):
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # 记录val acc最大时test acc
    max_val_acc = 0.0
    max_test_acc = 0.0
    min_val_cost = 100.0

    # test的时候用所有采样的均值
    adj_sum = []
    feature_sum = []
    Sample_node_sum = []

    early_stop = 0

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()

        # sample data for this epoch
        if(FLAGS.average_sample == 0):
            # sparse_sample_adj, sparse_sample_feature, Sample_node = Sample_graph(adj, features, K=FLAGS.K_sample_num,
            #                                                         init_prob=FLAGS.init_prob, random_flag=FLAGS.random_flag)


            sparse_sample_adj, sparse_sample_feature, Sample_node = Sample_graph_ktimes(adj, features,N=FLAGS.N_average_sample_num, K=FLAGS.K_sample_num,
                                                                                 init_prob=FLAGS.init_prob,
                                                                                 random_flag=FLAGS.random_flag)

            adj_sum.append(sparse_sample_adj)
            feature_sum.append(sparse_sample_feature.copy())
            Sample_node_sum.append(Sample_node)

            # Some preprocessing
            for i in range(len(sparse_sample_feature)):
                sparse_sample_feature[i] = to_tuple(sparse_sample_feature[i])
                # preprocess_features()

            support = [preprocess_adj(sparse_sample_adj[i], Sample_node[i]) for i in range(len(sparse_sample_adj))]

        # Construct feed dictionary
        feed_dict = construct_feed_dict(sparse_sample_feature, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # feed_dict的作用是更新placeholder的值

        # print(sparse_sample_feature)
        # print("************* feed_dict")
        # print(feed_dict)
        # print("************* placeholders")
        # print(placeholders)
        # print("*************")
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(sparse_sample_feature, support, y_val, val_mask, placeholders)
        cost_val.append(acc)

        test_cost, test_acc, test_duration = evaluate(sparse_sample_feature, support, y_test, test_mask, placeholders)
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "Test results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc))

        if(acc>max_val_acc):
            max_val_acc = acc
            max_test_acc = test_acc
            early_stop = 0
        else:
            early_stop +=1
        if(early_stop > FLAGS.early_stopping):
            print("Early stopping...")
            break

        # if epoch > FLAGS.early_stopping and cost_val[-1] < np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        #     print("Early stopping...")
        #     break

    if (FLAGS.average_sample == 0):
        test_adj = []
        test_features = []
        test_sample_node = []
        # 在测试的时候，把训练过程的采样结果求平均,模型就用最后一个模型？
        for i in range(FLAGS.K_sample_num):
            tmp = adj_sum[0][i]
            tmp_feature = feature_sum[0][i]
            tmp_node = Sample_node_sum[0][i]
            # 如果用early stop，就用实际训练的次数
            for j in range(1, FLAGS.epochs):
                tmp = tmp + adj_sum[j][i]
                tmp_feature += feature_sum[j][i]
                tmp_node += Sample_node_sum[j][i]
            test_adj.append(tmp)
            test_features.append(tmp_feature)
            test_sample_node.append(tmp_node)

        for i in range(len(test_features)):
            test_features[i] = to_tuple(preprocess_features(test_features[i]))
            # preprocess_features()

        support = [preprocess_adj(test_adj[i], test_sample_node[i]) for i in range(len(test_adj))]
        test_cost, test_acc, test_duration = evaluate(test_features, support, y_test, test_mask, placeholders)

    # Testing
    if (FLAGS.average_sample == 1):
        test_cost, test_acc, test_duration = evaluate(sparse_sample_feature, support, y_test, test_mask, placeholders)

    print("Test set results:", "cost=", "{:.5f}".format(test_cost),"accuracy=", "{:.5f}".format(test_acc),"max accuracy=", "{:.5f}".format(max_test_acc), "time=", "{:.5f}".format(test_duration))

    last_test_training.append(test_acc)
    output_test_training.append(max_test_acc)

print("Optimization Finished! random flag existed")

last_test = np.sum(last_test_training, axis=0) / float(training_num)
output_test = np.sum(output_test_training, axis=0) / float(training_num)

print("average output test metrics: ", output_test_training,output_test)
print("average last test metrics: ", last_test_training,last_test)

print("K: ",FLAGS.K_sample_num, "hidden1: ",FLAGS.hidden1, "hidden2: ",FLAGS.hidden2,
      "hidden3: ",FLAGS.hidden3,"lr: ",FLAGS.learning_rate, "init_prob: ",FLAGS.init_prob,
      "mlp_flag", FLAGS.mlp_flag, "mlp_weight_share", FLAGS.mlp_weight_share,
      "average_sample", FLAGS.average_sample, "N_average_sample_num", FLAGS.N_average_sample_num,
      "dropout", FLAGS.dropout, "random_flag", FLAGS.random_flag, "weight_decay", FLAGS.weight_decay, "dataset",FLAGS.dataset,"resnet",FLAGS.resnet,"alpha",FLAGS.alpha,"beta",FLAGS.beta)