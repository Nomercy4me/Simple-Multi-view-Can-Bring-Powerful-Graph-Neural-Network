# 多标签数据上的train，跟Cora三个基准数据相比需要修改评价指标，loss函数，数据读取
# 整个模型的结构和数据集无关，不需要修改模型结构

# --*- coding:UTF-8 -*-
from __future__ import division
from __future__ import print_function

import warnings

warnings.filterwarnings('ignore')

import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *
from models import GCN, MLP
from sample_ktimes import Sample_graph_ktimes
from ranking_metrics import *

# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'doubanmovie', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).') # 越大被dropout的比例越高
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
# 如果是用不同采样的结果相加，那么hidden2应该和输出的维度相同
flags.DEFINE_integer('hidden2', 27, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 27, 'Number of units in hidden layer 1.') #
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

#跟模型结构相关的超参数
flags.DEFINE_integer('K_sample_num', 2, 'Number of sampling.')
flags.DEFINE_float('init_prob', 1.0, 'Value for init sample prob.')

flags.DEFINE_integer('mlp_flag', 1, 'The flag of MLP after GCN.')
flags.DEFINE_integer('mlp_weight_share', 1, 'The flag of MLP weight share.') # 取1时所有MLP共享权重

flags.DEFINE_integer('average_sample', 1, 'The flag of average_sample only before training.') # 取1时只在训练前做一次采样，否则每轮训练都做一次采样
flags.DEFINE_integer('N_average_sample_num', 1, 'The Number of average sample num before training.') # 采样是来自于多次采样的均值，还是来自于1次采样
flags.DEFINE_integer('random_flag', 1, 'The flag of random sample.') # 取1时在做随机采样


# 和Resnet相关的超参数
flags.DEFINE_float('alpha', 0.2, 'Value for MLP representation.') # MLP结果的权重
flags.DEFINE_float('beta', 0.8, 'Value for GCN representation.') # GCN结果的权重
flags.DEFINE_integer('resnet', 1, 'The flag of resNet.') # 标志是否有ResNet


print("Start Optimization!")

print("K: ",FLAGS.K_sample_num, "hidden1: ",FLAGS.hidden1, "hidden2: ",FLAGS.hidden2,
      "hidden3: ",FLAGS.hidden3,"lr: ",FLAGS.learning_rate, "init_prob: ",FLAGS.init_prob,
      "mlp_flag", FLAGS.mlp_flag, "mlp_weight_share", FLAGS.mlp_weight_share,
      "average_sample", FLAGS.average_sample, "N_average_sample_num", FLAGS.N_average_sample_num,
      "dropout", FLAGS.dropout, "random_flag", FLAGS.random_flag, "weight_decay", FLAGS.weight_decay, "dataset",FLAGS.dataset, "resnet",FLAGS.resnet,"alpha",FLAGS.alpha,"beta",FLAGS.beta)

# Load data
if(FLAGS.dataset == "doubanmovie"):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_multilabel_data(FLAGS.dataset)
else:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# 在采样之前做feature的normalize，feature是按行normalize，对节点采样不影响feature的normalize，所以采样之前做就可以
# Some preprocessing
# 需要考虑wiki上 preprocess 特征 怎么做

features = multilabel_preprocess_features(features)

training_num = 3

# sample data
if(FLAGS.average_sample == 1):
    sparse_sample_adj, sparse_sample_feature, Sample_node = Sample_graph_ktimes(adj, features, N=FLAGS.N_average_sample_num, K=FLAGS.K_sample_num, init_prob = FLAGS.init_prob,random_flag=FLAGS.random_flag)
    for i in range(len(sparse_sample_feature)):
        # sparse_sample_feature[i] = to_tuple(sparse_sample_feature[i])
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
    outs_val = sess.run([model.predict(),model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1],outs_val[2], (time.time() - t_test)


# 在随机seed 下每个模型训练5次，取均值作为最终结果
output_test_training = []
best_test_training = []
last_test_training = []

for this_num in range(0, training_num):

    # Init variables
    sess.run(tf.global_variables_initializer())

    print(tf.trainable_variables () )

    cost_val = []

    # 记录val acc最大时test acc
    max_val_acc = 0.0
    max_test_acc = 0.0

    # test的时候用所有采样的均值
    adj_sum = []
    feature_sum = []
    Sample_node_sum = []

    node_num = adj.shape[0]
    class_num = y_test.shape[1]
    print(node_num,class_num)

    early_stop = 0
    k = [1,2,3,4,5]
    metric_num = 11
    best_val_metric = [0.0]*metric_num
    best_test_metric = [0.0]*metric_num
    output_test_metric = [0.0]*metric_num

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
        val_output, cost, acc, duration = evaluate(sparse_sample_feature, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        label_align = label_alignment(output=val_output, label=y_val, mask=val_mask, sample_num=node_num,
                                      class_num=class_num)
        val_metric = [0.0] * metric_num
        val_metric[0] = MAP(label_align, val_mask, node_num, class_num)
        for metric_index in range(1, 6):
            val_metric[metric_index] = F1score_at_K(label_align, y_val, val_mask, k[metric_index - 1], node_num)
        for metric_index in range(6, 11):
            val_metric[metric_index] = ndcg(y_val, label_align, val_mask, k[metric_index - 6], node_num)

        # Test
        test_output, test_cost, test_acc, test_duration = evaluate(sparse_sample_feature, support, y_test, test_mask, placeholders)

        label_align = label_alignment(output=test_output, label=y_test, mask=test_mask, sample_num=node_num,
                                      class_num=class_num)
        test_metric = [0.0] * metric_num
        test_metric[0] = MAP(label_align, test_mask, node_num, class_num)
        for metric_index in range(1, 6):
            test_metric[metric_index] = F1score_at_K(label_align, y_test, test_mask, k[metric_index - 1], node_num)
        for metric_index in range(6, 11):
            test_metric[metric_index] = ndcg(y_test, label_align, test_mask, k[metric_index - 6], node_num)


        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "val_loss=", "{:.5f}".format(cost), "val_map=", "{:.5f}".format(val_metric[0]),
              "test_loss=", "{:.5f}".format(test_cost), "test_map=", "{:.5f}".format(test_metric[0]),
              "time=", "{:.5f}".format(time.time() - t))

        if (val_metric[0] > best_val_metric[0]):
            early_stop = 0
        else:
            early_stop += 1

        # 记录各个评价指标最大的时候
        for i in range(metric_num):
            if (best_val_metric[i] < val_metric[i]):
                best_val_metric[i] = val_metric[i]
                output_test_metric[i] = test_metric[i]
            if (best_test_metric[i] < test_metric[i]):
                best_test_metric[i] = test_metric[i]

        if (early_stop > FLAGS.early_stopping):
            print("Early stopping...")
            break

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
        test_output,test_cost, test_acc, test_duration = evaluate(test_features, support, y_test, test_mask, placeholders)

    # Testing
    if (FLAGS.average_sample == 1):
        test_output, test_cost, test_acc, test_duration = evaluate(sparse_sample_feature, support, y_test, test_mask, placeholders)

    label_align = label_alignment(output=test_output, label=y_test, mask=test_mask, sample_num=node_num,
                                  class_num=class_num)
    test_metric = [0.0] * metric_num
    test_metric[0] = MAP(label_align, test_mask, node_num, class_num)
    for metric_index in range(1, 6):
        test_metric[metric_index] = F1score_at_K(label_align, y_test, test_mask, k[metric_index - 1], node_num)
    for metric_index in range(6, 11):
        test_metric[metric_index] = ndcg(y_test, label_align, test_mask, k[metric_index - 6], node_num)


    print("Test metrics: ", test_metric)
    print("best val metrics: ", best_val_metric)
    print("output test metrics: ", output_test_metric)
    print("best test metrics: ", best_test_metric)

    # 每次训练的结果接起来，多次求均值作为模型结果
    best_test_training.append(best_test_metric)
    output_test_training.append(output_test_metric)
    last_test_training.append(test_metric)


print("Optimization Finished!")

best_test = np.sum(best_test_training, axis=0) / float(training_num)
output_test = np.sum(output_test_training, axis=0) / float(training_num)
last_test = np.sum(last_test_training, axis=0) / float(training_num)

print("average output test metrics: ", output_test)
print("average best test metrics: ", best_test)
print("average last test metrics: ", last_test)


print("K: ",FLAGS.K_sample_num, "hidden1: ",FLAGS.hidden1, "hidden2: ",FLAGS.hidden2,
      "hidden3: ",FLAGS.hidden3,"lr: ",FLAGS.learning_rate, "init_prob: ",FLAGS.init_prob,
      "mlp_flag", FLAGS.mlp_flag, "mlp_weight_share", FLAGS.mlp_weight_share,
      "average_sample", FLAGS.average_sample, "N_average_sample_num", FLAGS.N_average_sample_num,
      "dropout", FLAGS.dropout, "random_flag", FLAGS.random_flag, "weight_decay", FLAGS.weight_decay, "dataset",FLAGS.dataset,"resnet",FLAGS.resnet,"alpha",FLAGS.alpha,"beta",FLAGS.beta)