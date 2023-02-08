# -*- coding:UTF-8 -*-
import time
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import argparse

from models.gat import GAT
from models.sp_gat import SpGAT
from utils import process
# from ranking_metrics import label_alignment,precision_at_K,recall_at_K,F1score_at_K,MAP,ndcg
from utils.sample_ktimes import Sample_graph_ktimes


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# cuda_visible_devices
flags.DEFINE_string('cuda', '0', 'Cuda string.')
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'GAT', 'Model string.')  #'deep_structure' 'deepwalk+feature' 'spectral_basis', 'wavelet_origin', 'gcn', 'gcn_cheby','wavelet','nmf', 'dense'
flags.DEFINE_integer('batch_size', 1, 'Number of epochs to train.')#1000
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')#1000
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')#200
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 5e-4, 'Initial weight decay.')
# deep neural model structure
# layer 取值范围2 3 4
flags.DEFINE_integer('layer_num', 2, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.') # 第一个GAT层
flags.DEFINE_integer('hidden2', 6, 'Number of units in hidden layer 2.')  # 在固定两个GAT层且使用Resnet，用不到
flags.DEFINE_integer('hidden3', 16, 'Number of units in hidden layer 3.')  # 第一个MLP层

flags.DEFINE_integer('head1', 8, 'Number of head in input layer and hidden layer.')
flags.DEFINE_integer('head2', 2, 'Number of head in hidden layer and output layer.')
flags.DEFINE_integer('head3', 1, 'Number of head in hidden layer and output layer.')

flags.DEFINE_float('attn_drop', 0.5, 'dropout of attention layer.')
flags.DEFINE_float('ffd_drop', 0.7, 'dropout of connect layer.')

flags.DEFINE_integer('K_sample_num', 8, 'Number of sample.')
flags.DEFINE_float('init_prob', 0.1, 'init_prob of sampling.')
flags.DEFINE_integer('average_sample', 1, 'The flag of average_sample only before training.') # 取1时只在训练前做一次采样，否则每轮训练都做一次采样
flags.DEFINE_integer('N_average_num', 50, 'number for everage sample block.')
flags.DEFINE_float('alpha', 0.5, 'alpha of mlp weight.')
flags.DEFINE_float('beta', 0.5, 'beta of gnn weight.')
flags.DEFINE_integer('FT_weight_share', 0, 'Flag of FT_weight_share.')
flags.DEFINE_integer('mlp_flag', 1, 'Flag of mlp.')
flags.DEFINE_integer('random_flag', 0, 'Flag of mlp.')

K_sample_num = FLAGS.K_sample_num
init_prob = FLAGS.init_prob
N_average_num = FLAGS.N_average_num
alpha = FLAGS.alpha
beta = FLAGS.beta
FT_weight_share = FLAGS.FT_weight_share
average_sample = FLAGS.average_sample
random_flag = FLAGS.random_flag

# training params
training_num = 5
import os
os.environ['CUDA_VISIBLE_DEVICES']= FLAGS.cuda

dataset = FLAGS.dataset
model_name = FLAGS.model
batch_size = FLAGS.batch_size
nb_epochs = FLAGS.epochs
patience = FLAGS.early_stopping
lr = FLAGS.learning_rate  # learning rate
l2_coef = FLAGS.weight_decay  # weight decay
hid_units = [FLAGS.hidden1] # numbers of hidden units per each attention head in each layer
n_heads = [FLAGS.head1,FLAGS.head2] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.relu

attn_drop_input = FLAGS.attn_drop
ffd_drop_input = FLAGS.ffd_drop
# model = GAT
model = SpGAT()

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. units mlp layer: ' + str(FLAGS.hidden3))
print('nb. attention heads: ' + str(n_heads))
print('dropout: ',[ffd_drop_input, attn_drop_input])
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))
print('K_sample_num: '+str(K_sample_num))
print('init_prob: '+str(init_prob))
print('average_sample: '+str(FLAGS.average_sample))
print('N_average_num: '+str(N_average_num))
print('FT_weight_share: '+str(FT_weight_share))
print('mlp_flag: '+str(FLAGS.mlp_flag))
print('alpha: '+str(FLAGS.alpha))
print('beta: '+str(FLAGS.beta))
print('random_flag: '+str(FLAGS.random_flag))

sparse = True

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_raw_data(dataset)

node_num = adj.shape[0]
class_num = y_test.shape[1]

features, spars = process.preprocess_features(features)

# smple data
if(average_sample == 1):
    # 每个都是K维的，每个元素表示第K个采样，在N次采样下，被采到的求和
    sparse_sample_adj, sparse_sample_feature, Sample_node = Sample_graph_ktimes(adj, features, N=N_average_num, K=K_sample_num,init_prob = init_prob,random_flag=random_flag)
    for i in range(len(sparse_sample_feature)):
        sparse_sample_feature[i] = process.preprocess_features(sparse_sample_feature[i])[0][np.newaxis]
        # sparse_sample_feature[i] = to_tuple(process.preprocess_features(sparse_sample_feature[i]))

    biases = [process.preprocess_sample_adj_bias(sparse_sample_adj[i], Sample_node[i]) for i in range(len(sparse_sample_adj))]


nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

# if sparse:
#     biases = process.preprocess_adj_bias(adj)
# else:
#     adj = adj.todense()
#     adj = adj[np.newaxis]
#     biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        # 将输入特征和邻接矩阵都变成数组，数组每个元素表示对应次采样采到的输入特征和邻接矩阵
        # ftr_in = [tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size)) for _ in range(K_sample_num)]
        # if sparse:
        #     #bias_idx = tf.placeholder(tf.int64)
        #     #bias_val = tf.placeholder(tf.float32)
        #     #bias_shape = tf.placeholder(tf.int64)
        #     bias_in = [tf.sparse_placeholder(dtype=tf.float32) for _ in range(K_sample_num)]
        # else:
        #     bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        # lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        # msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        # attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        # ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        # is_train = tf.placeholder(dtype=tf.bool, shape=())

        placeholders = {
            # support 和features都是数组，每个元素是一次采样下的adj和feature,先仅考虑邻接矩阵稀疏的情况
            'ftr_in': [tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size)) for _ in range(K_sample_num)],
            'bias_in': [tf.sparse_placeholder(dtype=tf.float32) for _ in range(K_sample_num)],
            'lbl_in': tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes)),
            'msk_in': tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes)),
            'attn_drop': tf.placeholder_with_default(0., shape=()),
            'ffd_drop': tf.placeholder_with_default(0., shape=()),
            'is_train': tf.placeholder(dtype=tf.bool, shape=())
        # helper variable for sparse dropout
        }


    logits = model.inference(K_sample_num,placeholders['ftr_in'], nb_classes, nb_nodes, placeholders['is_train'],
                             placeholders['attn_drop'], placeholders['ffd_drop'],
                                bias_mat=placeholders['bias_in'],
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)

    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(placeholders['lbl_in'], [-1, nb_classes])
    msk_resh = tf.reshape(placeholders['msk_in'], [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    print tf.trainable_variables()

    # 在随机seed 下每个模型训练5次，取均值作为最终结果
    output_test_training = []
    best_test_training = []
    last_test_training = []

    for this_num in range(0, training_num):

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0

        with tf.Session() as sess:
            sess.run(init_op)

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0
            val_best_acc = 0.0

            ts_best_acc = 0.0
            ts_output_acc = 0.0

            for epoch in range(nb_epochs):
                tr_step = 0
                tr_size = features.shape[0]

                while tr_step * batch_size < tr_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[tr_step * batch_size:(tr_step + 1) * batch_size]
                    # print "bbias:",bbias

                    train_flag = True
                    feed_dict = process.construct_feed_dict(sparse_sample_feature, bbias, y_train, train_mask,train_flag,placeholders,tr_step,batch_size,K_sample_num)
                    feed_dict.update({placeholders['attn_drop']: attn_drop_input})
                    feed_dict.update({placeholders['ffd_drop']: ffd_drop_input})

                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],feed_dict=feed_dict)

                    # _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    #                                     feed_dict={
                    #                                         ftr_in: features[
                    #                                                 tr_step * batch_size:(tr_step + 1) * batch_size],
                    #                                         bias_in: bbias,
                    #                                         lbl_in: y_train[
                    #                                                 tr_step * batch_size:(tr_step + 1) * batch_size],
                    #                                         msk_in: train_mask[
                    #                                                 tr_step * batch_size:(tr_step + 1) * batch_size],
                    #                                         is_train: True,
                    #                                         attn_drop: attn_drop_input, ffd_drop: ffd_drop_input})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                vl_step = 0
                vl_size = features.shape[0]
                # vl_size = 1,  因为 features 被增加了一个轴，所有节点还是一个batch 的送入
                while vl_step * batch_size < vl_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[vl_step * batch_size:(vl_step + 1) * batch_size]

                    train_flag = False
                    feed_dict = process.construct_feed_dict(sparse_sample_feature, bbias, y_val, val_mask,
                                                            train_flag, placeholders, vl_step, batch_size, K_sample_num)

                    val_output, loss_value_vl, acc_vl = sess.run([log_resh, loss, accuracy],
                                                                 feed_dict=feed_dict)
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                # 跑test 集结果，取val最好的结果时test上的表现
                ts_size = features.shape[0]
                ts_step = 0
                ts_loss = 0.0
                ts_acc = 0.0


                while ts_step * batch_size < ts_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]

                    train_flag = False
                    feed_dict = process.construct_feed_dict(sparse_sample_feature, bbias, y_test, test_mask,
                                                            train_flag, placeholders, ts_step, batch_size, K_sample_num)

                    test_output, loss_value_ts, acc_ts = sess.run([log_resh, loss, accuracy],feed_dict=feed_dict)
                    ts_loss += loss_value_ts
                    ts_acc += acc_ts
                    ts_step += 1

                # print one epoch
                print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                      (train_loss_avg / tr_step, train_acc_avg / tr_step,
                       val_loss_avg / vl_step, val_acc_avg / vl_step))
                print('Epoch: ', epoch + 1, ' Test loss:', ts_loss / ts_step, '; Test Acc:', ts_acc / ts_step)

                if ts_best_acc < ts_acc / ts_step:
                    ts_best_acc = ts_acc / ts_step
                if val_best_acc < val_acc_avg / vl_step:
                    val_best_acc = val_acc_avg / vl_step
                    ts_output_acc = ts_acc / ts_step

                # early stopping
                # or val_loss_avg / vl_step <= vlss_mn
                if val_acc_avg / vl_step >= vacc_mx:
                    vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                    # vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                        break

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

            print("dataset: ", FLAGS.dataset, " model: ", FLAGS.model,
                  ",learning_rate:", FLAGS.learning_rate, ",layer_num:", FLAGS.layer_num, ",hidden1:", FLAGS.hidden1,
                  ",hidden2:", FLAGS.hidden2, ",hidden3:", FLAGS.hidden3,
                  ",head1:", FLAGS.head1, ",head2:", FLAGS.head2, ",head3:", FLAGS.head3, ",attn_drop:",
                  FLAGS.attn_drop, ",ffd_drop:",
                  FLAGS.ffd_drop)

            print("Test set results:", "cost=", "{:.5f}".format(ts_loss))
            print("Test Acc: ", ts_acc)
            print("best val Acc: ", val_best_acc)
            print("output test Acc: ", ts_output_acc)
            print("best test Acc: ", ts_best_acc)

            sess.close()

            # 每次训练的结果接起来，多次求均值作为模型结果
            best_test_training.append(ts_best_acc)
            output_test_training.append(ts_output_acc)
            last_test_training.append(ts_acc)

# 多次结果求均值
best_test = np.sum(best_test_training, axis=0) / float(training_num)
output_test = np.sum(output_test_training, axis=0) / float(training_num)
last_test = np.sum(last_test_training, axis=0) / float(training_num)

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('model: ' + str(model))
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. units mlp layer: ' + str(FLAGS.hidden3))
print('nb. attention heads: ' + str(n_heads))
print('dropout: ',[ffd_drop_input, attn_drop_input])
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('K_sample_num: '+str(K_sample_num))
print('init_prob: '+str(init_prob))
print('average_sample: '+str(FLAGS.average_sample))
print('N_average_num: '+str(N_average_num))
print('FT_weight_share: '+str(FT_weight_share))
print('mlp_flag: '+str(FLAGS.mlp_flag))
print('alpha: '+str(FLAGS.alpha))
print('beta: '+str(FLAGS.beta))
print('random_flag: '+str(FLAGS.random_flag))

print('----- results -----')
print("average last test metrics: ", last_test,last_test_training)
print("average output test metrics: ", output_test,output_test_training)
print("average best test metrics: ", best_test,best_test_training)

print("********************************************************")
