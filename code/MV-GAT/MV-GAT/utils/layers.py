# -*- coding:UTF-8 -*-
# 定义head的实现方式
import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        # coefs = tf.nn.softmax(tf.nn.relu(logits) + bias_mat)
        
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(vars_ft,vars_at1,vars_at2,seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        # input 处理
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # tf.layers.conv1d(input, filters(卷积核的个数), kernel_size)
        # kernel size为1相当于 F * F'的参数个数，输出为n*F'(out_sz)
        # seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        # print(seq,vars)
        # print("********")
        # # seq_fts = tf.matmul(seq, vars)
        # print("********")
        seq_fts = tf.nn.conv1d(seq,vars_ft,1,padding='SAME')


        # simplest self-attention possible
        # 把每个节点的所有特征加权求和，权即为要学的size为1的kernel,
        # 两种加权求和的方式，把每个节点变为一个标量
        # 把连边特征也转换为一个标量
        # f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        # f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        # f_3 = tf.layers.conv1d(edge_features,1,1)
        f_1 = tf.nn.conv1d(seq_fts, vars_at1, 1,padding='SAME')
        f_2 = tf.nn.conv1d(seq_fts, vars_at2, 1,padding='SAME')

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))
        # f_3 = tf.reshape(f_3, (nb_edges, 1))

        # f1的每个元素作用在矩阵的每行上面
        f_1 = adj_mat * f_1
        # f2的每个元素作用在矩阵的每列上面
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        # print "adj_mat: ",adj_mat
        # print "f1",f_1
        # print "f2",f_2

        # 得到attention的结果，满足mask,但是怪怪的
        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)

        # 对attention做softmax
        coefs = tf.sparse_softmax(lrelu)

        # coef_drop:attn_drop
        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)

        # 对输入做了两次dropout
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        # 先不用bias
        ret = tf.contrib.layers.bias_add(vals)
        # ret = vals

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

