#-*-coding:UTF-8 -*-
#定义一个GAT的类，继承自基础的类
import numpy as np
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
from inits import *
import sys
sys.path.append("..")

from utils import layers
from models.base_gattn import BaseGAttN

class SpGAT(BaseGAttN):
    def __init__(self):
        self.vars = {}
    # inputs 和bias_mat 都是数组,第k个元素是第k次采样结果，也是第k个GAT的输入
    def inference(self,K_sample_num,inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, 
            residual=False):


        input_size = int(inputs[0].shape[2])  #(1, 2708, 1433)
        res = []
        tmp_units = [input_size] + hid_units + [nb_classes]
        tmp_heads = [1] + n_heads
        # 跑每个采样的结果
        # 需要在每个GAT里面共享参数,所以在进入k之前定义参数，然后把定义好的参数传进去
        if FLAGS.FT_weight_share == 1:
            self.vars['weights_GClayer_0'] = glorot([1, tmp_units[0], tmp_units[1]], name='weights_GClayer_0')
            self.vars['weights_GClayer_1'] = glorot([1, tmp_units[1] * tmp_heads[1], nb_classes], name='weights_GClayer_1')
        if FLAGS.mlp_flag == 1:
            self.vars['weights_FClayer1'] = glorot([1,nb_classes,FLAGS.hidden3], name='weights_FClayer1')
            self.vars['weights_FClayer2'] = glorot([1, FLAGS.hidden3, nb_classes], name='weights_FClayer2')

        # 根据head 个数定义attention参数
        for head_layer in range(len(n_heads)):
            for head in range(n_heads[head_layer]):
                if FLAGS.FT_weight_share == 1:
                    self.vars['weights_GClayer_' + str(head_layer) + 'head' + str(head)] = self.vars['weights_GClayer_'+str(head_layer)]
                else:
                    self.vars['weights_GClayer_'+str(head_layer)+'head'+str(head)] = \
                        glorot([1, tmp_units[head_layer]*tmp_heads[head_layer], tmp_units[head_layer+1]], name='weights_GClayer_'+str(head_layer)+'head'+str(head))
                self.vars[str(head_layer)+"layer"+str(head)+"head1"] = \
                    glorot([1,tmp_units[head_layer+1],1], name = str(head_layer)+"layer"+str(head)+"head1")
                self.vars[str(head_layer) + "layer" + str(head) + "head2"] = \
                    glorot([1, tmp_units[head_layer+1], 1], name=str(head_layer) + "layer" + str(head) + "head2")


        for k in range(K_sample_num):
            attns = []
            for _ in range(n_heads[0]):
                attns.append(layers.sp_attn_head(self.vars['weights_GClayer_'+str(0)+'head'+str(_)],
                    self.vars[str(0)+"layer"+str(_)+"head1"],self.vars[str(0) + "layer" + str(head) + "head2"],
                    inputs[k],
                    adj_mat=bias_mat[k],
                    out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False))


            h_1 = tf.concat(attns, axis=-1)
            # for i in range(1, len(hid_units)):
            #     attns = []
            #     for _ in range(n_heads[i]):
            #         attns.append(layers.sp_attn_head(self.vars['weights_GClayer_i'],self.vars[str(0)+"layer"+str(_)+"head"],h_1,
            #             adj_mat=bias_mat[k],
            #             out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
            #             in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            #     h_1 = tf.concat(attns, axis=-1)

            out = []
            for i in range(n_heads[-1]):
                out.append(layers.sp_attn_head(self.vars['weights_GClayer_'+str(1)+'head'+str(i)],
                    self.vars[str(1)+"layer"+str(i)+"head1"],self.vars[str(1)+"layer"+str(i)+"head2"],
                    h_1, adj_mat=bias_mat[k],
                    out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
            logits = tf.add_n(out) / n_heads[-1]

            if (FLAGS.mlp_flag == 1):
                if ffd_drop != 0.0:
                    seq_fts = tf.nn.dropout(logits, 1.0 - ffd_drop)
                seq_fts = activation(tf.nn.conv1d(seq_fts, self.vars['weights_FClayer1'], 1, padding='SAME'))
                if ffd_drop != 0.0:
                    seq_fts = tf.nn.dropout(seq_fts, 1.0 - ffd_drop)
                seq_fts = activation(tf.nn.conv1d(seq_fts, self.vars['weights_FClayer2'], 1, padding='SAME'))
                logits = FLAGS.alpha * seq_fts + FLAGS.beta * logits

            res.append(logits)

        # 把每个sample的结果拼接起来
        self.results = tf.add_n(res)

            #得到GAT跑出来的结果，再把这个结果放到MLP

            # dropout
            # if self.sparse_inputs:
            #     x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            # else:
            #     x = tf.nn.dropout(x, 1 - self.dropout)
            #
            # # transform
            # if (FLAGS.mlp_weight_share == 1):
            #     output = dot(x, self.weights, sparse=self.sparse_inputs)
            # else:
            #     output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
            #
            # # bias
            # if self.bias:
            #     output += self.vars['bias']
            #
            # return self.act(output)

        #得到多次采样的结果，再把采样的结果
    
        return self.results

    def predict(self):
        return tf.nn.sigmoid(self.results)
