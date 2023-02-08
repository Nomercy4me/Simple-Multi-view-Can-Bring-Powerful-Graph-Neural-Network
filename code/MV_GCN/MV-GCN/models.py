# --*- coding:UTF-8 -*-
from layers import *
from metrics import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        if(FLAGS.mlp_flag == 1):
            for k in range(FLAGS.K_sample_num):
                self.activations.append(self.inputs[k])
                for i in range(4):
                    hidden = self.layers[4 * k + i](self.activations[-1])
                    self.activations.append(hidden)

            if(FLAGS.K_sample_num == 1):
                self.activations.append(FLAGS.alpha*self.activations[4] + FLAGS.beta*self.activations[2])
            else:
                # 4是指4层，包括两个GCN层，两个全连接层组成MLP，如果是深层的模型，这里要换成对应层数
                # 如果有残差连接，则把GCN的结果和MLP的结果相加，这么做需要控制MLP和GCN的维度相同，好处是什么呢？梯度回传？
                if(FLAGS.resnet == 1):
                    tmp_tensor = FLAGS.alpha*self.activations[4] + FLAGS.beta*self.activations[2]
                else:
                    tmp_tensor = self.activations[4]
                for k in range(2, FLAGS.K_sample_num+1):
                    if(FLAGS.resnet == 1):
                        tmp_tensor = tf.add(tmp_tensor, FLAGS.alpha*self.activations[5 * k - 1] + FLAGS.beta*self.activations[5 * k - 3])
                    else:
                        tmp_tensor = tf.add(tmp_tensor, self.activations[5 * k - 1])
                self.activations.append(tmp_tensor)

        # # Build sequential layer model
        if(FLAGS.mlp_flag == 0):
            for k in range(FLAGS.K_sample_num):
                self.activations.append(self.inputs[k])
                for i in range(2):
                    hidden = self.layers[2 * k + i](self.activations[-1])
                    self.activations.append(hidden)

            if (FLAGS.K_sample_num == 1):
                self.activations.append(self.activations[-1])
            else:
                tmp_tensor = self.activations[2]
                # tmp_tensor = tf.nn.softmax(self.activations[2])
                for k in range(2, FLAGS.K_sample_num + 1):
                    tmp_tensor = tf.add(tmp_tensor, self.activations[3 * k - 1])
                    # tmp_tensor = tf.add(tmp_tensor, tf.nn.softmax(self.activations[3 * k - 1]))
                # tmp_sum_tensor = tf.reduce_sum(tmp_tensor,axis=1)
                # tmp_tensor = tmp_tensor/tmp_sum_tensor
                self.activations.append(tmp_tensor)

        # 拼接不合理，考虑使用加法
        # self.activations.append(
        #     tf.add([self.activations[3 * k - 1] for k in range(1, FLAGS.K_sample_num + 1)], axis=1))

        # # 再接两个Dense层，为MLP（保持单射）
        # self.activations.append(
        #     tf.concat([self.activations[3 * k - 1] for k in range(1, FLAGS.K_sample_num + 1)], axis=1))
        #
        # # 现在效果变差好像是两个dense层带来的
        # for layer in self.layers[2 * FLAGS.K_sample_num:]:
        #     hidden = layer(self.activations[-1])
        #     self.activations.append(hidden)

        self.outputs = self.activations[-1]

        # self.activations.append(self.inputs)
        # for layer in self.layers:
        #     hidden = layer(self.activations[-1])
        #     self.activations.append(hidden)
        # self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):

        vars = tf.trainable_variables()
        self.loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * FLAGS.weight_decay

        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # for var in self.vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #
        # # 当mlp的参数不共享时，这部分有参数，当没有mlp或者mlp参数共享时，这部分没参数
        # for var in self.layers[-1].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # for var in self.layers[-2].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if(FLAGS.dataset == 'doubanmovie'):
            self.loss += multilabel_masked_sigmoid_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])
        else:
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        # K个并行的GCN，每个运行在对应的被采样的图上
        self.vars['weights_GClayer1'] = glorot([self.input_dim, FLAGS.hidden1],name='weights_GClayer1' )
        if (FLAGS.mlp_flag == 0):
            self.vars['weights_GClayer2'] = glorot([FLAGS.hidden1, self.output_dim], name='weights_GClayer2')
        else:
            self.vars['weights_GClayer2'] = glorot([FLAGS.hidden1, FLAGS.hidden2], name='weights_GClayer2')

            # 把经过GCN之后的结果，通过一个MLP再加起来，相当于每一个都通过了四层的变换（两层GCN，两层MLP）
            # 这个MLP部分可以共享参数，也可以不同采样用不同的参数，如果不同采样用不同的参数，可以说成是在为不同采样调整

            self.vars['weights_MLPlayer1'] = glorot([FLAGS.hidden2, FLAGS.hidden3], name='weights_MLPlayer1')
            self.vars['weights_MLPlayer2'] = glorot([FLAGS.hidden3, self.output_dim], name='weights_MLPlayer2')

        for k in range(FLAGS.K_sample_num):
            self.layers.append(GraphConvolution(k, weights = self.vars['weights_GClayer1'],input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=True,
                                                logging=self.logging))

            if(FLAGS.mlp_flag == 0):
                self.layers.append(GraphConvolution(k, weights = self.vars['weights_GClayer2'],input_dim=FLAGS.hidden1,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))

            elif (FLAGS.mlp_flag == 1):
                # 第二层是用relu还是不用激活函数，这个是超参数，基准数据集影响不大，douban数据集不用激活函数
                self.layers.append(GraphConvolution(k, weights=self.vars['weights_GClayer2'], input_dim=FLAGS.hidden1,
                                                    output_dim=FLAGS.hidden2,
                                                    placeholders=self.placeholders,
                                                    act=lambda x: x,
                                                    dropout=True,
                                                    logging=self.logging))

                self.layers.append(Dense(k, weights = self.vars['weights_MLPlayer1'],input_dim=FLAGS.hidden2,
                                         output_dim=FLAGS.hidden3,
                                         placeholders=self.placeholders,
                                         act=tf.nn.relu,
                                         dropout=True,
                                         logging=self.logging))

                self.layers.append(Dense(k, weights = self.vars['weights_MLPlayer2'],input_dim=FLAGS.hidden3,
                                         output_dim=self.output_dim,
                                         placeholders=self.placeholders,
                                         act=lambda x: x,
                                         dropout=True,
                                         logging=self.logging))

        # 把多个采样图上的GCN结果拼接起来，用MLP做维度转换(MLP的目的是保持单射)，上面GCN的输出也都是output_dim
        # self.layers.append(Dense(input_dim=FLAGS.hidden2 * FLAGS.K_sample_num,
        #                          output_dim=FLAGS.hidden3,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.relu,
        #                          dropout=True,
        #                          logging=self.logging))
        #
        # self.layers.append(Dense(input_dim=FLAGS.hidden3,
        #                          output_dim=self.output_dim,
        #                          placeholders=self.placeholders,
        #                          act=lambda x: x,
        #                          dropout=True,
        #                          logging=self.logging))

    def predict(self):
        return tf.nn.sigmoid(self.outputs)
