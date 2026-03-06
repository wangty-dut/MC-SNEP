import tensorflow.compat.v1 as tf


# define the first layer of LSTM
class Lstm1(object):
    def __init__(self, input_width, state_width, batch_size):
        self.input_width = input_width  # 代表RNN输入的宽度/维
        self.state_width = state_width  # 代表RNN内部状态的宽度/维度
        self.batch_size = batch_size  # 代表在训练或推理过程中并行处理的序列/例子的数量
        self.times = 0  # 跟踪RNN的当前迭代/时间步骤
        self.Mfh, self.Mfg = self.init_weight_mat()  # RNN中遗忘门的权重矩阵
        # weight matrix of input gates
        self.Mih, self.Mig = self.init_weight_mat()
        self.Moh, self.Mog = self.init_weight_mat()
        self.Mch, self.Mcg = self.init_weight_mat()
        # self.c_list
        # `： 存储RNN每个时间步骤的单元状态（`c'）的列表。
        # - `self.h_list`： 存储RNN在每个时间步长的隐藏状态（`h`）的列表。
        # - `self.ft_list`： 存储RNN在每个时间步长的遗忘门值（`f`）的列表。
        # - `self.it_list`： 存储RNN在每个时间步长的输入门值（i）的列表。
        # - `self.ot_list`： 存储RNN在每个时间步长的输出门值(`o`)
        # 的列表。
        # `self.ut_list`： 列表存储RNN在每个时间步骤的即时状态（`c~`）
        # cell states c
        self.c_list = self.init_state_vec()
        # hidden states h
        self.h_list = self.init_state_vec()
        # forget gate f
        self.ft_list = self.init_state_vec()
        # input gate i
        self.it_list = self.init_state_vec()
        # output gate o
        self.ot_list = self.init_state_vec()
        # immediate state c~
        self.ut_list = self.init_state_vec()

    def init_weight_mat(self):
        '''
        initialize weight matrix
        '''
        Mh = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1), name='l1_gate_h')
        Mg = tf.Variable(tf.random_normal([self.input_width, self.state_width], stddev=0.1), name='l1_gate_g')
        return Mh, Mg

    def init_state_vec(self):
        '''
        initialize state vector
        '''
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.batch_size, self.state_width]))
        return state_vec_list

    def calc_gate(self, x, Mh, Mg):
        h = self.h_list[self.times - 1]
        net = tf.matmul(h, Mh) + tf.matmul(x, Mg)
        gate = tf.sigmoid(net)
        return gate

    def calc_gate1(self, x, Mh, Mg):
        h = self.h_list[self.times - 1]
        net = tf.matmul(h, Mh) + tf.matmul(x, Mg)
        gate = tf.tanh(net)
        return gate

    def forward(self, x):
        '''
        forward calculation
        '''

        while self.times < 60:
            self.times += 1
            # print(tf.shape(x))
            ft = self.calc_gate(x[:, self.times - 1, :], self.Mfh, self.Mfg)
            self.ft_list.append(ft)
            it = self.calc_gate(x[:, self.times - 1, :], self.Mih, self.Mig)
            self.it_list.append(it)
            ot = self.calc_gate(x[:, self.times - 1, :], self.Moh, self.Mog)
            self.ot_list.append(ot)
            ut = self.calc_gate1(x[:, self.times - 1, :], self.Mch, self.Mcg)
            self.ut_list.append(ut)
            c = ft * self.c_list[self.times - 1] + it * ut
            self.c_list.append(c)
            h = ot * tf.tanh(c)
            self.h_list.append(h)
        return self.h_list
