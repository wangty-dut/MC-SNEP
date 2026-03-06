import tensorflow.compat.v1 as tf
class mlp(object):
    def __init__(self,input_width,state_width,output_width, batch_size):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.output_width = output_width
        self.weights1 = tf.Variable(tf.random_normal([self.state_width,self.output_width], stddev=0.1))
        self.bias1 = tf.Variable(tf.random_normal([self.batch_size, self.output_width], stddev=0.1))
        self.weights2 = tf.Variable(tf.random_normal([self.output_width, self.input_width], stddev=0.1))
        self.bias2 = tf.Variable(tf.random_normal([self.batch_size, self.input_width], stddev=0.1))

    def forward(self,x):
        result1 = tf.nn.relu(tf.matmul(x, self.weights1) + self.bias1)
        result2 = tf.matmul(result1, self.weights2)  + self.bias2
        return result2