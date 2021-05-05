import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np


# 將輸入的數字value做二元編碼（因為要向量化vectorization），digit_count是碼數 ex. (3, 3) = [1,1,0] ； (11, 5) = [1,1,0,1,0]
def encore_binary(value, digit_count):
    return np.array([value >> d & 1 for d in range(digit_count)])


# fizz buzz的輸出為one hot -> [otherwise, fizz, buzz, fizzbuzz]
def one_hot_fizz_buss(value):
    if value % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif value % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif value % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])


#  training dataset preparing
DIGIT_COUNT = 10
# x = 101~1024的數字
train_x = np.array([encore_binary(i, DIGIT_COUNT) for i in range(101, 2 ** DIGIT_COUNT)])
# y = one hot過的 fizz buzz
train_y = np.array([one_hot_fizz_buss(i) for i in range(101, 2 ** DIGIT_COUNT)])

# TensorFlow parameter preparing
HIDDEN_UNIT_COUNT = 100
# input is n x digit_count matrix with float value
# output is n x 4 matrix with float value
X = tf.placeholder('float', [None, DIGIT_COUNT])
Y = tf.placeholder('float', [None, 4])


# initial weight randomly
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 初始化 hidden layer的weight
w_h = init_weights([DIGIT_COUNT, HIDDEN_UNIT_COUNT])
# 初始化 output的weight
w_o = init_weights([HIDDEN_UNIT_COUNT, 4])


# define model
def model(x_input, w_hidden, w_output):
    # activate with relu function
    h = tf.nn.relu(tf.matmul(x_input, w_hidden))
    return tf.matmul(h, w_output)


# define optimizer (gradient descent with eta = 0.05)
pre_y = model(X, w_h, w_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre_y, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
pre_opt = tf.argmax(pre_y, 1)


# convert vectorized output into string output
def vec2str(value_input, prediction):
    return [str(value_input), "fizz", "buzz", "fizzbuzz"][prediction]


BATCH_SIZE = 128
# start training
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(10000):
        # 產生長度為train_x的隨機序列
        p = np.random.permutation(range(len(train_x)))
        train_x, train_y = train_x[p], train_y[p]

        # 每批次BATCH_SIZE個數據丟去訓練
        for start in range(0, len(train_x), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(optimizer, feed_dict={X: train_x[start:end], Y: train_y[start:end]})

        # print(epoch, np.mean(np.argmax(train_y, axis=1)
        #                      == sess.run(pre_opt, feed_dict={X: train_x, Y: train_y})))

    # model has been trained so test it
    numbers = np.arange(1, 101)
    test_x = np.transpose(encore_binary(numbers, DIGIT_COUNT))
    test_y = sess.run(pre_opt, feed_dict={X: test_x})

    output = np.vectorize(vec2str)(numbers, test_y)
    print(output)