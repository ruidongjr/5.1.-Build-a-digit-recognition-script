"""
5. Machine learning
    5.1. Build a digit recognition script
    This script adopt XGBoost algorithm.
    step 1: Unzip digits.zip
    step 2: Read training & testing dataset
    step 3: Process data
    step 4: Comment alternatively:
    step 4.1: Method 1 => train XGBoost model and output score
    step 4.2: Method 2 => train DNN model and output score

    XGBoost
    Estimated Running Time:  4526s
    Test Accuracy:  97.2%

    DNN
    Estimated Running Time:  < 1800s
    Test Accuracy:  theoretically >98.4%   Note: Current model is not sufficiently trained due to time limitation.

    Future works:
        1. More conventional machine learning methods can be adopted, such as lightgbm and catboost. Sometime, regression
    methods with the help of bagging and boosting, ending with an optimized rounder function, can lead to better accuracy.
        2. Deep learning methods can be adopted, similar to the code for section 5.2, where CNN is developed.

    Note: Please place this script within the same folder of digits.zip

"""

# importing library
import time
import numpy as np
from zipfile import ZipFile
import os
import cv2
import xgboost as xgb #Import XGBoost package
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt # plot


def unzip(path='digits.zip'):
    with ZipFile(path, 'r') as zip_parent:
        # extracting all the files
        print('Extracting all the files now...')
        zip_parent.extractall()
        print('Done')


def read_data(path):
    data = []
    label = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Read image in Grayscale
            data.append(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE))
            label.append(float(root.split('\\')[-1]))
    return data, label


def process_data(X, y, X_test):
    """
    This function reshape train + test data; split training + val data: 90% vs 10%; augments data;
    :param X: Training data set
    :param y: Training data labels
    :param X_test: Testing data set
    :return: processed data sets
    """

    # data augmentation
    # datagen = ImageDataGenerator(
    #     rotation_range=15,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True,
    # )
    # datagen.fit(X[:,:,:,np.newaxis])

    # Flatten from 3-D array to 2-D array
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    X_test_new = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_test_new = sc.fit_transform(X_test_new)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=2)



    # Making sure that the values are float so that we can get decimal points after division
    X_train = X_train.astype('float32')
    X_test_new = X_test_new.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    X_train /= 255
    X_test_new /= 255

    return X_test_new, X_train, X_valid, y_train, y_valid


def xgb_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    This function run XGBoost algorithm
    :param X_train: train data
    :param y_train: train label
    :param X_valid: validate data
    :param y_valid: validate label
    :param X_test: test data
    :param y_test: test label
    :return: accuracy score
    """

    # Reading in data to DMatrix, an internal data structure used by XGBoost
    dmatTrain = xgb.DMatrix(X_train, label=y_train)
    dmatVal = xgb.DMatrix(X_valid, label=y_valid)
    dmatTest = xgb.DMatrix(X_test, label=y_test)

    param = {'booster': 'gbtree',  # gbtree is the default tree based model
             'verbosity': 1,  # Specify the amount of information to receive in an error message
             'max_depth': 7,  # Specify maximum tree depth #6
             'eta': 0.08, # Step size shrinkage used in update to prevent overfitting
             'objective': 'multi:softmax',  # Specify multiclass classification
             'num_class': 10,  # Specify number of class labels
             'subsample': 0.8,
             'colsample_bytree': 0.8,
             'alpha': 3,
             'lambda': 2,
             'eval_metric': 'merror'} # Multiclass classification error rate = #(wrong cases)/#(all cases)

    num_round = 550  # Specify the number of rounds for boosting
    early_stopping = 50

    eval_list = [(dmatTrain, "train"), (dmatVal, "validation")]

    print('Start training ...')
    start = time.time()
    bst = xgb.train(param, dmatTrain, num_round, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)
    print('training consumed: ', time.time() - start)
    train_pred = bst.predict(dmatTrain)  # Predict classes in the train data using the trained model

    print('Train Accuracy:', sum(train_pred == y_train) / len(train_pred))  # Calculate train accuracy
    test_pred = bst.predict(dmatTest)  # Predict classes in test data using the trained model
    print("Test Accuracy:", sum(test_pred == y_test) / len(test_pred))


class DNN:
    '''
      deep neural network (DNN)
      arg:  size = # layers + # neurons/layer
            lr = learning rate
            num_iter_train = training epochs
            num_iter_val = validation epochs
            checkpoint_path = to restore a trained DNN
            batch_size = mini-batch process
            memory_size = limited # training samples
      input: generated features in X
      output: predicted category in y

      note: tensorflow 2.0 is adopted
    '''

    def __init__(self, size, lr, num_iter_train, num_iter_val, checkpoint_path, batch_size, memory_size):
        # initialize hyper-parameters, parameters of DNN
        self.checkpoint_path = checkpoint_path
        self.size = size
        self.learning_rate = lr
        self.num_iter_train = num_iter_train
        self.num_iter_val = num_iter_val
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.save_iter = 1000
        self.cost_his = []
        self.sess = tf.compat.v1.Session()



        # build structure

    def _build_net(self):

        # helper function for _build_net
        def build_layer(l_name, input, in_size, out_size, w_initializer, activation_func=None):
            with tf.compat.v1.variable_scope('layer%d' % l_name):
                w = tf.compat.v1.get_variable('w%d' % l_name, [in_size, out_size], initializer=w_initializer)
                b = tf.compat.v1.get_variable('b%d' % l_name, [out_size], initializer=tf.constant_initializer(0.1))
                wxPb = tf.matmul(input, w) + b
                if activation_func is None:
                    output = wxPb
                else:
                    output = activation_func(wxPb)
            return output

        # tf convert from version 1.x to 2.0
        tf.compat.v1.disable_eager_execution()

        self.input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.size[0]), name='input')  #
        self.output = tf.compat.v1.placeholder(tf.float32, shape=(None, self.size[-1]), name='output')  #

        with tf.compat.v1.variable_scope('memory_net'):
            l1 = build_layer(1, self.input, self.size[0], self.size[1],
                             tf.random_normal_initializer(0., 2 / self.size[0]),
                             activation_func=tf.nn.relu)
            l2 = build_layer(2, l1, self.size[1], self.size[2], tf.random_normal_initializer(0., 2 / self.size[1]),
                             activation_func=tf.nn.relu)
            # l3 = build_layer(3, l2, self.size[2], self.size[3], tf.random_normal_initializer(0., 2 / self.size[2]),
            #                  activation_func=tf.nn.relu)
            # l4 = build_layer(4, l3, self.size[3], self.size[4], tf.random_normal_initializer(0., 2 / self.size[3]),
            #                  activation_func=tf.nn.relu)
            # l5 = build_layer(5, l4, self.size[4], self.size[5], tf.random_normal_initializer(0., 2 / self.size[4]), \
            #                  activation_func=tf.nn.relu)
            # l6 = build_layer(6, l5, self.size[5], self.size[6], tf.random_normal_initializer(0., 2 / self.size[5]), \
            #                  activation_func=tf.nn.relu)
            # l7 = build_layer(7, l6, self.size[6], self.size[7], tf.random_normal_initializer(0., 2 / self.size[6]), \
            #                  activation_func=tf.nn.relu)
            # l8 = build_layer(8, l7, self.size[7], self.size[8], tf.random_normal_initializer(0., 2 / self.size[7]), \
            #                  activation_func=tf.nn.relu)
            # l9 = build_layer(9, l8, self.size[8], self.size[9], tf.random_normal_initializer(0., 2 / self.size[8]), \
            #                  activation_func=tf.nn.relu)
            # l10 = build_layer(10, l9, self.size[9], self.size[10], tf.random_normal_initializer(0., 2 / self.size[9]), \
            #                   activation_func=tf.nn.relu)
            self.m_pred = build_layer(3, l2, self.size[2], self.size[-1],
                                      tf.random_normal_initializer(0., 1 / self.size[-1]), activation_func=None)

        with tf.compat.v1.variable_scope('loss'):
            # define loss function
            self.loss = tf.compat.v1.losses.mean_squared_error(labels=self.output, predictions=self.m_pred)

        with tf.compat.v1.variable_scope('train'):
            # define optimizer
            self._train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate, 0.9).minimize(self.loss,
                                                                                                name='train_op')

    # restore saved DNN
    # def _variables_to_restore(self, save_file, graph):
    #     # returns a list of variables that can be restored from a checkpoint
    #     reader = tf.train.NewCheckpointReader(save_file)
    #     saved_shapes = reader.get_variable_to_shape_map()
    #     var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
    #                         if var.name.split(':')[0] in saved_shapes])
    #     restore_vars = []
    #     for var_name, saved_var_name in var_names:
    #         curr_var = graph.get_tensor_by_name(var_name)
    #         var_shape = curr_var.get_shape().as_list()
    #         if var_shape == saved_shapes[saved_var_name]:
    #             restore_vars.append(curr_var)
    #     return restore_vars

    # training DNN
    def fit(self, X_train, y_train, X_val, y_val):

        self._build_net()

        # save model if needed
        # saver_all = tf.train.Saver(max_to_keep=1)
        # ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        initial_step = 0
        pred_his = []
        opt_his = []

        # train the model
        for i in range(initial_step, self.num_iter_train + self.num_iter_val):
            # training process
            if i < self.num_iter_train:
                input = X_train[i, :]
                my_pred = self.sess.run(self.m_pred, feed_dict={self.input: input[np.newaxis, :]})
                my_pred = np.squeeze(my_pred)
                pred_his.append(my_pred)
                opt_his.append(y_train[i])

                # consider DNN structure with limited memory setup
                if i > self.memory_size:
                    sample_index = np.random.permutation(np.arange(i - self.memory_size, i + 1))[
                                   :self.batch_size]  # unique
                else:
                    sample_index = np.random.choice(i + 1, size=self.batch_size)

                _, cost = self.sess.run([self._train_op, self.loss],
                                        feed_dict={self.input: X_train[sample_index, :],
                                                   self.output: y_train[sample_index, np.newaxis]})
                self.cost_his.append(cost)
            # validation process
            else:
                input = X_val[i - self.num_iter_train, :]
                my_pred = self.sess.run(self.m_pred, feed_dict={self.input: input[np.newaxis, :]})
                my_pred = np.squeeze(my_pred)
                pred_his.append(my_pred)
                opt_his.append(y_val[i - self.num_iter_train, np.newaxis])

            if (i + 1) % ((self.num_iter_train + self.num_iter_val) / 10) == 0:
                print('Progress to ', (i + 1) / (self.num_iter_train + self.num_iter_val), '. Cost is ',
                      self.cost_his[-1])

        return pred_his, opt_his

    # testing DNN
    def predict(self, X_test):
        predict = []
        for ii in range(X_test.shape[0]):
            input = X_test[ii, :]
            my_pred = self.sess.run(self.m_pred, feed_dict={self.input: input[np.newaxis, :]})
            my_pred = np.squeeze(my_pred)
            predict.append(my_pred)
        return predict

    # plot the training loss/cost
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost of DNN')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    # 1. Unzip the digits.zip file
    # unzip()

    # 2. Read training & testing dataset
    X, y = read_data(path='./data/mnist/training')
    X, y = np.array(X), np.array(y)
    X_test, y_test = read_data(path='./data/mnist/testing')
    X_test, y_test = np.array(X_test), np.array(y_test)

    # 3. process data
    X_test, X_train, X_valid, y_train, y_valid = process_data(X, y, X_test)

    # 4.1. Method 1 => train model and output score
    # xgb_pipeline(X_train, y_train, X_valid, y_valid, X_test, y_test)

    # 4.2. Method 2 => train DNN
    tf.compat.v1.reset_default_graph()

    my_DNN = DNN(size=[X_train.shape[1], 800, 800, 1],
                 lr=0.001,
                 num_iter_train=X_train.shape[0],
                 num_iter_val=X_valid.shape[0],
                 checkpoint_path='./ckpt',
                 batch_size=128,
                 memory_size=1024)

    y_pred, y_opt = my_DNN.fit(np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid))
    my_DNN.plot_cost()

    pr = my_DNN.predict(np.array(X_test))  # use trained DNN to predict X

    # TODO: fine-tune DNN
    print('-' * 30)
    print('Test accuracy:', sum(np.round(pr) == y_test) / y_test.shape[0])
    print('-' * 30)





