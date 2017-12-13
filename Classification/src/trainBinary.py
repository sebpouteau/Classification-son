import sys
import numpy as np
import tensorflow as tf
from src import function as func
from sklearn.preprocessing import StandardScaler
import random

# python knn.py ../Data/Res/train_features.csv ../Data/Res/train_labels.csv ../Data/Res/eval_features.csv 0 193 3000 1000 2 ../Data/csv/tt.csv
if __name__ == '__main__':
    if len(sys.argv) != 11:
        print("Usage : python3 knn.py  <train_features> <train_label> <eval_feature> <start_features> <end_features> <nbTrain> <nbEpoch> <nbClasse> <save_dest>")
        sys.exit()

    # Input
    train_features = sys.argv[1]
    train_label = sys.argv[2]
    eval_feature = sys.argv[3]
    start_features = int(sys.argv[4])
    end_features = int(sys.argv[5])
    nbTrain = int(sys.argv[6])
    nbEpoch = int(sys.argv[7])
    nbClasse = int(sys.argv[8])
    savepath = sys.argv[9]
    seed = 0
    for i in range(8):
        tr_features, tr_labels, ts_features, ts_labels, seed = func.extractBinary(1, train_features, train_label, nbTrain, start_features, end_features, i+1, seed)

        training_epochs = nbEpoch
        n_dim =  tr_features.shape[1]
        n_classes = nbClasse
        n_hidden_units_one = 70
        n_hidden_units_two = 140
        sd = 1 / np.sqrt(n_dim)
        learning_rate = 0.001
        batch_size = 100

        X = tf.placeholder(tf.float32,[None,n_dim], name="X")
        Y = tf.placeholder(tf.float32,[None,n_classes], name="Y")

        W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd), name="W_1")
        b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd), name="b_1")
        h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd), name="W_2")
        b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd), name="b_2")
        h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

        W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd), name="W" )
        b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name="b")
        y_ = tf.nn.softmax(tf.matmul(h_2,W) + b, name="y_")

        init = tf.global_variables_initializer()


        #cost_function = tf.reduce_mean(tf.pow(tf.subtract(y_, Y), 2))
        cost_function = -tf.reduce_sum(Y * tf.log(y_))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

        correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        cost_history = np.empty(shape=[1], dtype=float)
        y_true, y_pred = None, None

        saver = tf.train.Saver()


        with tf.Session() as sess:
                sess.run(init)
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    for batch_x, batch_y in func.get_batch(tr_features, tr_labels, batch_size):
                        t, cost = sess.run([optimizer, cost_function], feed_dict={X: batch_x, Y: batch_y})
                        avg_cost += cost / batch_size

                    cost_history = np.append(cost_history, cost)

                print("it ", epoch, " Test accuracy: ", round(sess.run(accuracy,
                                                           feed_dict={X: ts_features, Y: ts_labels}), 5), "cost", avg_cost)
                y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: ts_features})
                y_true = sess.run(tf.argmax(ts_labels, 1))
                save_path = saver.save(sess, savepath  + str(i + 1) + "/model" + str(i + 1))

