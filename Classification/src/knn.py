
import numpy as np
import sys
import tensorflow as tf
import numpy
from sklearn.preprocessing import StandardScaler
from src import function as func


# python knn.py 1 ../Data/Res/train_features.csv ../Data/Res/train_labels.csv ../Data/Res/eval_features.csv ../Data/Res/test.csv 0 193 3000 20 ../Data/csv/resultatKNNmfcc.csv
if __name__ == '__main__':
    if len(sys.argv) != 11:
        print("Usage : python3 knn.py <iftrain> <train_features> <train_label> <eval_feature> <eval_id_music> <start_features> <end_features> <nbTrain> <nbVoisinage> <save_dest>")
        sys.exit()

    # Input
    isTrain = int(sys.argv[1])
    train_features = sys.argv[2]
    train_label = sys.argv[3]
    eval_feature = sys.argv[4]
    eval_id_music = sys.argv[5]
    start_features = int(sys.argv[6])
    end_features = int(sys.argv[7])
    nbTrain = int(sys.argv[8])
    nbVoisinage = int(sys.argv[9])
    savepath = sys.argv[10]


    tr_features, tr_labels, ts_features, ts_labels = func.extract(isTrain, train_features, train_label, nbTrain, start_features, end_features)


    #placeholders for variable to be used in model
    xtr=tf.placeholder(tf.float32,[None,tr_features.shape[1]]) #traning input
    ytr=tf.placeholder(tf.float32,[None,8]) #traning label
    xte=tf.placeholder(tf.float32,[tr_features.shape[1]]) #testing input

    #K-near
    K=nbVoisinage
    nearest_neighbors=tf.Variable(tf.zeros([K]))

    distance = tf.negative(tf.reduce_sum(tf.abs(tf.subtract(xtr, xte)),axis=1)) #L1
    values,indices=tf.nn.top_k(distance,k=K,sorted=False)

    #a normal list to save
    nn = []
    for i in range(K):
        nn.append(tf.argmax(ytr[indices[i]], 0)) #taking the result indexes

    nearest_neighbors=nn
    y, idx, count = tf.unique_with_counts(nearest_neighbors)

    pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]

    #setting accuracy as 0
    accuracy=0

    #initialize of all variables
    init=tf.global_variables_initializer()
    if isTrain == 0:
        dataset_test = numpy.loadtxt(eval_id_music, delimiter=",",dtype=np.str)
        res = []
        res.append(['track_id', 'genre_id'])
        #changement de l ensemble d evalutation
        ts_features1 = np.loadtxt(open(eval_feature, "rb"), delimiter=",", dtype=np.str)
        ts_features = ts_features1[:, start_features:end_features];
        print(len (ts_features))
        sc1 = StandardScaler().fit(ts_features)
        ts_features = sc1.transform(ts_features)


    #start of tensor session
    with tf.Session() as sess:

        for i in range(ts_features.shape[0]):
            predicted_value=sess.run(pred,feed_dict={xtr:tr_features,ytr:tr_labels,xte:ts_features[i,:]})
            if isTrain == 0:
                res.append([dataset_test[i], predicted_value+1])
            else :
                if predicted_value == np.argmax(ts_labels[i]):
                    # if the prediction is right then a double value of 1./200 is added 200 here is the number of test
                        accuracy += 1. / len(ts_features)

        writer = tf.summary.FileWriter('./graphs',sess.graph)
        writer.close()


        if isTrain == 0:
            res.append(['098559', predicted_value])
            res.append(['098571', predicted_value])
            np.savetxt(savepath, res, delimiter=",", fmt="%s")
        else:
            print(K, "-th neighbors' Accuracy is:", accuracy)

        print("Calculation completed ! ! ")
