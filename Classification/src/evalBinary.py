import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy
from sklearn.preprocessing import StandardScaler
import sys

def execute(i):
    with tf.Session() as sess:
        # Restore variables from disk.
        saver = tf.train.import_meta_graph("Data/model/model" + str(i + 1) + "/model" + str(i + 1) + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint("Data/model/model" + str(i + 1) + "/."))
        print("Model restored")

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        y_ = graph.get_tensor_by_name("y_:0")
        W_1 = graph.get_tensor_by_name("W_1:0")

        return sess.run(y_, feed_dict={X: ts_features})
        # tab.append(y_.eval(session=sess, feed_dict={X: ts_features}))


# python eval.py model5.meta Data/Res/eval_features.csv test.csv Data/csv/resultat5.csv
if __name__ == '__main__':
    if len(sys.argv) != 7:
        print(
            "Usage : python3 knn.py <model> <eval_feature> <eval_id_music> <start_features> <end_features>  <save_dest>")
        sys.exit()

    # Input
    model = sys.argv[1]
    eval_feature = sys.argv[2]
    eval_id_music = sys.argv[3]
    start_features = int(sys.argv[4])
    end_features = int(sys.argv[5])
    savepath = sys.argv[6]

    dataset_test = numpy.loadtxt(eval_id_music, delimiter=",", dtype=np.str)
    ts_features1 = np.loadtxt(open(eval_feature, "rb"), delimiter=",", dtype=np.str)

    ts_features = ts_features1[:, start_features:end_features];
    sc1 = StandardScaler().fit(ts_features)
    ts_features = sc1.transform(ts_features)

    tab = []

    cpt = -1
    for i in range(5):
        tab.append(execute(i))
    for i in range(len(dataset_test)):
        for j in range(5):
            print(tab[j][i], end=" ")
        print()

    res = []

    res.append(['track_id', 'genre_id'])

    for i in range(len(dataset_test)):
        max = 0
        classe = 0
        for j in range(8):
            if tab[j][i][0] > max:
                max = tab[j][i][0]
                classe = j + 1

        res.append([dataset_test[i], classe])

    res.append(['098559', 1])
    res.append(['098571', 1])

    np.savetxt(savepath, res, delimiter=",", fmt="%s")
