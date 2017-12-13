import numpy as np
from sklearn.preprocessing import StandardScaler
import random

def extract(isTrain, train_features, train_label, nbTrain, start, end):
    tr_features1 = np.loadtxt(open(train_features, "rb"), delimiter=",", dtype=np.str)
    tr_label1 = np.loadtxt(open(train_label, "rb"), delimiter=",", dtype=np.int)

    if (isTrain == 0):
        nbTrain = len(tr_features1)

    tmp = []
    for i in range(len (tr_features1)):
        tmp.append(i)

    random.seed()
    random.shuffle(tmp)

    statTrain = [0,0,0,0,0,0,0,0]
    statEval = [0,0,0,0,0,0,0,0]

    tr_features = []
    tr_labels = []
    for i in range(0, nbTrain):
        tr_features.append(tr_features1[tmp[i], start:end])
        tr_labels.append(tr_label1[tmp[i]])
        statTrain[tr_label1[tmp[i]]-1] += 1
    print(statTrain)
    sc = StandardScaler().fit(tr_features)
    tr_features = sc.transform(tr_features)
    tr_labels = one_hot_encode(tr_labels)


    ts_features = []
    ts_labels = []
    if isTrain == 1:
        for i in range(nbTrain,len (tr_features1)):
            ts_features.append(tr_features1[tmp[i], start:end])
            ts_labels.append(tr_label1[tmp[i]])
            statEval[tr_label1[tmp[i]]-1] += 1
        print(statEval)

        sc1 = StandardScaler().fit(ts_features)
        ts_features = sc1.transform(ts_features)

        ts_labels = one_hot_encode(ts_labels)

    return tr_features, tr_labels, ts_features, ts_labels


def extractBinary(isTrain, train_features, train_label, nbTrain, start, end, classe,seed):
    tr_features1 = np.loadtxt(open(train_features, "rb"), delimiter=",", dtype=np.str)
    tr_label1 = np.loadtxt(open(train_label, "rb"), delimiter=",", dtype=np.int)

    if (isTrain == 0):
        nbTrain = len(tr_features1)

    tmp = []
    for i in range(len (tr_features1)):
        tmp.append(i)
    if (seed == 0):
        s = random.seed()
    else:
        seed = random.seed(seed)
    random.shuffle(tmp)

    statTrain = [0,0]
    statEval = [0,0]

    tr_features = []
    tr_labels = []
    for i in range(0, nbTrain):
        tr_features.append(tr_features1[tmp[i], start:end])
        if (tr_label1[tmp[i]] == classe):
            tr_labels.append(0)
            statTrain[0] += 1
        else:
            tr_labels.append(1)
            statTrain[1] += 1

    tr_features_tmp = []
    tr_labels_tmp = []

    cp = 0;
    cp1 = 0
    for i in range(0, nbTrain):
        if (tr_labels[i] == 0 and cp < statTrain[0]):
            tr_features_tmp.append(tr_features1[tmp[i], start:end])
            tr_labels_tmp.append(0)
            cp += 1
        if tr_labels[i] == 1 and cp1 < statTrain[0]:
                tr_features_tmp.append(tr_features1[tmp[i], start:end])
                tr_labels_tmp.append(1)
                cp1 += 1

    tr_features = tr_features_tmp
    tr_labels = tr_labels_tmp
    sc = StandardScaler().fit(tr_features)
    tr_features = sc.transform(tr_features)
    tr_labels = one_hot_encode(tr_labels)


    ts_features = []
    ts_labels = []
    if isTrain == 1:
        for i in range(nbTrain,len (tr_features1)):
            ts_features.append(tr_features1[tmp[i], start:end])

            if (tr_label1[tmp[i]] == classe):
                ts_labels.append(0)
                statEval[0] += 1
            else:
                ts_labels.append(1)
                statEval[1] += 1
        print(statEval)

        ts_features_tmp = []
        ts_labels_tmp = []

        cp = 0;
        cp1 = 0
        for i in range(nbTrain,len (tr_features1)):
            if (ts_labels[i-nbTrain] == 0 and cp < statEval[0]):
                ts_features_tmp.append(tr_features1[tmp[i], start:end])
                ts_labels_tmp.append(0)
                cp += 1
            if ts_labels[i-nbTrain] == 1 and cp1 < statEval[0]:
                ts_features_tmp.append(tr_features1[tmp[i], start:end])
                ts_labels_tmp.append(1)
                cp1 += 1

        ts_features = ts_features_tmp
        ts_labels = ts_labels_tmp
        print(len(ts_labels))
        sc1 = StandardScaler().fit(ts_features)
        ts_features = sc1.transform(ts_features)

        ts_labels = one_hot_encode(ts_labels)

    return tr_features, tr_labels, ts_features, ts_labels, seed


def one_hot_encode(labels):
    n_labels = len(labels)
    ## passer de 1-8 a 0-7
    for i in range(0, len(labels)):
        labels[i] = labels[i] - 1
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def get_batch(inputX, inputY, batch_size):
  duration = len(inputX)
  c = list(zip(inputX, inputY))
  inputX, inputY = zip(*c)

  for i in range(0,duration//batch_size):
    idx = i*batch_size
    yield inputX[idx:idx+batch_size], inputY[idx:idx+batch_size]