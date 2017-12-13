import librosa
import numpy as np
import sys


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)


    centroid = np.mean(librosa.feature.spectral_centroid(S=stft, sr=sample_rate).T,axis=0)

    rolloff = np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sample_rate).T,axis=0)

    tempo = np.mean(librosa.feature.tempogram(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)

    poly = np.mean(librosa.feature.poly_features(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs,chroma,mel,contrast,tonnetz, centroid, rolloff, tempo, poly


def parse_files(ids_data, folder, out_directory, trainOrTest, ifTrain, file_ext=".mp3"):
    features, labels = np.empty((0, 581)), np.empty(0)

    nb = 0
    for id in ids_data:
        file_id = id
        if ifTrain == 1:
            file_label = id[1]
            file_id = id[0]

        filepath = folder + str(file_id) + file_ext

        try:
            mfccs, chroma, mel, contrast, tonnetz, centroid, rolloff, tempo, poly = extract_feature(filepath)
        except Exception as e:
            print("Error encountered while parsing file: ", filepath)
            continue

        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz, centroid, rolloff, tempo, poly])
        features = np.vstack([features, ext_features])
        if ifTrain == 1:
            labels = np.append(labels, file_label)
        nb += 1
        print("" + str(nb) + " files parsed" + " Name: " + file_id)

        # save
        tmp_features = np.array(features)
        tmp_labels = np.array(labels, dtype=np.int)

        np.savetxt("" + out_directory + "/" + trainOrTest + "_features.csv", tmp_features, delimiter=",")
        if ifTrain == 1:
            np.savetxt("" + out_directory + "/" + trainOrTest + "_labels.csv", tmp_labels.astype(int), delimiter=",",
                   fmt='%i')
    if ifTrain == 1:
        return np.array(features), np.array(labels, dtype=np.int)
    else:
        return np.array(features)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage : python3 extractFile.py <Train / Eval> <data_ids_file.csv> <data_folder> <output_directory>")
        print("Train = 1, Eval = 0")
        sys.exit()

    # Input
    ifTrain = int(sys.argv[1])
    data_ids_file = sys.argv[2]
    data_folder = sys.argv[3]
    out_directory = sys.argv[4]

    set_ids = np.loadtxt(open(data_ids_file, "rb"), delimiter=",", skiprows=0, dtype=np.str)

    if ifTrain == 1:
        print("Generate file for Train and Test")

        train_features, train_labels = parse_files(set_ids, data_folder, out_directory, "train", ifTrain)

        np.savetxt("" + out_directory + "/train_features.csv", train_features, delimiter=",")
        np.savetxt("" + out_directory + "/train_labels.csv", train_labels.astype(int), delimiter=",", fmt='%i')
    else :
        print("Generate file for evaluation")
        train_features, train_labels = parse_files(set_ids, data_folder, out_directory, "eval", ifTrain)

        np.savetxt("" + out_directory + "/eval_features.csv", train_features, delimiter=",")
