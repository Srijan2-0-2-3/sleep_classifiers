import pandas as pd
import numpy as np
from source.analysis.setup.subject_builder import SubjectBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor
import pickle
import copy
from sklearn.model_selection import train_test_split


def create_csv():
    SubjectBuilder.get_subject_dictionary()


def update_csv():
    subject_ids = SubjectBuilder.get_all_subject_ids()
    for subject_id in subject_ids:
        print(subject_id)
        df = pd.read_csv(f'{subject_id}.csv', index_col=0)
        print(df.head())
        print(df.shape)
        print(df.dtypes)
        columns = ['count', 'hr', 'time', 'cosine', 'sleep_label']
        for column in columns:
            df[column] = df[column].apply(lambda x: str(x.split()[0].replace('[', '')))
            df[column] = df[column].apply(lambda x: str(x.split()[0].replace(']', '')))
            df[column] = df[column].astype(float)
        print(df.head())
        df.to_csv(f'{subject_id}.csv')
        # df = df.apply(pd.to_numeric)
        print(df.dtypes)


def load_csv():
    subject_ids = SubjectBuilder.get_all_subject_ids()
    for subject_id in subject_ids:
        df = pd.read_csv(f'{subject_id}.csv', index_col=0)
        # print(df.columns)
        # print(df.shape)
        # array = df.to_numpy()
        X = np.concatenate([df[['count', 'hr', 'time', 'cosine']].to_numpy()])
        # print(X.shape)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        print(f'{subject_id} standardised')
        Y = df['sleep_label'].to_numpy()

        Y = np.around(Y)

        Y = abs(Y.astype(np.int_))
        print(f'{subject_id} labelled')
        X = X[(Y > 0) & (Y < 5)]
        Y = Y[(Y > 0) & (Y < 5)]

        print(subject_id, "Cleaned")
        assert len(X) == len(Y)
        # count = 0
        # prev = Y[0]
        # LenSubsequences = []
        # for elem in Y:
        #     if (elem != prev):
        #         LenSubsequences.append(count)
        #         count = 0
        #     count += 1
        #     prev = elem
        # SubsequencesX = []
        # SubsequencesY = []
        # i = 0
        # for elem in LenSubsequences:
        #     for j in range(0, elem, 100):
        #         if (j + 100 <= elem):
        #             SubsequencesX.append(X[i + j:i + j + 100])
        #             SubsequencesY.append(Y[i + j + 50])
        #     i += elem
        # print(SubsequencesX)
        # assert len(SubsequencesX) == len(SubsequencesY)
        # X_WES = (np.array(SubsequencesX, dtype=np.float32)).reshape(-1, 100, 14)
        #
        # # Le etichette 0 e 5, 6, 7 sono state tolte, quindi si spostano le rimanenti
        # # da 1 - 4 a 0 - 3
        # Y = np.array(SubsequencesY, dtype=np.int_) - 1
        # # print(Y)
        # # y_WES = to_categorical(Y, num_classes = 4)
        # y_WES = Y
        #
        # # Selezione di 100 sottosequenze per ogni etichetta dal soggetto
        # idx = np.argsort(Y)
        # SubY, SubX = np.array(Y)[idx], np.array(X_WES)[idx]
        # count = 0
        # # print(SubY)
        # prev = SubY[0]
        # SubsequencesX, X = [], []
        # SubsequencesY, Y = [], []
        # for i, elem in enumerate(SubY):
        #     if (elem != prev):
        #         count = 0
        #     count += 1
        #     if (count <= 100):
        #         SubsequencesX.append(SubX[i])
        #         SubsequencesY.append(elem)
        #     prev = elem
        #
        # X_WES = (np.array(SubsequencesX, dtype=np.float32))
        # Y = np.array(SubsequencesY, dtype=np.int_)
        # # y_WES = to_categorical(Y, num_classes = 4)
        # y_WES = Y
        # print(subject_id, "Subsequences")
        X_WES = np.array(X, dtype=np.float32)
        print(X_WES.shape)
        y_WES = np.array(Y, dtype=np.int_)
        print(y_WES.shape)
        # Salvataggio del soggetto
        with open("/home/srijan/PycharmProjects/sleep_classifiers/X" + subject_id + ".pkl", 'wb') as handle:
            pickle.dump(X_WES, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("/home/srijan/PycharmProjects/sleep_classifiers/y" + subject_id + ".pkl", 'wb') as handle:
            pickle.dump(y_WES, handle, protocol=pickle.HIGHEST_PROTOCOL)
    X, y = None, None
    for S in subject_ids:
        Xs = pickle.load(open("/home/srijan/PycharmProjects/sleep_classifiers/X" + S + ".pkl", 'rb'),
                         encoding='latin1')
        ys = pickle.load(open("/home/srijan/PycharmProjects/sleep_classifiers/y" + S + ".pkl", 'rb'),
                         encoding='latin1')

        if (X is None):
            X = copy.deepcopy(Xs)
            y = copy.deepcopy(ys)
        else:
            X = np.concatenate([X, Xs], axis=0)
            y = np.concatenate([y, ys], axis=0)
        del Xs
        del ys
        print("Loaded " + S)

    print("Dataset loaded")
    _, Xts, _, yts = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=42)
    print(str(Xts.shape) + " " + str(yts.shape))
    train, targets = [], []
    for i, elem in enumerate(Xts):
        train.append(elem)
        targets.append(yts[i])
    with open("/home/srijan/PycharmProjects/sleep_classifiers/Xts.pkl", 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/srijan/PycharmProjects/sleep_classifiers/yts.pkl", 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Rimozione dei dati usati nel test set dai vari soggetti
    for S in subject_ids:
        Xs = pickle.load(open("/home/srijan/PycharmProjects/sleep_classifiers/X" + S + ".pkl", 'rb'),
                         encoding='latin1')
        ys = pickle.load(open("/home/srijan/PycharmProjects/sleep_classifiers/y" + S + ".pkl", 'rb'),
                         encoding='latin1')
        print(S + " " + str(Xs.shape) + " " + str(ys.shape), end=" -> ")
        j = []
        for xts in Xts:
            for i, xs in enumerate(Xs):
                if (xts == xs).all():
                    j.append(i)

        Xs = np.delete(Xs, j, axis=0)
        ys = np.delete(ys, j, axis=0)

        print(str(Xs.shape) + " " + str(ys.shape))

        train, targets = [], []
        for i, elem in enumerate(Xs):
            train.append(elem)
            targets.append(ys[i])
        with open("/home/srijan/PycharmProjects/sleep_classifiers/X" + S + ".pkl", 'wb') as handle:
            pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("/home/srijan/PycharmProjects/sleep_classifiers/y" + S + ".pkl", 'wb') as handle:
            pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    load_csv()
