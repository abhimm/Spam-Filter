__author__ = 'Abhinav'
import numpy as np
import pickle


def transform_training_data(train_data_raw_file_name, train_label_raw_file_name):

    train_data_raw_file = open(train_data_raw_file_name, 'rb')
    train_label_raw_file = open(train_label_raw_file_name, 'rb')

    train_data_raw = train_data_raw_file.readlines()
    train_label_raw = train_label_raw_file.readlines()

    # get vocabulary to generate features
    vocabulary_file = open('vocabulary.p', 'rb')
    vocabulary = pickle.load(vocabulary_file)

    # generate train label
    train_label = np.zeros((len(train_label_raw)))
    for i in range(len(train_label_raw)):
        temp = train_label_raw[i].strip().split()
        if temp[0] == 'ham':
            train_label[i] = 0
        else:
            train_label[i] = 1

    train_label_file = open('train_label.p', 'wb')
    pickle.dump(train_label, train_label_file)

    # transform training data
    train_data = np.zeros((len(train_data_raw), len(vocabulary)))
    for i in range(len(train_data_raw)):
        temp = train_data_raw[i].strip().split()

        j = 2
        while j < len(temp):
            train_data[i, vocabulary[temp[j]]] = float(temp[j+1])
            j += 2
    train_data_file = open('train_data.p', 'wb')
    pickle.dump(train_data, train_data_file)

    train_label_file.close()
    train_data_file.close()
    vocabulary_file.close()
    train_data_raw_file.close()
    train_label_raw_file.close()


def main():

    transform_training_data('train', 'train.index')


if __name__ == '__main__':
    main()
