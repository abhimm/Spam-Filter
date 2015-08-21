__author__ = 'Abhinav'
from classify_test_data import classify_naive_bayes
from gen_feature import gen_feature
from gen_vocabulary import gen_vocab
from transform_training_data import transform_training_data


def main():
    # initialize file names to be used later
    train_data_file_name = 'train'
    train_label_file_name = 'train.index'
    test_data_file_name = 'test'
    test_label_file_name = 'test.index'

    # generate vocabulary
    gen_vocab(train_data_file_name)
    # transform training data
    transform_training_data(train_data_file_name, train_label_file_name)
    # generate feature from the training data
    gen_feature()
    # classify test instance and print accuracy
    classify_naive_bayes(test_data_file_name, test_label_file_name)


if __name__ == '__main__':
    main()