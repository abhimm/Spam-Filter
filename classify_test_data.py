__author__ = 'Abhinav'
import numpy as np
import pickle


def classify_naive_bayes(test_data_file_name, test_label_file_name):
    # load pre-processed data
    test_data_file = open(test_data_file_name, 'rb')
    test_label_file = open(test_label_file_name, 'rb')
    test_data = test_data_file.readlines()
    test_label = test_label_file.readlines()

    vocabulary = pickle.load(open('vocabulary.p', 'rb'))
    word_probability_ham = pickle.load(open('word_probability_ham.p', 'rb'))
    word_probability_spam = pickle.load(open('word_probability_spam.p', 'rb'))
    unobserved_word_probability = pickle.load(open('unobserved_word_prob.p', 'rb'))
    class_probability = pickle.load(open('class_probability.p', 'rb'))
    test_label_mod = np.zeros(len(test_label), dtype=int)

    print "Data Loaded"
    # transform test label fro, ham spam to 0 1
    for i in range(len(test_label)):
        temp = test_label[i].strip().split()
        if temp[0] == 'spam':
            test_label_mod[i] = 1

    # classify every test instance
    test_result = np.zeros(len(test_data), dtype=int)

    index = 0
    for index in range(len(test_data)):
        feature = test_data[index].strip().split()
        cond_prob_ham = 0.0
        cond_prob_spam = 0.0
        i = 2
        while i < len(feature):
            if vocabulary.has_key(feature[i]):
                cond_prob_ham += (float(feature[i+1]) * word_probability_ham[vocabulary[feature[i]]])
                cond_prob_spam += (float(feature[i+1]) * word_probability_spam[vocabulary[feature[i]]])
            else:

                cond_prob_ham += (float(feature[i+1]) * unobserved_word_probability[0])
                cond_prob_ham += (float(feature[i+1]) * unobserved_word_probability[1])
            i += 2

        cond_prob_ham += class_probability[0]
        cond_prob_spam += class_probability[1]

        if cond_prob_ham < cond_prob_spam:
            test_result[index] = 1

    # print accuracy
    print "Accuracy: ", 100*float(np.where(test_result == test_label_mod)[0].shape[0])/len(test_label)


def main():

    classify_naive_bayes('test', 'test.index')


if __name__ == '__main__':
    main()