__author__ = 'Abhinav'
import pickle
import numpy as np
import math


def gen_feature():
    train_label_file = open('train_label.p', 'rb')
    train_feature_file = open('train_data.p', 'rb')
    train_label = pickle.load(train_label_file)
    train_feature = pickle.load(train_feature_file)

    # get class indices in training label data
    ham_indices = np.where(train_label == 0)[0]
    spam_indices = np.where(train_label == 1)[0]

    # get class probability
    class_probability = np.array([float(ham_indices.shape[0]) / float(train_label.shape[0]),
                                  float(spam_indices.shape[0]) / float(train_label.shape[0])],
                                 float)

    # store class probability for later use
    class_probability_file = open('class_probability.p', 'wb')
    pickle.dump(class_probability, class_probability_file)

    # get word probability for ham
    word_probability_ham = (1 + np.sum(train_feature[ham_indices, :], 0))/(np.sum(1 + np.sum(train_feature[ham_indices, :], 0)))

    # get word probability for spam
    word_probability_spam = (1 + np.sum(train_feature[spam_indices, :], 0))/(np.sum(1 + np.sum(train_feature[spam_indices, :], 0)))

    # take log to prevent underflow
    log_word_prob_ham = np.log10(word_probability_ham)
    log_word_prob_spam = np.log10(word_probability_spam)

    # dump the results for later use
    word_probability_ham_file = open('word_probability_ham.p', 'wb')
    word_probability_spam_file = open('word_probability_spam.p', 'wb')
    pickle.dump(log_word_prob_ham, word_probability_ham_file)
    pickle.dump(log_word_prob_spam, word_probability_spam_file)

    # get unobserved probability
    unobserved_word_prob = np.zeros(2, dtype=float)

    unobserved_word_prob[0] = math.log10(float(1)/np.sum(1 + np.sum(train_feature[ham_indices, :], 0)))
    unobserved_word_prob[1] = math.log10(float(1)/np.sum(1 + np.sum(train_feature[spam_indices, :], 0)))

    # dump unobserved word probability for later use
    unobserved_word_prob_file = open('unobserved_word_prob.p', 'wb')
    pickle.dump(unobserved_word_prob, unobserved_word_prob_file)

    unobserved_word_prob_file.close()
    word_probability_ham_file.close()
    word_probability_spam_file.close()
    train_feature_file.close()
    train_label_file.close()

def main():
    gen_feature()


if __name__ == '__main__':
    main()