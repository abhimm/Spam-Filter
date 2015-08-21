__author__ = 'Abhinav'
import pickle


# generate vocabulary
def gen_vocab(train_data_file_name):

    train_data_file = open(train_data_file_name, 'rb')
    train_data = train_data_file.readlines()
    # vocabulary dictionary with index
    vocabulary = dict()
    index = 0
    # get vocabulary from test data and training data
    for data in train_data:
        temp = data.strip().split()
        i = 2
        while i < len(temp):
            if not vocabulary.has_key(temp[i]):
                vocabulary[temp[i]] = index
                index += 1
            i += 2
    vocabulary_file = open('vocabulary.p', 'wb')
    pickle.dump(vocabulary, vocabulary_file)
    vocabulary_file.close()
    train_data_file.close()

def main():

    gen_vocab('train')

if __name__ == '__main__':
    main()



