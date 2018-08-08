from MasterProject.data_Preprocessing.Datasets import get_data
import nltk
from nltk import tokenize
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import numpy as np
from  MasterProject.data_Preprocessing.HAN_Classifier import Attention_Layer
import os
os.environ['KERAS_BACKEND']='theano'


# define the parameter
WORDS_NUM = 100
SEN_NUM = 15
MAX_NB_WORDS = 20000



#word2vec get the tokens for trainig
def get_normalized_data(dataset_name):
    reviews = []
    review_sentences = []
    review_tokens = []


    data = get_data.Datasets()
    if(dataset_name =="train"):
        train_data = data.get_train_data()
        data = train_data
    if(dataset_name =="test"):
        test_data = data.get_test_data()
        data = test_data
    if(dataset_name =="unlabeled"):
        unlabeled_data = data.get_unlabeled_data()
        data = unlabeled_data


    # clean the test dataset
    for review in data["review"]:
        cleaned_review  = data.clean_text_to_text(review)
        reviews.append(cleaned_review)
        sentence = tokenize.sent_tokenize(cleaned_review)
        # number of the review
        review_sentences.append(sentence)

        for s in sentence:
           if(len(s)>0):
               tokens = text_to_word_sequence(s)
               tokens = [token for token in tokens if token.isalpha()]
               review_tokens.append(tokens)


    return reviews, review_sentences, review_tokens


#get the test data labels (HAN classifier)
#return matrix form
def get_test_labels(test_data):

    test_labels = []

    #define the label
    for id in test_data["id"]:
        # print(id)
        id = id.strip('"')
        # print(id)
        id, score = id.split('_')
        score = int(score)
        if (score < 5):
            test_labels.append(0)
        if (score >= 7):
            test_labels.append(1)



    return test_labels





def get_nomalized_test_data(train_vocab):
    #get the train data
    data = get_data.Datasets()
    test_data = data.get_test_data()
    test_reviews = []
    test_sentences = []
    test_labels = []


    #test data preprocessing ----------

    #clean the test dataset
    for test in test_data["review"]:
        cleaned_test = data.clean_text_to_text(test)
        test_reviews.append(cleaned_test)
        sentences = tokenize.sent_tokenize(cleaned_test)
        test_sentences.append(sentences)

    #define the label
    for id in test_data["id"]:
        # print(id)
        id = id.strip('"')
        # print(id)
        id, score = id.split('_')
        score = int(score)
        if (score < 5):
            test_labels.append(0)
        if (score >= 7):
            test_labels.append(1)




    #test----
    #create a tokenizer and limit only dealt with top 20000 words
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(test_reviews)

    print("train_vocab")
    print(train_vocab)
    print(len(train_vocab))

    #define the test_matrix
    test_matrix = np.zeros((len(test_reviews),SEN_NUM,WORDS_NUM),dtype='int32') #(250000,15,100)

    #print(test_matrix.shape)
    non_exist = 0
    for review_index, review in enumerate(test_sentences):

        for sentence_index, sentence in enumerate(review):

            if(sentence_index<SEN_NUM):
                #print(sentence)
                tokens = text_to_word_sequence(sentence)

                num = 0

                for _, token in enumerate(tokens):
                   #see if the token is in the vocab

                   if(token not in train_vocab.keys()):
                        print(token)
                        non_exist += 1
                        continue
                   if(num< WORDS_NUM and train_vocab[token]<MAX_NB_WORDS):
                        test_matrix[review_index,sentence_index,num] = train_vocab[token]
                        num += 1



    print(non_exist)
    #test_labels-> tocategory

    predicted_labels = to_categorical(np.asarray(test_labels))




    return test_matrix, predicted_labels




