import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from MasterProject.data_Preprocessing.Datasets import get_data
from MasterProject.data_Preprocessing.model import get_model
from sklearn.cross_validation import train_test_split
from keras.layers import Dense,Input
from keras.layers import Embedding,GRU, Bidirectional, TimeDistributed
from keras.models import Model
from MasterProject.data_Preprocessing.HAN_Classifier.HAN_Helper import Helper
from keras.optimizers import rmsprop
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from sklearn.metrics import confusion_matrix
import itertools
import pickle
import matplotlib.pyplot as plt
from keras import regularizers
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk import tokenize


Model_MAN = get_model.model()
model_name = "300features_10min_10window_NEG15_extra.txt"
#define a heler
helper = Helper.Process()

#final parmeters
L2PARAM = 0.002
WORDS_NUM = 100
SEN_NUM = 15
MAX_NB_WORDS = 30000
DIM = 300
EPOCHS = 20
CONTEXT_DIM = 200
BATCH_SIZE = 64

L2_REGULAR = regularizers.l2(L2PARAM)

#-------------------------------------------------------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def Display(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def get_matrix(reviews, sentences,vocabulary):
    texts_matrix = np.zeros((len(reviews),SEN_NUM,WORDS_NUM),dtype='int32')
    print(texts_matrix.shape)

    for index1,review in enumerate(sentences):
        for index2, sentence in enumerate(review):
            if(index2<SEN_NUM):
                tokens = text_to_word_sequence(sentence)
                count = 0
                non_exist = 0
                for _, w in enumerate(tokens):
                    if(w not in vocabulary.keys()):
                        print("non_exist")
                        print(w)
                        non_exist += 1
                        continue
                    if(count<WORDS_NUM and vocabulary[w]<MAX_NB_WORDS):
                        texts_matrix[index1,index2, count] = vocabulary[w]
                        count = count + 1


    return texts_matrix

def clean_text_to_text(review):
    # remove the the',",\
    review = re.sub(r"\\", "", review)
    review = re.sub(r"\'", "", review)
    review = re.sub(r"\"", "", review)
    # return the lower case
    text = review.strip().lower()

    return text

def get_normalized_data(data):
    reviews = []
    review_sentences = []
    review_tokens = []

    # clean the test dataset
    for review in data["Column2"]:
        text = BeautifulSoup(review)
        cleaned_text = text.get_text().encode('ascii', 'ignore')
        cleaned_string = cleaned_text.decode('utf-8')
        cleaned_review = clean_text_to_text(cleaned_string)
        reviews.append(cleaned_review)
        sentence = tokenize.sent_tokenize(cleaned_review)
        # number of the review
        review_sentences.append(sentence)

        for s in sentence:
            if (len(s) > 0):
                tokens = text_to_word_sequence(s)
                # filter out non-alpha
                tokens = [token for token in tokens if token.isalpha()]
                # filter out those short letters
                tokens = [t for t in tokens if len(t) > 1]
                review_tokens.append(tokens)

    return reviews, review_sentences, review_tokens

class AttLayer(Layer):
    def __init__(self, regularizer=None, **kwargs):
        self.regularizer = regularizer
        self.supports_masking = True
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='W', shape=(input_shape[-1], CONTEXT_DIM), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        self.b = self.add_weight(name='b', shape=(CONTEXT_DIM,), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        self.u = self.add_weight(name='u', shape=(CONTEXT_DIM,), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.dot(K.tanh(K.dot(x, self.W) + self.b), self.u)
        ai = K.exp(eij)
        alphas = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
        if mask is not None:
            # use only the inputs specified by the mask
            alphas *= mask
        weighted_input = x * alphas.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {}
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return None


#----------------get the dataset--------------------------------
data_main = get_data.Datasets()

#----------------get the vocab------------------------------------
print("1: get the vocab!")
train_reviews, train_sentences,train_tokens = data_main.get_normalized_data("train")
unlabeled_reviews, unlabeled_sentences, unlabeled_tokens  = data_main.get_normalized_data("unlabeled")
test_reviews, test_sentences, test_tokens = data_main.get_normalized_data("test")

#------add extra dataset------------------------

extra_train = pd.read_csv("/Users/xiaoyiwen/Desktop/MasterProject/MasterProject/data_Preprocessing/Datasets/kaggle_data/training_set.csv")
extra_test = pd.read_csv("/Users/xiaoyiwen/Desktop/MasterProject/MasterProject/data_Preprocessing/Datasets/kaggle_data/test_set.csv")

reviews1, sentences1, tokens1 = get_normalized_data(extra_train)

reviews2, sentences2, tokens2 = get_normalized_data(extra_test)

#--------gather-allthedata---------------------

all_cleaned_reviews = train_reviews+ unlabeled_reviews + test_reviews + reviews1 + reviews2

tokenizer  = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_cleaned_reviews)

vocabulary = tokenizer.word_index

print("vocab")
print(len(vocabulary))


x_test = get_matrix(test_reviews, test_sentences,vocabulary)

f1 = open('x_test_extra','wb')
pickle.dump(x_test,f1)

