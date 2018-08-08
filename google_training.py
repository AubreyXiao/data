import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
from bs4 import BeautifulSoup
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from nltk import tokenize
from MasterProject.data_Preprocessing.Datasets import get_data
from MasterProject.data_Preprocessing.model import get_model
from sklearn.cross_validation import train_test_split
from keras.layers import Dense,Input,Flatten
from keras.layers import Conv1D,MaxPooling1D,Embedding,Merge,Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from MasterProject.data_Preprocessing.HAN_Classifier.HAN_Helper import Helper
from keras.models import Sequential
from keras.optimizers import rmsprop
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  GridSearchCV
from gensim.models import Word2Vec
import matplotlib
import pickle
import nltk
import re
nltk.download('punkt')

#final parmeters
L2PARAM = 0.01
WORDS_NUM = 100
SEN_NUM = 15
MAX_NB_WORDS = 15000
DIM = 300
EPOCHS = 10
CONTEXT_DIM = 100
BATCH_SIZE = 50

L2_REGULAR = regularizers.l2(L2PARAM)

#--------------display-------------------------------
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


# ---------------get_dataset---------------------------

def clean_text_to_text(review):
    # remove the the',",\
    review = re.sub(r"\\", "", review)
    review = re.sub(r"\'", "", review)
    review = re.sub(r"\"", "", review)
    # return the lower case
    text = review.strip().lower()

    return text


def load_word2vec_embedding(modelname,vocabulary):
    #word2vec.txt
    file = open(modelname, 'r')
    lines = file.readlines()[1:]
    file.close()

    #create a word2vec vocab
    word2vec_dict = dict()
    for line in lines:
        splits = line.split()
        word = splits[0]
        value = splits[1:]

        word2vec_dict[word] = np.asarray(value, dtype='float32')

    #create a word embedding layer
    embedding_matrix = np.random.random((len(vocabulary) + 1, DIM))
    count = 0
    for word, index in vocabulary.items():
        vector = word2vec_dict.get(word)
        if (vector is not None):
            count = count + 1
            embedding_matrix[index] = vector

    print(count)

    return embedding_matrix

#get the data input (matrix form)
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

def get_normalized_data(data):
    reviews = []
    review_sentences = []
    review_tokens = []

    # clean the test dataset
    for review in data["review"]:
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

#-------------attention-layer--------------------------

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

train_data =
test_data =
unlabeled_data =

#----------------get the vocab------------------------------------
train_reviews, train_sentences,train_tokens = get_normalized_data(train_data)
unlabeled_reviews, unlabeled_sentences, unlabeled_tokens  = get_normalized_data(unlabeled_data)
test_reviews, test_sentences, test_tokens = get_normalized_data(test_data)

all_cleaned_reviews = train_reviews+ unlabeled_reviews+test_reviews

tokenizer  = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_cleaned_reviews)

vocabulary = tokenizer.word_index
print("all the vocabulary(train + test + unlabeled data)")
print(vocabulary)
print(len(vocabulary))

#-----------------get the train matrix of the input--------------------------

print("get the x_train matrix")
train_matrix = get_matrix(train_reviews,train_sentences,vocabulary)
#(2,5000,15,100)
print(train_matrix.shape)

#--------------get_train_labels---------------------------------------
labels = []
for sentiment in train_data["sentiment"]:
    labels.append(sentiment)
train_labels = to_categorical(np.asarray(labels))

#----------------------test-matrix----------------------
print("get the test matrix")
x_test = get_matrix(test_reviews,test_sentences,vocabulary)
print(x_test.shape)

#---------get---test---labels-------------
test_labels = get_test_labels(test_data)
y_test = to_categorical(np.asarray(test_labels))


#---------------split the dataset---------------------------------

#8:2 = train: validation
x_train, x_val, y_train, y_val = train_test_split(train_matrix,train_labels,test_size=0.2)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


#--------create----embedding---layer------
# 1: create a embedding layer

modelname = ""
embedding_matrix = load_word2vec_embedding(modelname,vocabulary)


wordvec_embedding = Embedding(len(vocabulary) + 1, DIM, weights=[embedding_matrix], mask_zero=True,
                              embeddings_regularizer=L2_REGULAR, input_length=WORDS_NUM, trainable=True)

# 2: build up the hierarchical neural network
# create the sentence input
input_sen = Input(shape=(WORDS_NUM,), dtype='int32')
sentence_sequence = wordvec_embedding(input_sen)
l_lstm = Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=L2_REGULAR))(sentence_sequence)
# l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer(regularizer=L2_REGULAR)(l_lstm)
sen_encoder = Model(input_sen, l_att)

# 3: create review input
input_review = Input(shape=(SEN_NUM, WORDS_NUM), dtype='int32')
review_encoder = TimeDistributed(sen_encoder)(input_review)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=L2_REGULAR))(review_encoder)
# l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)

l_att_sent = AttLayer(regularizer=L2_REGULAR)(l_lstm_sent)

preds = Dense(2, activation='softmax', kernel_regularizer=L2_REGULAR)(l_att_sent)
model = Model(input_review, preds)

# 4: compile the model
#adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#rmsprop = rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])


#fit the model
print("fitting the_HAN model")
print(model.summary())
history = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=2)

Display(history)

#---train--------
loss, accuracy  = model.evaluate(train_matrix,train_labels,batch_size=125, verbose=2)
print('Train loss:', loss)
print('Train accuracy:', accuracy)

#----test--------
score, acc = model.evaluate(x_test,y_test,batch_size=125,verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)

