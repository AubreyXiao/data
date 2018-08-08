import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
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

#final parmeters
L2PARAM = 0.01
WORDS_NUM = 100
SEN_NUM = 15
MAX_NB_WORDS = 25000
DIM = 300
epochs = 20
CONTEXT_DIM = 100

L2_REGULAR = regularizers.l2(L2PARAM)



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

def create_HAN_classifier(WORDS_NUM,SEN_NUM ,AttLayer,lr=0.001,vocabulary=vocabulary,embedding_matrix = embedding_matrix):

    #1: create a embedding layer
    wordvec_embedding = Embedding(len(vocabulary) + 1, DIM, weights=[embedding_matrix], mask_zero=True,
                                  embeddings_regularizer=L2_REGULAR, input_length=WORDS_NUM, trainable=True)

    #2: build up the hierarchical neural network
    # create the sentence input
    input_sen = Input(shape=(WORDS_NUM,), dtype='int32')
    sentence_sequence = wordvec_embedding(input_sen)
    l_lstm = Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=L2_REGULAR))(sentence_sequence)
    # l_dense = TimeDistributed(Dense(200))(l_lstm)
    l_att = AttLayer(regularizer=L2_REGULAR)(l_lstm)
    sen_encoder = Model(input_sen, l_att)

    #3: create review input
    input_review = Input(shape=(SEN_NUM, WORDS_NUM), dtype='int32')
    review_encoder = TimeDistributed(sen_encoder)(input_review)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=L2_REGULAR))(review_encoder)
    # l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)

    l_att_sent = AttLayer(regularizer=L2_REGULAR)(l_lstm_sent)

    preds = Dense(2, activation='softmax', kernel_regularizer=L2_REGULAR)(l_att_sent)
    model = Model(input_review, preds)

    #4: compile the model
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    return model



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
train_data = data_main.get_train_data()
test_data = data_main.get_test_data()


#----------------get the vocab------------------------------------
train_reviews, train_sentences,train_tokens = data_main.get_normalized_data("train")
unlabeled_reviews, unlabeled_sentences, unlabeled_tokens  = data_main.get_normalized_data("unlabeled")
test_reviews, test_sentences, test_tokens = data_main.get_normalized_data("test")

all_cleaned_reviews = train_reviews+ unlabeled_reviews+test_reviews


print(all_cleaned_reviews)
print(len(all_cleaned_reviews))

tokenizer  = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(all_cleaned_reviews)

vocabulary = tokenizer.word_index

dictionary = tokenizer.word_counts

print("all the vocabulary(train + test)")
print(vocabulary)
print(len(vocabulary))

#-----------------get the train matrix of the input--------------------------

#get the matrix form of the
print("get the x_train matrix")
train_matrix = get_matrix(train_reviews,train_sentences,vocabulary)
#(2,5000,15,100)
print(train_matrix.shape)

#--------------get_train_labels---------------------------------------
labels = []
for sentiment in train_data["sentiment"]:
    labels.append(sentiment)
train_labels = to_categorical(np.asarray(labels))


#---------------split the dataset---------------------------------

#8:2 = train: validation
x_train, x_val, y_train, y_val = train_test_split(train_matrix,train_labels,test_size=0.1)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)




#----------grid----search---------------
#--------use_val_for_tuning_hyper-------
modelname = ""
load_word2vec_embedding(modelname,vocabulary)
classifier = KerasClassifier(build_fn=create_HAN_classifier, WORDS_NUM = WORDS_NUM, SEN_NUM= SEN_NUM,AttLayer = AttLayer,)

#create a dict for parameters
batch_size = [32,64,100]
epochs = [10]
param_grid = dict(batch_size = batch_size, epochs = epochs)

scores = ['precision', 'recall']
for score in scores:
    grid = GridSearchCV(estimator=classifier, param_grid= param_grid,n_jobs=-1,cv=3,scoring='%s_macro' % score)
    grid_result = grid.fit(x_val,y_val)
    print("Best parameters set found on development set:")
    print(grid_result.best_params_)
    print("Grid scores on development set:")

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with : %r" % (mean, stdev*2, param))



#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))















