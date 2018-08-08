import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from MasterProject.data_Preprocessing.Datasets import get_data
from MasterProject.data_Preprocessing.model import get_model
from sklearn.cross_validation import train_test_split
from keras.layers import Dense,Input,Flatten
from keras.layers import Conv1D,MaxPooling1D,Embedding,Merge,Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from MasterProject.data_Preprocessing.HAN_Classifier.HAN_Helper import Helper
from nltk.collections import Counter
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  GridSearchCV

import pickle
#define a heler
helper = Helper.Process()

#final parmeters
WORDS_NUM = 100
SEN_NUM = 15
MAX_NB_WORDS = 20000
DIM = 300


def save_list(keys, filename):
    data = '\n'.join(keys)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def get_matrix(reviews, sentences,vocabulary,common_keys):
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
                    if(count<WORDS_NUM and w in common_keys):
                        texts_matrix[index1,index2, count] = vocabulary[w]
                        count = count + 1


    return texts_matrix


class AttLayer(Layer):
    #initial the attention layer
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[-1],),
                                 initializer='normal',
                                 trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def create_HAN_classifier(WORDS_NUM,SEN_NUM ,AttLayer):
    Model_MAN = get_model.model()
    model = "300DIM_10min_10window.txt"
    word2vec_dict = Model_MAN.load_embedding(model)
    embedding_matrix = np.random.random((len(vocabulary) + 1, DIM))
    count = 0
    for word, index in vocabulary.items():
        vector = word2vec_dict.get(word)
        if (vector is not None):
            count = count + 1
            embedding_matrix[index] = vector

    print("word embedding matrix")
    print(count)

    # create a embedding layer
    wordvec_embedding = Embedding(len(vocabulary) + 1, DIM, weights=[embedding_matrix], input_length=WORDS_NUM,
                                  trainable=True)

    # sentence_level:
    input_sen = Input(shape=(WORDS_NUM,), dtype='int32')
    sentence_sequence = wordvec_embedding(input_sen)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(sentence_sequence)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    l_att = AttLayer()(l_dense)
    sen_encoder = Model(input_sen, l_att)

    # review_level:
    input_review = Input(shape=(SEN_NUM, WORDS_NUM), dtype='int32')
    review_encoder = TimeDistributed(sen_encoder)(input_review)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    l_att_sent = AttLayer()(l_dense_sent)
    preds = Dense(2, activation='softmax')(l_att_sent)
    model = Model(input_review, preds)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model


def get_most_common_vocab(MAX_NB_WORDS,vocab):
    items1 = [key[0] for key in vocab.most_common(MAX_NB_WORDS)]
    return items1

def load_vocab(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()

    return text

def create_vocab_dict(vocab):
    dict = {}
    for index, item in enumerate(vocab):
         dict[item] = index+1


    return dict

#----------------get the dataset--------------------------------
data_main = get_data.Datasets()

#----------------get the vocab------------------------------------
train_reviews, train_sentences,train_tokens = data_main.get_normalized_data("train")
unlabeled_reviews, unlabeled_sentences, unlabeled_tokens  = data_main.get_normalized_data("unlabeled")
test_reviews, test_sentences, test_tokens = data_main.get_normalized_data("test")

all_cleaned_reviews = train_reviews+ unlabeled_reviews+test_reviews

#create a counter
all_tokens = train_tokens + test_tokens +unlabeled_tokens
counter = Counter()

for tokens in all_tokens:
    counter.update(tokens)
print(len(counter))


common_keys = get_most_common_vocab(MAX_NB_WORDS,counter)
print("common keys")
print(common_keys)



#load the vocab
vocab_name = "tuning_vocab_txt"
text = load_vocab(vocab_name)
slices = set(text.split())
vocabulary = create_vocab_dict(slices)
print(vocabulary)
print(len(vocabulary))
print(len(slices))
#
#
# print(all_cleaned_reviews)
# print(len(all_cleaned_reviews))
#
# tokenizer  = Tokenizer(num_words=20000)
# tokenizer.fit_on_texts(all_cleaned_reviews)
#
# word_index = tokenizer.word_index
# word_counts = tokenizer.word_counts
# keys = list(word_index.keys())
# vocabulary = dict()
#
# for index, word in enumerate(keys):
#     vocabulary[word] = index +1
#
#
# print("all the vocabulary(train + test)")
# print(vocabulary)
# print(len(vocabulary))

#-----------------get the train matrix of the input--------------------------

#get the matrix form of the
print("get the x_train matrix")
train_matrix = get_matrix(train_reviews,train_sentences,vocabulary,common_keys)
#(2,5000,15,100)
print(train_matrix.shape)

#--------------get the labels---------------------------------------

train_labels = helper.get_labels("train")

y_train = to_categorical(np.asarray(train_labels))


#---------------split the dataset---------------------------------

x_train, x_val, y_train, y_val = train_test_split(train_matrix, train_labels, test_size = 0.2)
print("validation set")
print("x_val")
print(x_val.shape)
print("y_val")
print(y_val.shape)

#initial create a embedding matrix to save the weights of the word vector
#use validation for grid search

classifier = KerasClassifier(build_fn=create_HAN_classifier, WORDS_NUM = WORDS_NUM, SEN_NUM= SEN_NUM,AttLayer = AttLayer)

#create a dict for parameters
batch_size = [10,20,30,40,50,60,70,80,100,125]
epochs = [10,20,30]
param_grid = dict(batch_size = batch_size, epochs = epochs)
grid = GridSearchCV(estimator=classifier, param_grid= param_grid,n_jobs=-1)
grid_result = grid.fit(x_val,y_val)

#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with : %r" % (mean, stdev, param))
























