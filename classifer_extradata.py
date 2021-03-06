import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from MasterProject.data_Preprocessing.Datasets import get_data
from MasterProject.data_Preprocessing.model import get_model
from sklearn.cross_validation import train_test_split
from keras.layers import Dense, Input
from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import TensorBoard
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
model_name = "300features_50min_5window_NEG10_polarity_extra.txt"
# define a helerbn
helper = Helper.Process()

# final parmeters
L2PARAM = 0.002
WORDS_NUM =100
SEN_NUM = 15
MAX_NB_WORDS = 17841
DIM = 300
EPOCHS = 20
CONTEXT_DIM = 100
BATCH_SIZE = 64
GRU_UNITS = 100

L2_REGULAR = regularizers.l2(L2PARAM)
optimizer1 = rmsprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#optimizer1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

tensorboard = TensorBoard(log_dir='./logs',histogram_freq=0, write_graph= True, write_images=True)

# ----------------get the dataset--------------------------------
data_main = get_data.Datasets()
polarity_data  = data_main.get_polarity_data()

# -------------------------------------------------------------------

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


def get_matrix(reviews, sentences, vocabulary):
    texts_matrix = np.zeros((len(reviews), SEN_NUM, WORDS_NUM), dtype='int32')
    print(texts_matrix.shape)

    for index1, review in enumerate(sentences):
        for index2, sentence in enumerate(review):
            if (index2 < SEN_NUM):
                tokens = text_to_word_sequence(sentence)
                count = 0
                non_exist = 0
                for _, w in enumerate(tokens):
                    if (w not in vocabulary.keys()):
                        print("non_exist")
                        print(w)
                        non_exist += 1
                        continue
                    if (count < WORDS_NUM and vocabulary[w] < MAX_NB_WORDS):
                        texts_matrix[index1, index2, count] = vocabulary[w]
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


# def get_normalized_data(data):
#     reviews = []
#     review_sentences = []
#     review_tokens = []
#
#     # clean the test dataset
#     for review in data["Column2"]:
#         text = BeautifulSoup(review)
#         cleaned_text = text.get_text().encode('ascii', 'ignore')
#         cleaned_string = cleaned_text.decode('utf-8')
#         cleaned_review = clean_text_to_text(cleaned_string)
#         reviews.append(cleaned_review)
#         sentence = tokenize.sent_tokenize(cleaned_review)
#         # number of the review
#         review_sentences.append(sentence)
#
#         for s in sentence:
#             if (len(s) > 0):
#                 tokens = text_to_word_sequence(s)
#                 # filter out non-alpha
#                 tokens = [token for token in tokens if token.isalpha()]
#                 # filter out those short letters
#                 tokens = [t for t in tokens if len(t) > 1]
#                 review_tokens.append(tokens)
#
#     return reviews, review_sentences, review_tokens


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




# ----------------get the vocab------------------------------------
print("1: get the vocab!")
train_reviews, train_sentences, train_tokens = data_main.get_normalized_data("train")
unlabeled_reviews, unlabeled_sentences, unlabeled_tokens = data_main.get_normalized_data("unlabeled")
test_reviews, test_sentences, test_tokens = data_main.get_normalized_data("test")
po_reviews, po_sentences , po_tokens = data_main.get_normalized_data("polarity")



# -------create vocab--------------------

all_cleaned_reviews = train_reviews + unlabeled_reviews + test_reviews + po_reviews

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_cleaned_reviews)

vocabulary = tokenizer.word_index

print("vocab")
print(len(vocabulary))

# -----------------get the train matrix of the input--------------------------
print("2: get the x_train matrix")
train_review = train_reviews + po_reviews
train_sen = train_sentences + po_sentences

train_matrix = get_matrix(train_review, train_sen, vocabulary)
# get the matrix form of the

print("train matrix(extra)")
print(train_matrix.shape)

x_test = get_matrix(test_reviews, test_sentences, vocabulary)

# --------------get the labels---------------------------------------
print("4: get the labels(train + test)")
test_labels = helper.get_labels("test")
y_test = to_categorical(np.asarray(test_labels))
# --------------------------------------------------------------------
train_label1 = helper.get_labels("train")

label2 = []
for sentiment in polarity_data["sentiment"]:
    label2.append(sentiment)

labels = train_label1 + label2

train_labels = to_categorical(np.asarray(labels))

# ---------------split the dataset---------------------------------
print("5:split the data")
x_train, x_val, y_train, y_val = train_test_split(train_matrix, train_labels,test_size=0.1)

print("the data shape")
print(train_matrix.shape)

print("train set")
print(x_train.shape)
print("val_det")
print(x_val.shape)
# initial create a embedding matrix to save the weights of the word vector

# ----------------word--embedding---------------------------------

print("6: create the word embedding ")
word2vec_dict = Model_MAN.load_embedding(model_name)
embedding_matrix = np.random.random((len(vocabulary) + 1, DIM))
count = 0
for word, index in vocabulary.items():
    vector = word2vec_dict.get(word)
    if (vector is not None):
        count = count + 1
        embedding_matrix[index] = vector

print(count)

# create a embedding layer
wordvec_embedding = Embedding(len(vocabulary) + 1, DIM, weights=[embedding_matrix], mask_zero=True,
                              embeddings_regularizer=L2_REGULAR, input_length=WORDS_NUM, trainable=True)

# --------------------------------------------------
print("7: HAN ")
# build up the hierarchical neural network
# create the sentence input
input_sen = Input(shape=(WORDS_NUM,), dtype='int32')
sentence_sequence = wordvec_embedding(input_sen)
l_lstm = Bidirectional(GRU(GRU_UNITS, return_sequences=True, kernel_regularizer=L2_REGULAR))(sentence_sequence)
# l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer(regularizer=L2_REGULAR)(l_lstm)
sen_encoder = Model(input_sen, l_att)

# create review input
input_review = Input(shape=(SEN_NUM, WORDS_NUM), dtype='int32')
review_encoder = TimeDistributed(sen_encoder)(input_review)
l_lstm_sent = Bidirectional(GRU(GRU_UNITS, return_sequences=True, kernel_regularizer=L2_REGULAR))(review_encoder)
# l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)

l_att_sent = AttLayer(regularizer=L2_REGULAR)(l_lstm_sent)

preds = Dense(2, activation='softmax', kernel_regularizer=L2_REGULAR)(l_att_sent)
model = Model(input_review, preds)

# compile the model
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# optimizer2 = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
# optimizer3 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['acc'])

# -----fit-------------------------

print("8: fitting the_HAN model")
print(model.summary())
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,callbacks=[tensorboard])

# ------save----model--------------
# serialize model to JSON
print("9: save model")

model_json = model.to_json()
with open("HAN11_polarity_data.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("HAN11_polarity_data.h5")
print("Saved model to disk")

Display(history)

# ------test----accuracy--------

loss, accuracy = model.evaluate(train_matrix, train_labels, batch_size=125, verbose=2)
print('Train loss:', loss)
print('Train accuracy:', accuracy)

score, acc = model.evaluate(x_test, y_test, batch_size=125, verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)

# ----------predict------------------------------
HAN_pred = model.predict(x_test, batch_size=125, verbose=2)

f1 = open('HAN11_polarity_pred','wb')
pickle.dump(HAN_pred,f1)


f2 = open('y_test_labels', 'rb')
y_test_1 = pickle.load(f2)

y_pred = np.argmax(HAN_pred, axis=1)
y_true = np.argmax(y_test_1, axis=1)

class_names = ['positive', 'negative']

cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

plot_confusion_matrix(cm, classes=class_names, normalize=False,
                      title='Non_Normalized confusion matrix')
plt.show()





