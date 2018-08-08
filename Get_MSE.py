import os
os.environ['KERAS_BACKEND'] = 'theano'
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras.optimizers import rmsprop
# load the model
from keras.engine.topology import Layer, InputSpec
from keras import backend as K
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn. metrics import roc_curve

import pickle

CONTEXT_DIM = 200

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


def load_HAN(json, h5):

    json_file = open(json, 'r')

    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json, custom_objects={'AttLayer': AttLayer()})

    # load_weight
    loaded_model.load_weights(h5)

    print("loaded model from disk")

    optimizer1 = rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['acc'])

    return loaded_model


def get_MSE(y_true,y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    return MSE

def get_Precision(y_pred1,y_true1):
    y_pred = np.argmax(y_pred1,axis=1)
    y_true = np.argmax(y_true1,axis=1)
    precision = precision_score(y_true=y_true, y_pred=y_pred)

    return precision

def get_Recall(y_true, y_pred):
    y_pred1 = np.argmax(y_pred, axis=1)
    y_true1 = np.argmax(y_true, axis=1)
    recall = recall_score(y_true= y_true1, y_pred=y_pred1)
    return recall


def get_F1_Score(y_true, y_pred):
    y_pred1 = np.argmax(y_pred, axis=1)
    y_true1 = np.argmax(y_true, axis=1)
    F1_score = f1_score(y_true=y_true1, y_pred=y_pred1)

    return F1_score


def ROC_AUC_chart(y_true, y_preds):


    fpr, tpr, _ = roc_curve(y_true[:,1], y_preds[:,1])
    plt.plot(fpr,tpr)

    plt.xlabel("Fasle Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


def get_all_metrics(y_true, y_pred):

    print(get_MSE(y_true=y_true,y_pred=y_true))
    print(get_Precision(y_pred1=y_pred,y_true1=y_true))
    print(get_Recall(y_true=y_true,y_pred=y_pred))
    print(get_F1_Score(y_true=y_true, y_pred=y_pred))

    ROC_AUC_chart(y_pred)



#--------main-----------------------------------

JSON_file="/Users/xiaoyiwen/Documents/HAN5_polarity(89.62%)/HAN5_polarity_data.json"
h5_file ="/Users/xiaoyiwen/Documents/HAN5_polarity(89.62%)/HAN5_polarity_data.h5"


model = load_HAN(json=JSON_file, h5=h5_file)

#
# f1 = open('x_test_extra','rb')
# x_test = pickle.load(f1)

f2 = open('y_test_labels','rb')
y_true = pickle.load(f2)


# f4 = open('Y_prediction_labels','rb')
# y_preds2 = pickle.load(f4)

# y_preds2 = model.predict(x_test,batch_size=125, verbose=1)
#
# f3  = open('y_preds2','wb')
# pickle.dump(y_preds2, f3)

f2 = open('/Users/xiaoyiwen/Documents/HAN5_polarity(89.62%)/HAN5_polarity_pred','rb')
y_preds2 = pickle.load(f2)

# #score, acc = model.evaluate(x_test,y_true,batch_size=125,verbose=1)
#
# print('Test score:', score)
# print('Test accuracy:', acc)

#
# print(get_MSE(y_true=y_true,y_pred=y_preds1))
# print(get_Precision(y_pred1=y_preds1,y_true1=y_true))
# print(get_Recall(y_true=y_true, y_pred=y_preds1))
# print(get_F1_Score(y_true=y_true, y_pred=y_preds1))
# ROC_AUC_chart(y_true=y_true,y_preds=y_preds1)




print("MSE")
print(get_MSE(y_true=y_true,y_pred=y_preds2))
print("Precision")
print(get_Precision(y_pred1=y_preds2,y_true1=y_true))
print("Recall")
print(get_Recall(y_true=y_true, y_pred=y_preds2))
print("F1_Score")
print(get_F1_Score(y_true=y_true, y_pred=y_preds2))

ROC_AUC_chart(y_true=y_true,y_preds=y_preds2)