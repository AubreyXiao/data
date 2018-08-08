import os
os.environ['KERAS_BACKEND']='theano'
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

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
