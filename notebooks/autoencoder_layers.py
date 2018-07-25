from keras import backend as K
from keras.engine.topology import Layer

class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = 0.5*K.sum(1 + log_var - 
                             K.square(mu) - 
                             K.exp(log_var),
                             axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs

class SharedEmbeddingLayer(Layer):
    def __init__(self, gamma, *args, **kwargs):
        self.gamma = gamma
        self.is_placeholder = True
        super(SharedEmbeddingLayer, self).__init__(*args, **kwargs)

    def call(self, embeddings):
        reg = 0
        for ej in embeddings:
            for ei in embeddings:
                #reg += K.tf.losses.mean_squared_error(ej, ei)
                reg += K.mean(K.abs(ej-ei))
        self.add_loss(self.gamma*reg)
        return embeddings
