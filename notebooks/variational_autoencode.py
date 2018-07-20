from autoencode import Autoencoder
from keras.engine.topology import Layer

from comet_ml import Experiment
import keras.layers as kl
import keras.models as km
import keras.regularizers as kr
import keras.optimizers as ko
import keras.callbacks as kc

from keras import backend as K
import keras

import pandas as pd
from sklearn.decomposition import PCA
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_validate

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.base import clone

import numpy as np
import seaborn as sns

from log_utils import plot_mean_std_loss, GroupedCometLogger
from data_reader import DataReader
from sys import argv


import json
from pprint import pprint

from autoencoder_layers import KLDivergenceLayer



def nll(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

class VariationalAutoencoder(Autoencoder):

    def __init__(self, encoder_params, decoder_params, input_shape,
                 latent_shape, optimizer_params=None):
        """ """
        self._input_params = {k: v for k, v in locals().items() if k != 'self'}

        self._build(encoder_params, decoder_params, input_shape,
                    latent_shape, optimizer_params=optimizer_params, loss='mean_absolute_error')

    def _create_autoencoder(self, encoder_config, decoder_config,
                           input_shape=None, latent_shape=None):
        """Creates an autoencoder model from dicts containing the parameters"""

        self.encoder_layers = self._create_layers(encoder_config)
        self.decoder_layers = self._create_layers(decoder_config,
                                            input_shape=latent_shape)

        
        # x må være input
        # h må lages som siste encoder lag eller noe sånt

        x = kl.Input(shape=input_shape)

        h = self._stack_layers(input=x,layers=self.encoder_layers)
        
        encoder = km.Model(inputs=x, outputs=h) 


        z_mu = kl.Dense(latent_shape[0])(h)
        z_log_var = kl.Dense(latent_shape[0])(h)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = kl.Lambda(lambda t: K.exp(.5*t))(z_log_var)


        eps = kl.Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_shape[0])))
        z_eps = kl.Multiply()([z_sigma, eps])
        z = kl.Add()([z_mu, z_eps])

        #from ipdb import set_trace; set_trace()
        decoder = km.Sequential(self.decoder_layers)

        vae = km.Model(inputs=[x, eps], outputs= decoder(z))
        return encoder, decoder, vae



    def _stack_layers(self, input, layers):
        current_input_layer = layers[0](input)
        for layer in layers[1:]:
            current_input_layer = layer(current_input_layer)

        return current_input_layer

if __name__== "__main__":
    filenames = ["X1_train.csv", "X2_train.csv", "X3_train.csv"]

    config_filename = argv[1]
    with open(config_filename) as f:
        config = json.load(f)

    pprint(config)

    data_reader = DataReader(data_set_filenames=filenames, groups_filename="ID_train.csv")
    data = data_reader.get_all_data()

    input_shape = (data.shape[1],)
    # latent_shape = (100,)
    latent_dim = config["encoder"][-1]["kwargs"]["units"]
    latent_shape = (latent_dim,)

    """
    config = {
        "encoder": [
            {
                "name": "hidden1_encoder",
                "type": "Dense",
                "kwargs": {
                    "units": 2500,
                    "activation": "relu"
                },
                "regularizer": {
                    "type": "l1",
                    "value": 1e-3
                }
            },
            {
                "name": "hidden2_encoder",
                "type": "Dense",
                "kwargs": {
                    "units": 2000,
                    "activation": "relu"
                },
                "regularizer": {
                    "type": "l1",
                    "value": 1e-3
                }
            },
            {
                "name": "latent",
                "type": "Dense",
                "kwargs": {
                    "units": latent_shape[0],
                    "activation": "linear"
                },

                "regularizer": {
                    "type": "l1",
                    "value": 1e-3
                }
            }
        ],
        "decoder": [
            {
                "name": "hidden2_decoder",
                "type": "Dense",
                "kwargs": {
                    "units": 2000,
                    "activation": "relu"
                }
            },
            {
                "name": "hidden1_decoder",
                "type": "Dense",
                "kwargs": {
                    "units": 2500,
                    "activation": "relu"
                }
            },
            {
                "name": "output",
                "type": "Dense",
                "kwargs": {
                    "units": data.shape[1],
                    "activation": "linear"
                }
            }
        ]
    }
    """

    ae = VariationalAutoencoder(config["encoder"],
                     config["decoder"],
                     input_shape=input_shape,
                     latent_shape=latent_shape,
                     optimizer_params=None)

    groups = data_reader.get_groups()

    experiment = Experiment(project_name="variational autoencoder", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing different architectures")
    experiment.log_parameter("Architecture file name", config_filename)
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])
    scores = ae.cross_validate(data, groups, experiment=experiment, epochs=10000, n_splits=4)

    #ae.save("saved_model.h5")

    experiment.log_other("scores", scores)

    print(scores)
