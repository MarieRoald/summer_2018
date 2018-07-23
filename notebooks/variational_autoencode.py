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
                 latent_shape, optimizer_params=None, loss=None):
        """ """
        self._input_params = {k: v for k, v in locals().items() if k != 'self'}

        if loss is None:
            loss = nll

        self._build(encoder_params, decoder_params, input_shape,
                    latent_shape, optimizer_params=optimizer_params, loss=loss)

    def _sample_z(self, z_mu, z_sigma, shape):
        eps = kl.Input(tensor=K.random_normal(shape=shape))

        z_eps = kl.Multiply()([z_sigma, eps])
        z = kl.Add()([z_mu, z_eps])
        return z, eps

    def _create_variational_parameters(self, h, latent_dim):
        z_mu = kl.Dense(latent_dim)(h)
        z_log_var = kl.Dense(latent_dim)(h)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = kl.Lambda(lambda t: K.exp(.5*t))(z_log_var)
        return z_mu, z_sigma
    
    def _create_autoencoder(self, encoder_config, decoder_config,
                           input_shape, latent_shape):
        """Creates a variational autoencoder model from dicts containing the parameters"""
        encoder, encoder_input, encoder_layers = self._create_model(encoder_config, input_shape)
        decoder, decoder_input, decoder_layers = self._create_model(decoder_config, latent_shape)

        z_mu, z_sigma = self._create_variational_parameters(encoder.output, latent_shape[0])
        z, eps = self._sample_z(z_mu, z_sigma, shape=(K.shape(encoder_input)[0], latent_shape[0]))

        vae = km.Model(inputs=[encoder_input, eps], name="variational autoencoder", outputs=decoder(z))
        return encoder, decoder, vae

if __name__== "__main__":
    filenames = ["X1_train.csv", "X2_train.csv", "X3_train.csv"]
    
    data_reader = DataReader(data_set_filenames=filenames, groups_filename="ID_train.csv")
    data = data_reader.combined_data

    if len(argv) > 1:
        config_filename = argv[1]
        with open(config_filename) as f:
            config = json.load(f)
            latent_dim = config["encoder"][-1]["kwargs"]["units"]
    else:
        latent_dim = 100    
        config = {
            "encoder": [
                {
                    "name": "hidden1",
                    "type": "Dense",
                    "kwargs": {
                        "units": 250,
                        "activation": "relu"
                    },
                    "regularizer": {
                        "type": "l1",
                        "value": 1e-3
                    }
                },            {
                    "name": "batchnorm1",
                    "type": "BatchNormalization"
                },
                {
                    "name": "hidden2",
                    "type": "Dense",
                    "kwargs": {
                        "units": 200,
                        "activation": "relu"
                    },
                    "regularizer": {
                        "type": "l1",
                        "value": 1e-3
                    }
                },            {
                    "name": "batchnorm2",
                    "type": "BatchNormalization"
                },
                {
                    "name": "latent",
                    "type": "Dense",

                    "regularizer": {
                        "type": "l1",
                        "value": 1e-3
                    }
                }
            ]
        }
        config_filename = None

    input_shape = (data.shape[1],)
    latent_shape = (latent_dim,)


    ae = VariationalAutoencoder(config["encoder"],
                     None,
                     input_shape=input_shape,
                     latent_shape=latent_shape,
                     optimizer_params=None)

    groups = data_reader.groups

    experiment = Experiment(project_name="variational autoencoder", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing different architectures")
    experiment.log_parameter("Architecture file name", config_filename)
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])

    ae.summary()

    scores = ae.cross_validate(data, groups, experiment=experiment, epochs=10000, n_splits=4, callbacks = [kc.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)])

    #ae.save("saved_model.h5")

    experiment.log_other("scores", scores)

    print(scores)
