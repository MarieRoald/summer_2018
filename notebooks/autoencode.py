from comet_ml import Experiment
import keras.layers as kl
import keras.models as km
import keras.regularizers as kr
import keras.optimizers as ko

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

class Autoencoder:
    def __init__(self, encoder_params, decoder_params, input_shape,
                 latent_shape, optimizer_params=None, loss="mean_squared_error"):
        """ """

        self._input_params = {k: v for k, v in locals().items() if k != 'self'}

        self._build(encoder_params, decoder_params, input_shape,
                    latent_shape, optimizer_params=optimizer_params, loss=loss)

    def _build(self, encoder_params, decoder_params, input_shape,
               latent_shape, optimizer_params=None, loss="mean_squared_error"):
        
        if encoder_params[-1]["kwargs"]["units"] != latent_shape[0]:
            raise ValueError("Latent shape must be equal to the number of units"
                             " in the last layer of the encoder.")

        encoder, decoder, ae = self._create_autoencoder(encoder_params,
                                                       decoder_params,
                                                       input_shape,
                                                       latent_shape)
        self.encoder = encoder
        self.decoder = decoder
        self.ae = ae

        self.optimizer = self._create_optimizer(optimizer_params)
        self.ae.compile(optimizer=self.optimizer, loss=loss)

    def reset(self):
        self._build(**self.get_params())

    def _create_optimizer(self, optimizer_params):
        """Creates a keras optimizer from dict containing the parameters"""

        if optimizer_params is None:
            optimizer_params = {
                "type": "SGD",
                "kwargs": {
                    "momentum": 0.9
                }
            }

        optimizer_type = getattr(ko, optimizer_params["type"])
        optimizer = optimizer_type(**optimizer_params["kwargs"])
        return optimizer

    def fit(self, X, *args, **kwargs):
        """Trains a keras autoencoder model"""

        self.ae.fit(X, X, **kwargs)

    def get_params(self, deep=False):
        return self._input_params

    def predict(self, X):
        return self.ae.predict(X)

    def encode(self, input):
        return self.encoder.predict(input)

    def decode(self, embedding):
        return self.decoder.predict(embedding)

    def save(self, filename):
        return self.ae.save(filename)

    def _create_autoencoder(self, encoder_config, decoder_config,
                           input_shape=None, latent_shape=None):
        """Creates an autoencoder model from dicts containing the parameters"""

        self.encoder_layers = self._create_layers(encoder_config,
                                            input_shape=input_shape)
        self.decoder_layers = self._create_layers(decoder_config,
                                            input_shape=latent_shape)

        encoder = km.Sequential(self.encoder_layers)
        decoder = km.Sequential(self.decoder_layers)
        autoencoder_model = km.Sequential(self.encoder_layers + self.decoder_layers[1:] )

        return encoder, decoder, autoencoder_model

    def _create_layers(self, layer_dict_list, input_shape=None):
        """Creates a list of keras layers from a list of parameter dicts"""
        layers = []

        if input_shape is not None:
            layers.append(kl.InputLayer(input_shape=input_shape))

        for config_dict in layer_dict_list:
            layertype = getattr(kl,config_dict["type"])

            if "regularizer" in config_dict:
                reg_params = config_dict["regularizer"]
                regularizer_type = getattr(kr, reg_params["type"])
                regularizer = regularizer_type(reg_params["value"])
                layers.append(layertype(name=config_dict["name"],
                                        activity_regularizer=regularizer,
                                        **config_dict["kwargs"]))
            else:
                layers.append(layertype(name=config_dict["name"],
                                        **config_dict["kwargs"]))

        return layers

    def _standardize_data(self, train_data, val_data):

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        return train_data, val_data, scaler

    def _rmse(self, val_data):
        predictions = self.predict(val_data)
        val_rmse = np.sqrt(((predictions - val_data)**2).mean())
        return val_rmse
    
    def _crossval_plots(self, train_losses, train_steps, val_losses, val_steps):
        fig = plt.figure()
        ax = fig.add_subplot(111) 

        val_losses = np.array(val_losses)
        train_losses = np.array(train_losses)
        colors = sns.color_palette()
        
        plot_mean_std_loss(train_losses, train_steps, ax=ax, color=colors[1], legend="train")
        plot_mean_std_loss(val_losses, val_steps, ax=ax, color=colors[0], legend="val")

    def cross_validate(self, data, groups, experiment,  n_splits=10, 
                       standardize=True, epochs=100):

        data = np.asarray(data)
        kfold = GroupKFold(n_splits=n_splits)

        val_losses = []
        train_losses = []
        val_errors = []

        for i, (train_idx, val_idx) in enumerate(kfold.split(data, data, groups)):
            self.reset()
            train_data, val_data = data[train_idx], data[val_idx]
            comet_logger = GroupedCometLogger(experiment, f"cv_fold_{i}")

            if standardize:
                train_data, val_data, _ = self._standardize_data(train_data, val_data)

            ae.fit(train_data, epochs=epochs, validation_data=(val_data, val_data),
                   callbacks=[comet_logger])

            val_losses.append(comet_logger.val_loss)
            train_losses.append(comet_logger.train_loss)
            val_errors.append(self._rmse(val_data))

        fig = self._crossval_plots(train_losses, comet_logger.train_steps, 
                                   val_losses, comet_logger.val_steps)
        experiment.log_figure("Cross validation loss", fig)

        return val_errors

if __name__== "__main__":
    filenames = ["X1_train.csv", "X2_train.csv", "X3_train.csv"]
    data_reader = DataReader(data_set_filenames=filenames, groups_filename="ID_train.csv")
    data = data_reader.get_all_data()

    input_shape = (data.shape[1],)
    latent_shape = (100,)

    config = {
        "encoder": [
            {
                "name": "hidden1_encoder",
                "type": "Dense",
                "kwargs": {
                    "units": 200,
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
                    "activation": "relu"
                }
            }
        ],
        "decoder": [
            {
                "name": "hidden1_decoder",
                "type": "Dense",
                "kwargs": {
                    "units": 200,
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

    ae = Autoencoder(config["encoder"],
                     config["decoder"],
                     input_shape=input_shape,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    group_kfold = GroupKFold(n_splits=5)
    groups = data_reader.get_groups()

    experiment = Experiment(project_name="comet test", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Cross validate test")
    scores = ae.cross_validate(data, groups, experiment=experiment)

    print(scores)
