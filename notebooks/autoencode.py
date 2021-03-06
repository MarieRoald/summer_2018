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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
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
import copy

class Autoencoder:
    """Class to create an autoencoder model with the Keras API. 

    The autoencoder architecture is defined with a list of dictionaries. 
    Each dictionary contains the parameters to create a Keras Layer Instance. 
    The parameter dictionaries have to include a `"name"` and a `"type"`. 
    The name is the name of the layer instance, and the type is the name of the 
    Keras Layer to instantiate.  

    Parameters
    -----------
    encoder_params: list of dictionaries
        List of dictionaries containing the layer architecture for the encoder. 
    input_shape: tuple
        Tuple describing the shape of the input (not including batch size).
    latent_shape: tuple
        Tuple describing the shape of the latent embedding (not including 
        batch size).
    decoder_params: dict
        Dictionary containing the architecture for the decoder. If not given, 
        the decoder architecture is generated from the encoder architecture.
    optimizer_params: dict (optional)
        Dictionary containing parameters to create optimizer. If not given,
        the default optimizer is "adam" 
    loss: String (name of objective function) or keras objective function
        Loss function that is passed to keras model compilation 
    

    Examples:
    ---------
    encoder_config = [
        {
            "name": "hidden1",
            "type": "Dense",
            "kwargs": {
                "units": 2500,
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
                "units": 2000,
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
    """
    def _suffix_config_layer_names(self, config, suffix):
        new_config = copy.deepcopy(config)

        for layer_config in new_config:
            layer_config["name"] = layer_config["name"] + suffix
        return new_config

    def __init__(self, encoder_params, decoder_params, input_shape,
                 latent_shape, optimizer_params=None, 
                 loss="mean_squared_error"):
        """Initiate Autoencoder instance"""

        self._input_params = {k: v for k, v in locals().items() if k != 'self'}

        self._build(encoder_params, decoder_params, input_shape,
                    latent_shape, optimizer_params=optimizer_params, loss=loss)

    def _create_model(self, config, input_shape):
        layers = self._create_layers(config)
        model_input = kl.Input(shape=input_shape)
        encoder = self._create_model_from_layers(input=model_input, layers=layers)
        return encoder, model_input, layers

    def _create_model_from_layers(self, input, layers):
        output = self._stack_layers(input=input, layers=layers)
        model = km.Model(inputs=input, outputs=output)
        return model

    def _stack_layers(self, input, layers):
        current_output = layers[0](input)
        for layer in layers[1:]:
            current_output = layer(current_output)
        return current_output
    
    def _initiate_params(self, encoder_params, decoder_params, input_shape, latent_shape):

        self._check_encoder_params(encoder_params, latent_shape)
        if decoder_params is None:
            decoder_params = self._create_decoder_parameters_from_encoder(encoder_params, output_dim=None)
            encoder_params = self._suffix_config_layer_names(encoder_params, "_encoder")
        self._check_decoder_params(decoder_params, input_shape)

        return encoder_params, decoder_params

    def _build(self, encoder_params, decoder_params, input_shape,
               latent_shape, optimizer_params=None, loss="mean_squared_error"):

        encoder_params, decoder_params = self._initiate_params(encoder_params, decoder_params, input_shape, latent_shape)
        encoder, decoder, ae = self._create_autoencoder(encoder_params,
                                                       decoder_params,
                                                       input_shape,
                                                       latent_shape)
        self.encoder = encoder
        self.decoder = decoder
        self.ae = ae

        self.optimizer = self._create_optimizer(optimizer_params)
        self.ae.compile(optimizer=self.optimizer, loss=loss)

    def _check_encoder_params(self, encoder_params, latent_shape):
        if "kwargs" in encoder_params[-1]:
            if "units" in encoder_params[-1]["kwargs"]:
                if encoder_params[-1]["kwargs"]["units"] is None:
                    encoder_params[-1]["kwargs"]["units"] = latent_shape[0]

                if encoder_params[-1]["kwargs"]["units"] != latent_shape[0]:
                    raise ValueError("Latent shape must be None or equal to the number of units"
                                     " in the last layer of the encoder.")
            else:
                encoder_params[-1]["kwargs"]["units"] = latent_shape[0]
        else:
            encoder_params[-1]["kwargs"] = {"units": latent_shape[0]}

        return encoder_params

    def _check_decoder_params(self, decoder_params, input_shape):
        if "kwargs" in decoder_params[-1]:
            if "units" in decoder_params[-1]["kwargs"]:
                if decoder_params[-1]["kwargs"]["units"] is None:
                    decoder_params[-1]["kwargs"]["units"] = input_shape[0]

                if decoder_params[-1]["kwargs"]["units"] != input_shape[0]:
                    raise ValueError("Decoder output dimension of final layer must be None"
                                     " or equal to the input dimension"
                                     " (it is automatically set to be identical"
                                     " to the input shape of the encoder)")
            else:
                decoder_params[-1]["kwargs"]["units"] = input_shape[0]
        else:
            decoder_params[-1]["kwargs"] = {"units": input_shape[0]}

    def reset(self):
        self._build(**self.get_params())

    def _create_optimizer(self, optimizer_params):
        """Creates a keras optimizer from dict containing the parameters"""

        if optimizer_params is None:
            optimizer_params = {
                "type": "adam",
            }
            
        optimizer_type = getattr(ko, optimizer_params["type"])
        optimizer = optimizer_type(**optimizer_params.get("kwargs",{}))
        return optimizer

    def _create_decoder_parameters_from_encoder(self, encoder_config, output_dim=None, suffix="_decoder"):

        decoder_config = copy.deepcopy(encoder_config)
        if suffix is not None:
            decoder_config = self._suffix_config_layer_names(decoder_config, suffix)

        decoder_config = [l for l in decoder_config[-2::-1] if l["type"] != "Dropout"]

        output_layer = {
            "name": "output",
            "type": "Dense",
            "kwargs": {
                "units": output_dim,
                "activation": "linear"
            }
        }

        decoder_config.append(output_layer)
        return decoder_config

    def fit(self, X, validation_data=None, *args, **kwargs):
        """Trains a keras autoencoder model"""

        if validation_data is not None:
            validation_data = self._create_validation_data(validation_data)

        self.ae.fit(X, X, validation_data=validation_data, *args, **kwargs)

    def _create_validation_data(self, val_data):
        return (val_data, val_data)

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
                           input_shape, latent_shape):
        """Creates an autoencoder model from dicts containing the parameters"""

        encoder, encoder_input, encoder_layers = self._create_model(encoder_config, input_shape)
        decoder, decoder_input, decoder_layers = self._create_model(decoder_config, latent_shape)

        autoencoder_model = self._create_model_from_layers(input=encoder_input, layers=encoder_layers + decoder_layers)

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
                                        **config_dict.get("kwargs", {})))
            else:
                layers.append(layertype(name=config_dict["name"],
                                        **config_dict.get("kwargs", {})))

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

        max_train_it = min(map(len, train_losses))
        train_losses = [train_loss[:max_train_it] for train_loss in train_losses]
        train_steps = train_steps[:max_train_it]

        max_val_it = min(map(len, val_losses))
        val_losses = [val_loss[:max_val_it] for val_loss in val_losses]
        val_steps = val_steps[:max_val_it]

        val_losses = np.array(val_losses)
        train_losses = np.array(train_losses)
        colors = sns.color_palette()

        plot_mean_std_loss(train_losses, train_steps, ax=ax, color=colors[1], legend="train")
        plot_mean_std_loss(val_losses, val_steps, ax=ax, color=colors[0], legend="val")

    def _generate_kfold_split(self, n_splits, data, groups):
        kfold = GroupKFold(n_splits=n_splits)
        return kfold.split(data, data, groups)

    def _prepare_data(self, data):
        return np.asarray(data)

    def _index_data(self, data, idx):
        return data[idx]
    
    def summary(self):
        return self.ae.summary()

    def cross_validate(self, data, groups, experiment, n_splits=10, 
                       standardize=True, epochs=100, callbacks=None, log_prefix=""):

        data = self._prepare_data(data)

        val_losses = []
        train_losses = []
        val_errors = []

        if callbacks is None:
            callbacks = []

        for i, (train_idx, val_idx) in enumerate(self._generate_kfold_split(n_splits, data, groups)):
            self.reset()
            train_data = self._index_data(data, train_idx)
            val_data = self._index_data(data, val_idx)

            comet_logger = GroupedCometLogger(experiment, f"{log_prefix}cv_fold_{i}")
            if standardize:
                train_data, val_data, _ = \
                    self._standardize_data(train_data, val_data)

            self.fit(train_data, epochs=epochs, validation_data=val_data,
                   callbacks=[comet_logger]+callbacks)

            val_losses.append(comet_logger.val_loss)
            train_losses.append(comet_logger.train_loss)
            val_errors.append(self._rmse(val_data))

        fig = self._crossval_plots(train_losses, comet_logger.train_steps, 
                                   val_losses, comet_logger.val_steps)
        experiment.log_figure("Cross validation loss", fig)

        return val_errors 

    def generate_group_train_test_split(self, data, groups, random_state=100):
        return next(GroupShuffleSplit(random_state=random_state).split(data, groups=groups))

    def run_training(self, data, groups, random_state=100, standardize=True, epochs=100, callbacks=None, log_prefix="" ):
        data = self._prepare_data(data)

        if callbacks is None:
            callbacks = []

        train_idx, val_idx = self.generate_group_train_test_split(data,groups,random_state=random_state)

        train_data = self._index_data(data, train_idx)
        val_data = self._index_data(data, val_idx)

        if standardize:
            train_data, val_data, _ = self._standardize_data(train_data, val_data)

        self.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)

        train_pred = self.encode(train_data)
        val_pred = self.encode(val_data)

        np.save("train_data_embedding.npy", train_pred)
        np.save("val_data_embedding.npy", val_pred)
        print(train_pred.shape)
        print(val_pred.shape)

        return self._rmse(val_data)

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
                        "units": 2500,
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
                        "units": 2000,
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


    ae = Autoencoder(config["encoder"],
                     None,
                     input_shape=input_shape,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    groups = data_reader.groups

    experiment = Experiment(project_name="Concatenated autoencoder", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing different layers")
    experiment.log_parameter("Architecture file name", config_filename)
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])

    ae.summary()

    scores = ae.cross_validate(data, groups, experiment=experiment, epochs=10000, n_splits=4, callbacks = [kc.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)])

    #ae.save("saved_model.h5")
    experiment.log_other("scores", scores)

    print(scores)
