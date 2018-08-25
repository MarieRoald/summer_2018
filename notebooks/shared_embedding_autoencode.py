
from comet_ml import Experiment
from autoencode import Autoencoder
from multimodal_autoencode import MultimodalBase
import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
from keras import backend as K
from keras.engine.topology import Layer
from log_utils import plot_mean_std_loss, GroupedCometLogger
from data_reader import DataReader
import copy
import numpy as np
from sklearn.model_selection import GroupKFold
from pprint import pprint

from sys import argv

import json
from autoencoder_layers import SharedEmbeddingLayer


class SharedEmbeddingAutoencoder(MultimodalBase):

    def _check_decoder_params(self, decoder_params, input_shapes):
        if "kwargs" in decoder_params[-1]:
            if "units" in decoder_params[-1]["kwargs"] \
            and decoder_params[-1]["kwargs"]["units"] is not None:
                raise ValueError("Decoder output of final layer must be None"
                                 " (it is automatically set to be identical"
                                 " to the input shape of the encoder)")
        else:
            decoder_params[-1]["kwargs"] = {}    
    
    def __init__(self, encoder_params, decoder_params, input_shapes,
                 latent_shape, gamma=0.1, optimizer_params=None, loss="mean_squared_error"):
        """ """
        self.gamma = gamma
        self._input_params = {k: v for k, v in locals().items() if k != 'self'}

        self._build(encoder_params, decoder_params, input_shapes,
                    latent_shape, optimizer_params=optimizer_params, loss=loss)

    def _build(self, encoder_params, decoder_params, input_shapes,
               latent_shape, optimizer_params=None, loss="mean_squared_error"):
        
        self._check_encoder_params(encoder_params, latent_shape)
        if decoder_params is None:
            decoder_params = self._create_decoder_parameters_from_encoder(encoder_params, output_dim=None)
        else:
            self._check_decoder_params(decoder_params, input_shapes)

        encoder_params = self._create_n_encoder_dicts(encoder_params, n=len(input_shapes))
        decoder_params = self._create_n_decoder_dicts(decoder_params, output_shapes=input_shapes)
        
        encoders, decoders, ae = self._create_autoencoder(encoder_params,
                                                       decoder_params,
                                                       input_shapes,
                                                       latent_shape)
        self.encoders = encoders
        self.decoders = decoders
        self.ae = ae

        self.optimizer = self._create_optimizer(optimizer_params)
        self.ae.compile(optimizer=self.optimizer, loss=loss)

    def _rmse(self, val_data):
        predictions = self.predict(val_data)
        val_rmse = []
        for p, vd in zip(predictions, val_data):
            val_rmse.append(np.sqrt(np.mean((p-vd)**2)))
        return val_rmse

    def _create_autoencoder(self, encoder_params, decoder_config, input_shapes, latent_shape):
        """Creates an autoencoder model from dicts containing the parameters"""

        encoders, encoder_inputs = self._create_encoders(encoder_params, input_shapes)

        embeddings = [encoder.output for encoder in encoders]
        embeddings = SharedEmbeddingLayer(gamma=self.gamma)(embeddings)

        decoders, decoder_outputs = self._create_decoders(decoder_config, latent_shape, embeddings)
        combined_autoencoder = km.Model(inputs=encoder_inputs, outputs=decoder_outputs)

        return encoders, decoders, combined_autoencoder

if __name__== "__main__":
    filenames = ["X1_train.csv", "X2_train.csv", "X3_train.csv"]
    data_reader = DataReader(data_set_filenames=filenames, groups_filename="ID_train.csv")
    data = data_reader.seperate_data
    data_combined = data_reader.combined_data

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

    latent_shape = (latent_dim,)

    input_shapes = [(d.shape[1],) for d in data]

    ae = SharedEmbeddingAutoencoder(config["encoder"],
                     None,
                     input_shapes=input_shapes,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    groups = data_reader.groups

    experiment = Experiment(project_name="Shared Embedding Autoencoder", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "shared embedding test")
    experiment.log_parameter("Architecture file name", config_filename)
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])
    ae.summary()
    scores = ae.cross_validate(data, groups, experiment=experiment, epochs=1000, n_splits=4, callbacks = [kc.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)] )

    print(scores)