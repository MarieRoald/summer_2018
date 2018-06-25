from comet_ml import Experiment
import keras.layers as kl
import keras.models as km
import keras.regularizers as kr

from keras import backend as K

import pandas as pd
from sklearn.decomposition import PCA
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_validate

from sklearn.pipeline import make_pipeline

from sklearn.base import clone


class Autoencoder:
    def __init__(self, encoder_params, decoder_params, input_shape,
                 latent_shape, optimizer_params, loss="mean_squared_error"):
        """ """
        if encoder_params[-1]["kwargs"]["units"] != latent_shape[0]:
            raise ValueError("Latent shape must be equal to the number of units"
                             " in the last layer of the encoder.")

        self._input_params = {k: v for k, v in locals().items() if k != 'self'}

        encoder, decoder, ae = self.create_autoencoder(encoder_params,
                                                       decoder_params,
                                                       input_shape,
                                                       latent_shape)
        self.encoder = encoder
        self.decoder = decoder
        self.ae = ae

        # TODO: fix that shit
        self.ae.compile(optimizer="adam", loss=loss)

    def fit(self, X,*args, shuffle=True, epochs=500, batch_size=32, verbose=1,
            **kwargs):
        """ """
        self.ae.fit(X, X, shuffle=shuffle, epochs=epochs, batch_size=batch_size,
                    verbose=verbose)

    def get_params(self, deep=False):
        return self._input_params

    def predict(self, X):
        return self.ae.predict(X)

    def encode(self, input):
        return self.encoder.predict(input)

    def decode(self, embedding):
        return self.decoder.predict(embedding)

    def create_autoencoder(self, encoder_config, decoder_config,
                           input_shape=None, latent_shape=None):
        """Creates an autoencoder model from dicts containing the parameters"""
        encoder_layers = self.create_layers(encoder_config,
                                            input_shape=input_shape)
        decoder_layers = self.create_layers(decoder_config,
                                            input_shape=latent_shape)

        encoder = km.Sequential(encoder_layers)
        decoder = km.Sequential(decoder_layers)
        autoencoder_model = km.Sequential(encoder_layers + decoder_layers)

        return encoder, decoder, autoencoder_model

    def create_layers(self, layer_dict_list, input_shape=None):
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

def my_cross_validate(clf, data, target, groups, n_splits=5):

    group_kfold = GroupKFold(n_splits=n_splits)

    val_losses = []

    for train_idxs, val_idxs in group_kfold.split():
        new_clf = clone(clf)
        new_clf.fit(data[train_idxs])
        predictions = new_clf.predict(data[val_idxs])
        
        # Should we assume that clf has evaluate?

        val_loss = clf.evaluate(data[val_idx], target[val_idxs])
        val_losses.append(val_loss)


    


if __name__== "__main__":
    data1 = pd.read_csv("X1_train.csv", index_col=0)
    data2 = pd.read_csv("X2_train.csv", index_col=0)
    data3 = pd.read_csv("X3_train.csv", index_col=0)

    data = pd.concat([data1, data2, data3], axis=1)

    input_shape = (data.shape[1],)
    latent_shape = (16,)

    config = {
        "encoder": [
            {
                "name": "hidden1_encoder",
                "type": "Dense",
                "kwargs": {
                    "units": 32,
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
                    "units": 32,
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


    ae.fit(X=data)
    group_kfold = GroupKFold(n_splits=5)
    groups = pd.read_csv("ID_train.csv", index_col=0,
                          names=["Sample ID", "Person ID"])

    scaler = StandardScaler()
    clf = make_pipeline(scaler, ae)


    experiment = Experiment(project_name="Autoencoder demo", api_key="50kNmWUHJrWHz3FlgtpITIsB1")

    """
    scores = cross_validate(clf, data, data, groups=groups,
                            scoring="neg_mean_squared_error",
                            cv=group_kfold, return_train_score=True)
    """

    scores = my_cross_validate(clf, data, data, groups)

    print(scores)
