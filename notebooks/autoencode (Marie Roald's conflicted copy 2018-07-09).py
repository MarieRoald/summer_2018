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

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.base import clone

import numpy as np
import keras

import seaborn as sns


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

    def fit(self, X,*args, shuffle=True, epochs=200, batch_size=32, verbose=1,
            **kwargs):
        """ """
        self.ae.fit(X, X, shuffle=shuffle, epochs=epochs, batch_size=batch_size,
                    verbose=verbose, **kwargs)

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
		
def plot_loss(mean_loss, std_loss, steps):

	colors = sns.color_palette()
	fig = plt.figure()
	ax = fig.add_subplot(111) 

	ax.fill_between(steps, mean_loss + std_loss, np.maximum(mean_loss - std_loss, 0),
					facecolor=colors[1], alpha=0.2,
					color = colors[1], label='std')
					
	ax.plot(steps, mean_loss, color = colors[1], linestyle="--")
	
	return fig
		
def my_cross_validate(clf, data, target, groups, experiment, n_splits=10):

    data = data.values
    target = target.values

    group_kfold = GroupKFold(n_splits=n_splits)

    i = 0
	
    val_losses = []
    val_errors = []
	
    for train_index, test_index in group_kfold.split(data, target, groups):
        
        comet_logger = CrossEntropyCometLogger(experiment, f"cv_fold_{i}")
        new_clf = clone(clf)
        history = new_clf.fit(data[train_index], autoencoder__validation_data=(data[test_index], 
							  data[test_index]), autoencoder__callbacks=[comet_logger])
							  
        val_losses.append(comet_logger.val_loss) 
							  
        i += 1
        predictions = new_clf.predict(data[test_index])
        
        val_rmse = np.sqrt(((predictions - target[test_index]) ** 2).mean())

        val_errors.append(val_rmse)
		
    val_losses = np.array(val_losses)
    mean_loss = val_losses.mean(0)
    std_loss = val_losses.std(0)
    fig = plot_loss(mean_loss, std_loss, np.array(comet_logger.val_steps))
    experiment.log_figure("fig_test", fig)
    fig.show()

    return val_errors

class CrossEntropyCometLogger(keras.callbacks.Callback):

    def __init__(self, experiment, log_name):
        self.experiment = experiment
        self.step_count = 1
        self.epoch_count = 1
        self.log_name = log_name
        self.train_loss = []
        self.train_steps = []
        self.val_loss = []
        self.val_steps = []
        
    def on_batch_end(self, batch, logs=None):
        logs = {} if logs is None else logs
        self.experiment.log_metric(f"{self.log_name}_loss", logs.get('loss'), step=self.step_count)
        self.train_steps.append(self.step_count)
        self.train_loss.append(logs.get("loss"))
        self.step_count = self.step_count + 1 

    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        if "val_loss" in logs:
            self.experiment.log_metric(f"{self.log_name}_val_loss", logs.get('loss'), step=self.step_count)
            self.val_steps.append(self.step_count)
            self.val_loss.append(logs.get("loss"))



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

    group_kfold = GroupKFold(n_splits=5)
    groups = pd.read_csv("ID_train.csv", index_col=0,
                          names=["Sample ID", "Person ID"])

    scaler = StandardScaler()
    clf = Pipeline(steps=[('standardscaler', scaler), ('autoencoder', ae)])
    
    experiment = Experiment(project_name="comet test", api_key="50kNmWUHJrWHz3FlgtpITIsB1")

    comet_logger = CrossEntropyCometLogger(experiment, "TEST")
    
    clf.fit(data, autoencoder__callbacks=[comet_logger])
    """
    scores = cross_validate(clf, data, data, groups=groups,
                            scoring="neg_mean_squared_error",
                            cv=group_kfold, return_train_score=True)

    """
    scores = my_cross_validate(clf, data, data, groups, experiment=experiment)

    print([score/data.shape[1] for score in scores])

