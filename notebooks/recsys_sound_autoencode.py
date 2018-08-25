import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from comet_ml import Experiment
from autoencode import Autoencoder
import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
from log_utils import plot_mean_std_loss, GroupedCometLogger
from data_reader import DataReader
import copy
import numpy as np
from sklearn.model_selection import GroupKFold

from sys import argv


import json
from pprint import pprint

from multimodal_autoencode import MultimodalAutoencoder

if __name__== "__main__":

    print("Loading data...")

    devset = pd.read_csv("../data/recsys/movieClipsRatingsTrain_AvgStd.csv", index_col=0)

    movie_names = devset.index
    groups = devset["movieId"]

    audio_features = pd.read_csv("../data/recsys/Audio/i-vector/movieClips_ivec_feats_trte_sitem_gmms_512_tvDim_400_fullId.csv", index_col=0, header=None)
    audio_features_new = pd.read_csv("../data/recsys/Audio/Block level features/All/BLF_all_fullId.csv", index_col=0, header=None)

    audio_features = audio_features.loc[movie_names]
    audio_features_new = audio_features_new.loc[movie_names]

    data_ = [audio_features, audio_features_new]
    data = pd.concat(data_, axis=1)

    input_shape = (data.shape[1],)
    if len(argv) > 1:
        config_filename = argv[1]
        with open(config_filename) as f:
            config = json.load(f)
        latent_dim = config["encoder"][-1]["kwargs"]["units"]
        #config["decoder"][-1]["kwargs"]["units"] = None
    else:
        latent_dim = 128
        config =  {
            "encoder": [
                {
                    "kwargs": {
                        "activation": "relu",
                        "units": 1024
                    },
                    "name": "hidden1",
                    "type": "Dense"
                },
                {
                    "name": "batchnorm1",
                    "type": "BatchNormalization"
                },
                {
                    "kwargs": {
                        "rate": 0
                    },
                    "name": "dropout",
                    "type": "Dropout"
                },
                {
                    "kwargs": {
                        "activation": "relu",
                        "units": 128
                    },
                    "name": "latent",
                    "regularizer": {
                        "type": "l1",
                        "value": 0.001
                    },
                    "type": "Dense"
                }
            ]
        }
        
        config_filename = None

    latent_shape = (latent_dim,)

    ae = Autoencoder(config["encoder"],
                    None,
                    input_shape=input_shape,
                    latent_shape=latent_shape,
                    loss="mean_squared_error",
                    optimizer_params=None)

    experiment = Experiment(project_name="Recsys Sound Autoencoder", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing different layers")
    experiment.log_parameter("Architecture file name", config_filename)
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])

    ae.summary()
    scores = ae.run_training(data, groups, random_state=100, epochs=1000, standardize=True, callbacks = [kc.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)])

    print(scores)
    experiment.log_metric("score", np.mean(scores))
    print(np.mean(scores))
