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
from shared_embedding_autoencode import SharedEmbeddingAutoencoder

if __name__== "__main__":

    print("Loading data...")

    devset = pd.read_csv("../data/recsys/movieClipsRatingsTrain_AvgStd.csv", index_col=0)

    movie_names = devset.index
    groups = devset["movieId"]

    audio_features = pd.read_csv("../data/recsys/Audio/Block level features/All/BLF_all_fullId.csv", index_col=0, header=None)
    video_features = pd.read_csv("../data/recsys/Visual/AlexNetFc7Final/Avg/AlexNetFeatures - AVG - fc7.csv", index_col=0, header=None)

    audio_features = audio_features.loc[movie_names]
    video_features = video_features.loc[movie_names]

    metadata_genre = pd.read_csv("../data/recsys/Metadata/GenreFeatures.csv", index_col=0)
    
    metadata = metadata_genre.loc[devset.loc[movie_names]["movieId"]]
    metadata_features = pd.DataFrame(index=movie_names, data = metadata.values, columns=metadata.columns)    

    data = [metadata_features, audio_features, video_features]
    # data = pd.concat(data, axis=1)

    input_shapes = [(d.shape[1],) for d in data]
    #input_shape = (data.shape[1],)

    if len(argv) > 1:
        config_filename = argv[1]
        with open(config_filename) as f:
            config = json.load(f)
        latent_dim = config["encoder"][-1]["kwargs"]["units"]
        #config["decoder"][-1]["kwargs"]["units"] = None
    else:
        latent_dim = 256    
        config = {
                "encoder": [
                    {
                        "name": "hidden1",
                        "type": "Dense",
                        "kwargs": {
                            "units": 1024,
                            "activation": "relu"
                        }
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

    for g in [0, 1, 0.1, 0.001, 0.0001]:

        ae = SharedEmbeddingAutoencoder(config["encoder"],
                        None,
                        input_shapes=input_shapes,
                        latent_shape=latent_shape,
                        loss="mean_squared_error",
                        optimizer_params=None,
                        gamma=g)

        experiment = Experiment(project_name="Recsys SharedEmbeddingAutoencoder (different sound)", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
        experiment.log_parameter("Experiment name", "Testing different layers")
        experiment.log_parameter("Architecture file name", config_filename)
        experiment.log_multiple_params(config)
        experiment.log_parameter("Latent dim", latent_shape[0])
        experiment.log_parameter("Gammma", g)

        ae.summary()
        #scores = ae.cross_validate(data, groups, experiment=experiment, epochs=1000, n_splits=4, callbacks = [kc.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)])
        scores = ae.run_training(data, groups, random_state=100, epochs=1000, callbacks = [kc.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)])

        print(scores)
        print(np.mean(scores))

        experiment.log_metric("score", np.mean(scores))
