from comet_ml import Experiment
import keras.layers as kl
import keras.models as km
import keras.regularizers as kr

from keras import backend as K

import pandas as pd
from sklearn.decomposition import PCA
from pprint import pprint
import matplotlib.pyplot as plt

def create_layers(layer_dict_list, input_shape=None):
    """Creates a list of keras layers from a list of dictionaries containing parameters"""
    layers = []

    if input_shape is not None:
        layers.append(kl.InputLayer(input_shape=input_shape))

    for config_dict in layer_dict_list:
        layertype = getattr(kl,config_dict["type"])
        if "regularizer" in config_dict:
            regularizer = getattr(kr,config_dict["regularizer"]["type"])(config_dict["regularizer"]["value"])
            layers.append(layertype(name=config_dict["name"], activity_regularizer=regularizer, **config_dict["kwargs"]))
        else:
            layers.append(layertype(name=config_dict["name"], **config_dict["kwargs"]))

    return layers
        
def create_autoencoder(encoder_config, decoder_config, input_shape=None, latent_shape=None):
    """Creates an autoencoder model from dictionaries containing the parameters"""
    encoder_layers = create_layers(encoder_config, input_shape=input_shape)
    decoder_layers = create_layers(decoder_config, input_shape=latent_shape)

    #Jeg kan jo også legge til input layers utenfor create layers funksjonen?

    encoder = km.Sequential(encoder_layers)
    decoder = km.Sequential(decoder_layers)
    autoencoder_model = km.Sequential(encoder_layers + decoder_layers)

    return encoder, decoder, autoencoder_model

if __name__== "__main__":
    data1 = pd.read_csv("X1.csv", index_col=0)
    data2 = pd.read_csv("X2.csv", index_col=0)
    data3 = pd.read_csv("X3.csv", index_col=0)
    
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

    encoder, decoder, ae = create_autoencoder(config["encoder"], 
                                    config["decoder"], 
                                    input_shape=input_shape, 
                                    latent_shape=latent_shape)

    ae.compile(optimizer="adam", loss="mean_squared_error")
    experiment = Experiment(project_name="Autoencoder demo", api_key="50kNmWUHJrWHz3FlgtpITIsB1")

    ae.fit(data, data, shuffle=True, epochs=1000, batch_size=32, validation_split=0.3, verbose=1)
    encoded = encoder.predict(data)

    encoded_pca = PCA().fit_transform(encoded)



    print(ae.evaluate(data, data)/data.shape[1])

    plt.scatter(encoded_pca[:,0], encoded_pca[:,1])
    plt.show()

