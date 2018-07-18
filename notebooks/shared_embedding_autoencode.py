
from comet_ml import Experiment
from autoencode import Autoencoder
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

from sys import argv

import json


class SharedEmbeddingLayer(Layer):
    def __init__(self, gamma, *args, **kwargs):
        self.gamma = gamma
        self.is_placeholder = True
        super(SharedEmbeddingLayer, self).__init__(*args, **kwargs)

    def call(self, embeddings):
        return embeddings


class SharedEmbeddingAutoencoder(Autoencoder):

    def __init__(self, encoder_params, decoder_params, input_shapes,
                 latent_shape, optimizer_params=None, loss="mean_squared_error"):
        """ """

        self._input_params = {k: v for k, v in locals().items() if k != 'self'}

        self._build(encoder_params, decoder_params, input_shapes,
                    latent_shape, optimizer_params=optimizer_params, loss=loss)

    def _build(self, encoder_params, decoder_params, input_shapes,
               latent_shape, optimizer_params=None, loss="mean_squared_error"):
        
        if encoder_params[-1]["kwargs"]["units"] != latent_shape[0]:
            raise ValueError("Latent shape must be equal to the number of units"
                             " in the last layer of the encoder.")

        encoders, decoders, aes, ae = self._create_autoencoder(encoder_params,
                                                       decoder_params,
                                                       input_shapes,
                                                       latent_shape)
        self.encoders = encoders
        self.decoders = decoders
        self.aes = aes
        self.ae = ae

        self.optimizer = self._create_optimizer(optimizer_params)
        self.ae.compile(optimizer=self.optimizer, loss=loss)

    def _rmse(self, val_data):
        predictions = self.predict(val_data)
        val_rmse = []
        for p, vd in zip(predictions, val_data):
            val_rmse.append(np.sqrt(np.mean((p-vd)**2)))
        return val_rmse

    def _create_autoencoder(self, encoder_config, decoder_config, input_shapes, latent_shape):
        """Creates an autoencoder model from dicts containing the parameters"""
        # TODO: skal jeg kunne ta inn en liste med input og output navn?
        encoder_layers = []
        encoder_inputs = []
        
        decoder_layers = []
        decoder_inputs = [] # unødvendig?

        encoders = [] #shares architecture
        decoders = [] #shares architecture
        autoencoders = [] #shares architecture 
        

        #TODO: kanskje litt merkelig at self.decoder_layers ikke har input lag,
        #      men Autoencoder sin self.decoder_layers har det

        #TODO: hva gjør jeg med decoderene?

        #TODO: self decoder_layers of encoder layers har bare lagene til en model.

        for i, input_shape in enumerate(input_shapes):
            
            current_encoder_config = copy.deepcopy(encoder_config)
            
            for layer_config in current_encoder_config:
                layer_config["name"] = layer_config["name"] + f"_encoder_{i}" 


            current_encoder, current_encoder_input, current_encoder_layers = self._create_model(current_encoder_config, input_shape)

            encoders.append(current_encoder)
            encoder_inputs.append(current_encoder_input)
            

        embeddings = [encoder.output for encoder in encoders]
        sharedembeddinglayer = SharedEmbeddingLayer(gamma=1)
        print(embeddings)
        embeddings = sharedembeddinglayer(embeddings)
        #from ipdb import set_trace; set_trace()
        #TODO: skal vi ha en liste med separate autoencoders?

        outputs = []
        for i, embedding in enumerate(embeddings):

            current_decoder_config = copy.deepcopy(decoder_config)

            for layer_config in current_decoder_config:
                layer_config["name"] = layer_config["name"] + f"_decoder_{i}" 

            # TODO: Dette må fikses
            current_decoder_config[-1]["kwargs"]["units"] = input_shapes[i][0]

            current_decoder, current_decoder_input, current_decoder_layers = self._create_model(current_decoder_config, latent_shape)

            decoders.append(current_decoder)
            #current_autoencoder_layers = current_encoder_layers+self.decoder_layers

            print(current_decoder_layers)

            current_layer = current_decoder_layers[0](embedding)
            for layer in current_decoder_layers[1:]:
                current_layer = layer(current_layer)

            outputs.append(current_layer)
            print(i)
            print(encoder_inputs)
            print(embeddings)

        combined_autoencoder = km.Model(input=encoder_inputs, outputs=outputs)

        return encoders, decoders, autoencoders, combined_autoencoder



    def fit(self, X, *args, **kwargs):
        """Trains a keras autoencoder model"""

        self.ae.fit(x=X, y=X, *args, **kwargs)

    def cross_validate(self, data, groups, experiment, n_splits=10, 
                       standardize=True, epochs=100):

        data = [np.asarray(d) for d in data]
        kfold = GroupKFold(n_splits=n_splits)

        val_losses = []
        train_losses = []

        print(data[0])
        val_errors = []

        for i, (train_idx, val_idx) in enumerate(kfold.split(data[0], data[0], groups)):
            self.reset()
            print(train_idx)
            train_data = [d[train_idx] for d in data]
            val_data = [d[val_idx] for d in data]
            comet_logger = GroupedCometLogger(experiment, f"cv_fold_{i}")

            train_data_scaled = []
            val_data_scaled = []
            for td, vd in zip(train_data,val_data):
                if standardize:
                    td_scaled, vd_scaled, _ = self._standardize_data(td, vd)
                else:
                    td_scaled, vd_scaled = td, vd
                train_data_scaled.append(td_scaled)
                val_data_scaled.append(vd_scaled)

            train_data, val_data = train_data_scaled, val_data_scaled
            self.fit(train_data, epochs=epochs, validation_data=(val_data, val_data), callbacks=[comet_logger, kc.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)])

            val_losses.append(comet_logger.val_loss)
            train_losses.append(comet_logger.train_loss)
            val_errors.append(self._rmse(val_data))

        fig = self._crossval_plots(train_losses, comet_logger.train_steps, 
                                   val_losses, comet_logger.val_steps)
        experiment.log_figure("Cross validation loss", fig)

        return val_errors

    def _create_model(self, config, input_shape):
        layers = self._create_layers(config)
        model_input = kl.Input(shape=input_shape)
        encoder = self._create_model_from_layers(input=model_input, layers=layers)
        return encoder, model_input, layers

    def _stack_layers(self, layers):
        current_input_layer = layers[0]
        for layer in layers[1:]:
            current_input_layer = layer(current_input_layer)

    def _create_model_from_layers(self, input, layers):
        current_input_layer = layers[0](input)
        for layer in layers[1:]:
            current_input_layer = layer(current_input_layer)

        model = km.Model(inputs=input, outputs=current_input_layer)
        return model

if __name__== "__main__":
    filenames = ["X1_train.csv", "X2_train.csv", "X3_train.csv"]
    data_reader = DataReader(data_set_filenames=filenames, groups_filename="ID_train.csv")
    data = data_reader.get_seperate_data()
    data_combined = data_reader.get_all_data()

    config_filename = argv[1]
    with open(config_filename) as f:
        config = json.load(f)

    latent_dim = config["encoder"][-1]["kwargs"]["units"]
    latent_shape = (latent_dim,)


    input_shapes = [(d.shape[1],) for d in data]


    ae = SharedEmbeddingAutoencoder(config["encoder"],
                     config["decoder"],
                     input_shapes=input_shapes,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    groups = data_reader.get_groups()

    experiment = Experiment(project_name="Shared Embedding Autoencoder", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "shared embedding test")
    experiment.log_parameter("Architecture file name", config_filename)
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])
    scores = ae.cross_validate(data, groups, experiment=experiment, epochs=1000, n_splits=4)

    print(scores)