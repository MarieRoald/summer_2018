from comet_ml import Experiment
from autoencode import Autoencoder
import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
from log_utils import plot_mean_std_loss, GroupedCometLogger
from data_reader import DataReader
import copy
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from sys import argv


import json
from pprint import pprint
class MultimodalBase(Autoencoder):
    def __init__(self, encoder_params, decoder_params, input_shapes,
                 latent_shape, optimizer_params=None, loss="mean_squared_error"):
        """ """
        self._input_params = {k: v for k, v in locals().items() if k != 'self'}

        self._build(encoder_params, decoder_params, input_shapes,
                    latent_shape, optimizer_params=optimizer_params, loss=loss)

    def _create_n_encoder_dicts(self, encoder_params, n):
        encoder_params_list = []
        for i in range(n):
            config = self._suffix_config_layer_names(encoder_params, f"_encoder_{i}")
            encoder_params_list.append(config)
        return encoder_params_list

    def _create_n_decoder_dicts(self, decoder_params, output_shapes):
        decoder_params_list = []
        for i, output_shape in enumerate(output_shapes):
            config = self._suffix_config_layer_names(decoder_params, f"_decoder_{i}")
            config[-1]["kwargs"]["units"] = output_shape[0]
            decoder_params_list.append(config)
        return decoder_params_list

    def _create_encoders(self, encoder_params, input_shapes):
        encoders = []
        encoder_inputs = []

        for params, shape in zip(encoder_params, input_shapes):
            encoder, input, layers = self._create_model(params, shape)
            encoders.append(encoder)
            encoder_inputs.append(input)
        return encoders, encoder_inputs

    def _create_decoders(self, decoder_params, latent_shape, embeddings):
        decoders = []
        decoder_outputs = []
        for params, embedding in zip(decoder_params, embeddings):
            decoder, input, layers = self._create_model(params, latent_shape)
            output = self._stack_layers(input=embedding, layers=layers)

            decoders.append(decoder)
            decoder_outputs.append(output)
        return decoders, decoder_outputs

    def _standardize_data(self, train_data_list, val_data_list):
        train_data_scaled = []
        val_data_scaled = []
        scalers = []

        for td, vd in zip(train_data_list,val_data_list):
            td_scaled, vd_scaled, scaler = super()._standardize_data(td, vd)

            train_data_scaled.append(td_scaled)
            val_data_scaled.append(vd_scaled)
            scalers.append(scaler)

        return train_data_scaled, val_data_scaled, scalers

    def _generate_kfold_split(self, n_splits, data, groups):
        kfold = GroupKFold(n_splits=n_splits)
        return kfold.split(data[0], data[0], groups)
    
    def generate_group_train_test_split(self, data, groups, random_state=100):
        return next(GroupShuffleSplit(random_state=random_state).split(data[0], groups=groups))

    def _prepare_data(self, data):
        data = [np.asarray(d) for d in data]
        return data

    def _index_data(self, data, idx):
        return [d[idx] for d in data]


class MultimodalAutoencoder(MultimodalBase):
    
    def _build(self, encoder_params, decoder_params, input_shapes,
               latent_shape, optimizer_params=None, loss="mean_squared_error"):
        
        self._check_encoder_params(encoder_params, latent_shape)

        if decoder_params is None:
            #from ipdb import set_trace; set_trace()
            dim = 0
            for shape in input_shapes:
                dim += shape[0]
            decoder_params = self._create_decoder_parameters_from_encoder(encoder_params, output_dim=dim)

        encoder_params = self._create_n_encoder_dicts(encoder_params, n=len(input_shapes))

        # TODO: self._check_decoder_params(decoder_params, input_shapes)

        encoder, decoder, single_modal_aes, ae = self._create_autoencoder(encoder_params,
                                                       decoder_params,
                                                       input_shapes,
                                                       latent_shape)
        self.encoder = encoder
        self.decoder = decoder
        self.ae = ae
        self.single_modal_aes = single_modal_aes

        self.optimizer = self._create_optimizer(optimizer_params)
        self.ae.compile(optimizer=self.optimizer, loss=loss)

    def fit(self, X, validation_data=None, *args, **kwargs):
        """Trains a keras autoencoder model
        
        Parameters
        ----------
        X : List of numpy arrays.
        """
        if validation_data is not None:
            validation_data = self._create_validation_data(validation_data)

        self.ae.fit(x=X, y=self._create_output(X), validation_data=validation_data, **kwargs)

    def _join_dataset(self,X):
        return np.concatenate(X,1)

    def _create_output(self,X):
        n = len(X)
        return [self._join_dataset(X)]*n

    def _create_autoencoder(self, encoder_params, decoder_params, input_shapes, latent_shape):
        """Creates an autoencoder model from dicts containing the parameters"""
        encoder_inputs = []
        encoders = [] #shares architecture
        autoencoders = [] #shares architecture and decoder weights
        
        decoder, _, decoder_layers = self._create_model(decoder_params, latent_shape)

        for params, input_shape in zip(encoder_params, input_shapes):
            encoder, input, encoder_layers = self._create_model(params, input_shape)

            autoencoder = self._create_model_from_layers(input=input, layers=encoder_layers+decoder_layers)

            encoder_inputs.append(input)
            encoders.append(encoder)
            autoencoders.append(autoencoder)

        outputs = [autoencoder.output for autoencoder in autoencoders]
        combined_autoencoder = km.Model(inputs=encoder_inputs, outputs=outputs)

        return encoders, decoder, autoencoders, combined_autoencoder

    def _rmse(self, val_data):
        #predictions = self.predict(val_data)
        #val_rmse = np.sqrt(((self._join_dataset(predictions) - self._join_dataset(self._create_output(val_data)))**2).mean())
        #return val_rmse
        
        
        predictions = self.predict(val_data)
        val_rmse = []
        for p in predictions:
            val_rmse.append(np.sqrt(np.mean((p-self._create_output(val_data))**2)))
        return val_rmse

    def _create_validation_data(self, val_data):
        return (val_data, self._create_output(val_data))

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

    ae = MultimodalAutoencoder(config["encoder"],
                     None,
                     input_shapes=input_shapes,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    groups = data_reader.groups

    experiment = Experiment(project_name="Multimodal Autoencoder", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "multimodal test")
    experiment.log_parameter("Architecture file name", config_filename)
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])
    scores = ae.cross_validate(data, groups, experiment=experiment, epochs=1000, n_splits=4, callbacks = [kc.EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)])

    print(scores)