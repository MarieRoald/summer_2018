from autoencode import Autoencoder
import keras.models as km
import keras.layers as kl
from log_utils import plot_mean_std_loss, GroupedCometLogger
from data_reader import DataReader

class MultimodalAutoencoder(Autoencoder):

    
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

    def _create_autoencoder(self, encoder_config, decoder_config, input_shapes, latent_shape):
        """Creates an autoencoder model from dicts containing the parameters"""
        # TODO: skal jeg kunne ta inn en liste med input og output navn?

        encoder_layers = []
        encoder_inputs = []
        encoders = [] #shares architecture
        autoencoders = [] #shares architecture and decoder weights
        
        decoder, decoder_input, self.decoder_layers = self._create_decoder(decoder_config, latent_shape)

        #TODO: kanskje litt merkelig at self.decoder_layers ikke har input lag,
        #      men Autoencoder sin self.decoder_layers har det
        for i, input_shape in enumerate(input_shapes):

            current_encoder, current_input, current_encoder_layers = self._create_encoder(encoder_config, input_shape)

            encoder_inputs.append(current_input)
            encoder_layers.append(current_encoder_layers)
            encoders.append(current_encoder)

            current_autoencoder_layers = current_encoder_layers+self.decoder_layers
            current_autoencoder = self._create_model_from_layers(input=current_input, layers=current_autoencoder_layers)
            autoencoders.append(current_autoencoder)
            
        combined_autoencoder = km.Model(input=encoder_inputs, outputs=autoencoders)

        self.encoder_layers = encoder_layers
        return encoders, decoder, autoencoders, combined_autoencoder

    # TODO: denne er helt lik _create_decoder
    def _create_encoder(self, encoder_config, input_shape):
        encoder_layers = self._create_layers(encoder_config)
        encoder_input = kl.Input(shape=input_shape)
        encoder = self._create_model_from_layers(input=encoder_input, layers=encoder_layers)
        return encoder, encoder_input, encoder_layers

    def _create_decoder(self, decoder_config, latent_shape):
        decoder_layers = self._create_layers(decoder_config)
        decoder_input = kl.Input(shape=latent_shape)
        decoder = self._create_model_from_layers(input=decoder_input, layers=self.decoder_layers) 
        return decoder, decoder_input, decoder_layers

    def _stack_layers(self, layers):
        current_input_layer = layers[0]
        for layer in layers[1:]:
            current_input_layer = layer(current_input_layer)

    def _create_model_from_layers(self, input, layers):
        current_input_layer = layers[0]
        for layer in layers[1:]:
            current_input_layer = layer(current_input_layer)

        model = km.Model(inputs=layers[0], outputs=current_input_layer)
        return model

    filenames = ["X1_train.csv", "X2_train.csv", "X3_train.csv"]
    data_reader = DataReader(data_set_filenames=filenames, groups_filename="ID_train.csv")
    data = data_reader.get_seperate_data()


    input_shapes = [(d.shape[1],) for d in data]
    latent_shape = (1000,)

    config = {
        "encoder": [
            {
                "name": "hidden1_encoder",
                "type": "Dense",
                "kwargs": {
                    "units": 2500,
                    "activation": "relu"
                },
                "regularizer": {
                    "type": "l1",
                    "value": 1e-3
                }
            },
            {
                "name": "hidden2_encoder",
                "type": "Dense",
                "kwargs": {
                    "units": 2000,
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
                    "activation": "linear"
                },

                "regularizer": {
                    "type": "l1",
                    "value": 1e-3
                }
            }
        ],
        "decoder": [
            {
                "name": "hidden2_decoder",
                "type": "Dense",
                "kwargs": {
                    "units": 2000,
                    "activation": "relu"
                }
            },
            {
                "name": "hidden1_decoder",
                "type": "Dense",
                "kwargs": {
                    "units": 2500,
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

    ae = MultimodalAutoencoder(config["encoder"],
                     config["decoder"],
                     input_shapes=input_shapes,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    groups = data_reader.get_groups()

    experiment = Experiment(project_name="comet test", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "More layers and units test")
    scores = ae.cross_validate(data, groups, experiment=experiment, epochs=100, n_splits=10)