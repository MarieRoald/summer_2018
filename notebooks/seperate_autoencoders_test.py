from comet_ml import Experiment
import numpy as np
from autoencode import Autoencoder
from data_reader import DataReader

filenames = ["X1_train.csv", "X2_train.csv", "X3_train.csv"]



data_reader = DataReader(data_set_filenames=filenames, groups_filename="ID_train.csv")
data = data_reader.get_seperate_data()

l1_regs = [0] + [10**(-i) for i in range(1,6)]
input_shapes = [(d.shape[1],) for d in data]

for latent_dim in [32, 128, 256, 512]:
    for hidden_dim in [128, 256]:
        for l1 in l1_regs:
            latent_shape = (latent_dim,)


            config = [
                {
                    "decoder": [
                        {
                            "kwargs": {
                                "activation": "relu",
                                "units": hidden_dim
                            },
                            "name": "hidden1_decoder",
                            "type": "Dense"
                        },
                        {
                            "kwargs": {
                                "activation": "linear",
                                "units": data[0].shape[1]
                            },
                            "name": "output",
                            "type": "Dense"
                        }
                    ],
                    "encoder": [
                        {
                            "kwargs": {
                                "activation": "relu",
                                "units": hidden_dim
                            },
                            "name": "hidden1_encoder",
                            "type": "Dense"
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
                                "activation": "sigmoid",
                                "units": latent_shape[0]
                            },
                            "name": "latent",
                            "regularizer": {
                                "type": "l1",
                                "value": l1
                            },
                            "type": "Dense"
                        }
                    ]
                },
                {
                    "decoder": [
                        {
                            "kwargs": {
                                "activation": "relu",
                                "units": hidden_dim
                            },
                            "name": "hidden1_decoder",
                            "type": "Dense"
                        },
                        {
                            "kwargs": {
                                "activation": "linear",
                                "units": data[1].shape[1]
                            },
                            "name": "output",
                            "type": "Dense"
                        }
                    ],
                    "encoder": [
                        {
                            "kwargs": {
                                "activation": "relu",
                                "units": hidden_dim
                            },
                            "name": "hidden1_encoder",
                            "type": "Dense"
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
                                "activation": "sigmoid",
                                "units": latent_shape[0]
                            },
                            "name": "latent",
                            "regularizer": {
                                "type": "l1",
                                "value": l1
                            },
                            "type": "Dense"
                        }
                    ]
                },
                {
                    "decoder": [
                        {
                            "kwargs": {
                                "activation": "relu",
                                "units": hidden_dim
                            },
                            "name": "hidden1_decoder",
                            "type": "Dense"
                        },
                        {
                            "kwargs": {
                                "activation": "linear",
                                "units": data[2].shape[1]
                            },
                            "name": "output",
                            "type": "Dense"
                        }
                    ],
                    "encoder": [
                        {
                            "kwargs": {
                                "activation": "relu",
                                "units": hidden_dim
                            },
                            "name": "hidden1_encoder",
                            "type": "Dense"
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
                                "activation": "sigmoid",
                                "units": latent_shape[0]
                            },
                            "name": "latent",
                            "regularizer": {
                                "type": "l1",
                                "value": l1
                            },
                            "type": "Dense"
                        }
                    ]
                }
            ]
            # latent_dim = config["encoder"][-1]["kwargs"]["units"]
            # latent_shape = (latent_dim,)
            experiment = Experiment(project_name="Seperate autoencoders", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
            experiment.log_parameter("Experiment name", "Testing different layers")
            experiment.log_parameter("Latent dim", latent_shape[0])

            groups = data_reader.get_groups()
            all_scores = []

            for i in range(3):
                ae = Autoencoder(config[i]["encoder"],
                                config[i]["decoder"],
                                input_shape=input_shapes[i],
                                latent_shape=latent_shape,
                                loss="mean_squared_error",
                                optimizer_params=None)

                experiment.log_multiple_params(config[i])

                scores = ae.cross_validate(data[i],
                                            groups, 
                                            experiment=experiment, 
                                            epochs=10000, 
                                            n_splits=4,
                                            log_prefix=f"dataset_{i}_")

                all_scores.append(scores)

                mean_scores= np.mean(scores)

                experiment.log_metric(f"mean_scores_{i}", mean_scores)

                experiment.log_other(f"scores_{i}", scores)

            experiment.log_metric(f"mean_all_scores", np.mean(all_scores))
            print(all_scores)
