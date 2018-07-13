import json

hidden_dims = [256, 512, 1024]
latent_dims = [8, 32, 128, 512, 1024]
l1_regs = [0] + [10**(-i) for i in range(1,6)]
dropouts = [0, 0.5]
final_activation = ["sigmoid", "relu"]


i = 0
for hd in hidden_dims:
    for ld in latent_dims:
        for l1 in l1_regs:
            for dr in dropouts:
                for fa in final_activation:
                    config = {
                        "encoder": [
                            {
                                "name": "hidden1_encoder",
                                "type": "Dense",
                                "kwargs": {
                                    "units": hd,
                                    "activation": "relu"
                                }
                            },
                            {
                                "name": "dropout",
                                "type": "Dropout",
                                "kwargs": {
                                    "rate":dr
                                }

                            },
                            {
                                "name": "latent",
                                "type": "Dense",
                                "kwargs": {
                                    "units": ld,
                                    "activation": fa
                                },

                                "regularizer": {
                                    "type": "l1",
                                    "value": l1
                                }
                            }
                        ],
                        "decoder": [
                            {
                                "name": "hidden1_decoder",
                                "type": "Dense",
                                "kwargs": {
                                    "units": hd,
                                    "activation": "relu"
                                }
                            },
                            {
                                "name": "output",
                                "type": "Dense",
                                "kwargs": {
                                    "units": 3048,
                                    "activation": "linear"
                                }
                            }
                        ]
                    }
                    with open(f"architectures/two_layer_{i}.json", "w") as outfile:
                        json.dump(config, outfile, sort_keys=True, indent=4)

                    i += 1



hidden_dims1 = [512, 1024]
hidden_dims2 = [256, 512]
hidden_dims3 = [256, 512]
latent_dims = [8, 32, 128, 256]
l1_regs = [0] + [10**(-i) for i in range(1,6)]
dropouts = [0, 0.5]
final_activation = ["sigmoid", "relu"]


i = 0
for hd1 in hidden_dims1:
    for hd2 in hidden_dims2:
        for hd3 in hidden_dims3:
            for ld in latent_dims:
                for l1 in l1_regs:
                    for dr in dropouts:
                        for fa in final_activation:
                            config = {
                                "encoder": [
                                    {
                                        "name": "hidden1_encoder",
                                        "type": "Dense",
                                        "kwargs": {
                                            "units": hd1,
                                            "activation": "relu"
                                        }
                                    },
                                    {
                                        "name": "hidden2_encoder",
                                        "type": "Dense",
                                        "kwargs": {
                                            "units": hd2,
                                            "activation": "relu"
                                        }
                                    },
                                    {
                                        "name": "hidden3_encoder",
                                        "type": "Dense",
                                        "kwargs": {
                                            "units": hd3,
                                            "activation": "relu"
                                        }
                                    },
                                    {
                                        "name": "dropout",
                                        "type": "Dropout",
                                        "kwargs": {
                                            "rate":dr
                                        }

                                    },
                                    {
                                        "name": "latent",
                                        "type": "Dense",
                                        "kwargs": {
                                            "units": ld,
                                            "activation": fa
                                        },

                                        "regularizer": {
                                            "type": "l1",
                                            "value": l1
                                        }
                                    }
                                ],
                                "decoder": [
                                    {
                                        "name": "hidden3_decoder",
                                        "type": "Dense",
                                        "kwargs": {
                                            "units": hd3,
                                            "activation": "relu"
                                        }
                                    },
                                    {
                                        "name": "hidden2_decoder",
                                        "type": "Dense",
                                        "kwargs": {
                                            "units": hd2,
                                            "activation": "relu"
                                        }
                                    },
                                    {
                                        "name": "hidden1_decoder",
                                        "type": "Dense",
                                        "kwargs": {
                                            "units": hd1,
                                            "activation": "relu"
                                        }
                                    },
                                    {
                                        "name": "output",
                                        "type": "Dense",
                                        "kwargs": {
                                            "units": 3048,
                                            "activation": "linear"
                                        }
                                    }
                                ]
                            }
                            with open(f"architectures/four_layer_{i}.json", "w") as outfile:
                                json.dump(config, outfile, sort_keys=True, indent=4)

                            i += 1


hidden_dims1 = [1024]
hidden_dims2 = [512]
hidden_dims3 = [256]
hidden_dims4 = [256,128]
latent_dims = [8, 32, 128]
l1_regs = [0] + [10**(-i) for i in range(1,6)]
dropouts = [0, 0.5]
final_activation = ["sigmoid", "relu"]

i = 0
for hd1 in hidden_dims1:
    for hd2 in hidden_dims2:
        for hd3 in hidden_dims3:
            for hd4 in hidden_dims4:
                for ld in latent_dims:
                    for l1 in l1_regs:
                        for dr in dropouts:
                            for fa in final_activation:
                                config = {
                                    "encoder": [
                                        {
                                            "name": "hidden1_encoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd1,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden2_encoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd1,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden3_encoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd2,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden4_encoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd2,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden5_encoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd3,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden6_encoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd3,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden7_encoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd4,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "dropout",
                                            "type": "Dropout",
                                            "kwargs": {
                                                "rate":dr
                                            }

                                        },
                                        {
                                            "name": "latent",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": ld,
                                                "activation": fa
                                            },

                                            "regularizer": {
                                                "type": "l1",
                                                "value": l1
                                            }
                                        }
                                    ],
                                    "decoder": [
                                        {
                                            "name": "hidden7_decoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd4,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden6_decoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd3,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden5_decoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd3,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden4_decoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd2,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden3_decoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd2,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden2_decoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd1,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "hidden1_decoder",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": hd1,
                                                "activation": "relu"
                                            }
                                        },
                                        {
                                            "name": "output",
                                            "type": "Dense",
                                            "kwargs": {
                                                "units": 3048,
                                                "activation": "linear"
                                            }
                                        }
                                    ]
                                }
                                with open(f"architectures/eight_layer_{i}.json", "w") as outfile:
                                    json.dump(config, outfile, sort_keys=True, indent=4)

                                i += 1



