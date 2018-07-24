from comet_ml import Experiment
from keras.datasets import mnist
import numpy as np
from autoencode import Autoencoder
from variational_autoencode import VariationalAutoencoder   
from multimodal_autoencode import MultimodalAutoencoder
from shared_embedding_autoencode import SharedEmbeddingAutoencoder

import matplotlib.pyplot as plt
def test_variational_autoencoder():    
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))    
    
    config = {
        "encoder": [
            {
                "kwargs": {
                    "activation": "relu",
                    "units": 256
                },
                "name": "hidden1",
                "type": "Dense"
            },
            {
                "kwargs": {
                    "activation": "relu",
                },
                "name": "latent",
                "regularizer": {
                    "type": "l1",
                    "value": 0
                },
                "type": "Dense"
            }
        ],
        "decoder": [
            {
                "kwargs": {
                    "activation": "relu",
                    "units": 256
                },
                "name": "hidden1",
                "type": "Dense"
            },
            {
                "kwargs": {
                    "activation": "sigmoid",
                },
                "name": "output",
                "type": "Dense"
            }
        ]

    }

    latent_dim = 2
    latent_shape = (latent_dim,)
    input_shape = (x_train.shape[1],)

    print(latent_shape)
    print(input_shape)

    ae = VariationalAutoencoder(config["encoder"],
                     config["decoder"],
                     input_shape=input_shape,
                     latent_shape=latent_shape,
                     optimizer_params=None)

    #experiment = Experiment(api_key="ac4P1dtMEjJf1d9hIo9CIuSXC", project_name="mnist-autoencode")
    experiment = Experiment(project_name="MNIST test", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing vae")
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])

    ae.fit(x_train, batch_size=1000, epochs=10, validation_data=x_test)

    predictions = ae.predict(x_test)

    scores = np.sqrt(((predictions - x_test)**2).mean())
    experiment.log_other("scores", scores)
    print(f"score: {scores}")

    pred_imgs = predictions.reshape(-1,28,28)
    fig = plt.figure()
    for i,img in enumerate(pred_imgs[:5]):
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img)
        plt.axis('off') 
        fig.add_subplot(2, 5, i+6)
        plt.imshow(x_test[i].reshape(28,28))
        plt.axis('off') 
    plt.show()

def test_autoencoder():    
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))    



    config = {
        "encoder": [
            {
                "kwargs": {
                    "activation": "relu",
                    "units": 256
                },
                "name": "hidden1",
                "type": "Dense"
            },
            {
                "name": "batchnorm",
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
                    "activation": "sigmoid",
                },
                "name": "latent",
                "regularizer": {
                    "type": "l1",
                    "value": 0
                },
                "type": "Dense"
            }
        ]
    }

    latent_dim = 32
    latent_shape = (latent_dim,)
    input_shape = (x_train.shape[1],)

    print(latent_shape)
    print(input_shape)

    ae = Autoencoder(config["encoder"],
                     None,
                     input_shape=input_shape,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    #experiment = Experiment(api_key="ac4P1dtMEjJf1d9hIo9CIuSXC", project_name="mnist-autoencode")
    experiment = Experiment(project_name="MNIST test", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing ae")
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])


    ae.fit(x_train, batch_size=1000, epochs=5, validation_data=x_test)

    predictions = ae.predict(x_test)

    scores = np.sqrt(((predictions - x_test)**2).mean())
    experiment.log_other("scores", scores)
    print(scores)

    print(predictions.shape)
    pred_imgs = predictions.reshape(-1,28,28)
    fig = plt.figure()
    for i,img in enumerate(pred_imgs[:5]):
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img)
        plt.axis('off') 
        fig.add_subplot(2, 5, i+6)
        plt.imshow(x_test[i].reshape(28,28))
        plt.axis('off') 
    plt.show()

def test_multimodal_autoencoder():    
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) 


    f_idx = np.random.permutation(x_test.shape[1])
    n = int(len(f_idx)/3)
    print(n)
    print(x_train[f_idx[n*0:n*0+n]].shape)
    x_train_list = [x_train[:,f_idx[n*i:n*i+n]] for i in range(3)]
    x_test_list = [x_test[:,f_idx[n*i:n*i+n]] for i in range(3)]

    input_shapes = [(d.shape[1],) for d in x_train_list]

    config = {
        "encoder": [
            {
                "kwargs": {
                    "activation": "relu",
                    "units": 256
                },
                "name": "hidden1",
                "type": "Dense"
            },
            {
                "name": "batchnorm",
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
                    "activation": "sigmoid",
                },
                "name": "latent",
                "regularizer": {
                    "type": "l1",
                    "value": 0
                },
                "type": "Dense"
            }
        ]
    }

    latent_dim = 32
    latent_shape = (latent_dim,)

    print(latent_shape)
    print(input_shapes)

    ae = MultimodalAutoencoder(config["encoder"],
                     None,
                     input_shapes=input_shapes,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    #experiment = Experiment(api_key="ac4P1dtMEjJf1d9hIo9CIuSXC", project_name="mnist-autoencode")
    experiment = Experiment(project_name="MNIST test", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing mae")
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])


    ae.fit(x_train_list, batch_size=1000, epochs=10, validation_data=x_test_list)

    predictions = ae.predict(x_test_list)
    print(predictions)
    #scores = [np.sqrt(((p- x)**2).mean()) for p,x in zip(predictions, x_test_list)]
    #experiment.log_other("scores", scores)
    #print(scores)

    #x_test_list = [x_test[f_idx[n*i:n*i+n]] for i in range(2)]
    all_predictions = []
    for i in range(3):
        predictions_combined = np.empty_like(x_test)
        print(x_test.shape)
        print(predictions[i].shape)
        print(predictions_combined[:,f_idx].shape)
        predictions_combined[:, f_idx[:-1]] = predictions[i]

        all_predictions.append(predictions_combined.reshape(-1,28,28))


    #pred_imgs = predictions_combined.reshape(-1,28,28)
    for pred_imgs in all_predictions:
        fig = plt.figure()
        for i,img in enumerate(pred_imgs[:5]):
            fig.add_subplot(2, 5, i+1)
            plt.imshow(img)
            plt.axis('off') 
            fig.add_subplot(2, 5, i+6)
            plt.imshow(x_test[i].reshape(28,28))
            plt.axis('off') 
    plt.show()


def test_shared_embedding_autoencoder():    
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) 


    f_idx = np.random.permutation(x_test.shape[1])
    n = int(len(f_idx)/3)
    print(n)
    print(x_train[f_idx[n*0:n*0+n]].shape)
    x_train_list = [x_train[:,f_idx[n*i:n*i+n]] for i in range(3)]
    x_test_list = [x_test[:,f_idx[n*i:n*i+n]] for i in range(3)]

    input_shapes = [(d.shape[1],) for d in x_train_list]

    config = {
        "encoder": [
            {
                "kwargs": {
                    "activation": "relu",
                    "units": 256
                },
                "name": "hidden1",
                "type": "Dense"
            },
            {
                "name": "batchnorm",
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
                    "activation": "sigmoid",
                },
                "name": "latent",
                "regularizer": {
                    "type": "l1",
                    "value": 0
                },
                "type": "Dense"
            }
        ]
    }

    latent_dim = 32
    latent_shape = (latent_dim,)

    print(latent_shape)
    print(input_shapes)

    ae = SharedEmbeddingAutoencoder(config["encoder"],
                     None,
                     input_shapes=input_shapes,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)

    #experiment = Experiment(api_key="ac4P1dtMEjJf1d9hIo9CIuSXC", project_name="mnist-autoencode")
    experiment = Experiment(project_name="MNIST test", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing mae")
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])


    ae.fit(x_train_list, batch_size=1000, epochs=10, validation_data=x_test_list)

    predictions = ae.predict(x_test_list)
    print([p.shape for p in predictions])
    #scores = [np.sqrt(((p- x)**2).mean()) for p,x in zip(predictions, x_test_list)]
    #experiment.log_other("scores", scores)
    #print(scores)

    #x_test_list = [x_test[f_idx[n*i:n*i+n]] for i in range(2)]
    
    predictions_combined = np.empty_like(x_test)
    for i in range(3):
        predictions_combined[:,f_idx[n*i:n*i+n]] = predictions[i]

    pred_imgs = predictions_combined.reshape(-1,28,28)
    print(pred_imgs.shape)
    fig = plt.figure()
    for i,img in enumerate(pred_imgs[:5]):
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img)
        plt.axis('off') 
        fig.add_subplot(2, 5, i+6)
        plt.imshow(x_test[i].reshape(28,28))
        plt.axis('off') 
    plt.show()

if __name__== "__main__":
    #test_autoencoder()
    #test_multimodal_autoencoder()
    test_variational_autoencoder()
    #test_shared_embedding_autoencoder()