from comet_ml import Experiment
from keras.datasets import mnist
import numpy as np
from autoencode import Autoencoder
import matplotlib.pyplot as plt

if __name__== "__main__":
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    config = {
        "decoder": [
            {
                "kwargs": {
                    "activation": "relu",
                    "units": 256
                },
                "name": "hidden1_decoder",
                "type": "Dense"
            },
            {
                "kwargs": {
                    "activation": "linear",
                    "units": x_train.shape[1]
                },
                "name": "output",
                "type": "Dense"
            }
        ],
        "encoder": [
            {
                "kwargs": {
                    "activation": "relu",
                    "units": 256
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
                    "units": 32
                },
                "name": "latent",
                "regularizer": {
                    "type": "l1",
                    "value": 0.1
                },
                "type": "Dense"
            }
        ]
    }

    latent_dim = config["encoder"][-1]["kwargs"]["units"]
    latent_shape = (latent_dim,)
    input_shape = (x_train.shape[1],)

    print(latent_shape)
    print(input_shape)

    ae = Autoencoder(config["encoder"],
                     config["decoder"],
                     input_shape=input_shape,
                     latent_shape=latent_shape,
                     loss="mean_squared_error",
                     optimizer_params=None)


    experiment = Experiment(api_key="ac4P1dtMEjJf1d9hIo9CIuSXC", project_name="mnist-autoencode")
    # experiment = Experiment(project_name="MNIST test", api_key="50kNmWUHJrWHz3FlgtpITIsB1")
    experiment.log_parameter("Experiment name", "Testing different layers")
    experiment.log_multiple_params(config)
    experiment.log_parameter("Latent dim", latent_shape[0])


    ae.fit(x_train, epochs=5, validation_data=(x_test, x_test))

    predictions = ae.predict(x_test)

    scores = np.sqrt(((predictions - x_test)**2).mean())
    experiment.log_other("scores", scores)
    print(scores)

    print(predictions.shape)
    pred_imgs = predictions.reshape(-1,28,28)
    for i,img in enumerate(pred_imgs[:5]):
        fig = plt.figure()
        fig.add_subplot(211)
        plt.imshow(img)
        fig.add_subplot(212)
        plt.imshow(x_test[i].reshape(28,28))
    plt.show()
