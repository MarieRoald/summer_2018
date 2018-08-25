import json
from pprint import pprint

def create_dropout_params(dr):
    dropout_params = {
                        "name": "dropout",
                        "type": "Dropout",
                        "kwargs": {
                            "rate":dr
                        }
                      }
    return dropout_params

def create_latent_layer_params(latent_dim, activation, l1):
    latent_params = {
                        "name": "latent",
                        "type": "Dense",
                        "kwargs": {
                            "units": latent_dim,
                            "activation": activation
                        },
                        "regularizer": {
                            "type": "l1",
                            "value": l1
                        }
                    }
    return latent_params
def create_hidden_layer_params(hidden_dim, i):
    hidden_params = {
                        "name": f"hidden{i}",
                        "type": "Dense",
                        "kwargs": {
                            "units": hidden_dim,
                            "activation": "relu"
                        }
                    }
    return hidden_params
def create_batchnorm_layer_params(i):
    batchnorm_params = {
                            "name": f"batchnorm{i}",
                            "type": "BatchNormalization"
                       }
    return batchnorm_params

def create_architecture(hidden_dims, latent_dim, dropout_rate, l1, final_activation):
   architecture = []
   for i,h in enumerate(hidden_dims):
       architecture.append(create_hidden_layer_params(h,i+1))
       architecture.append(create_batchnorm_layer_params(i+1))

   architecture.append(create_dropout_params(dropout_rate))
   architecture.append(create_latent_layer_params(latent_dim, final_activation, l1))
   return architecture

hidden_dims = [256, 512, 1024]
latent_dims = [32, 128, 512, 1024]
dropout_rate = [0, 0.5]

l1_regs = [0, 0.01, 0.001, 0.0001]

count = 0
for h in hidden_dims:
    for l in latent_dims:
        for dr in dropout_rate:
           for l1 in l1_regs: 
                if l<h:
                    print(h,l)
                    count += 1
                    config = {"encoder":create_architecture([h],l,dropout_rate=dr,l1=l1,final_activation="relu")}
                    with open(f"recsys_architectures2/two_layer_{count}.json", "w") as outfile:
                        json.dump(config, outfile, sort_keys=True, indent=4)

print(count)