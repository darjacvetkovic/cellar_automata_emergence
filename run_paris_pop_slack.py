import numpy as np
import pandas as pd
import GoL as gol
import train_dynamics as td
import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt
from celluloid import Camera
from tqdm import tqdm
import numpy as np
from torch import nn
from pathlib import Path
import seaborn as sns
import random
import os
from torch.utils.data import IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

torch.set_printoptions(precision=4)


seed = td.seed_python_numpy_torch_cuda(seed=None)
path = f"saved_models_dynamics/GoL/{seed}/"
Path(path).mkdir(parents=True, exist_ok=False)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
torch.set_default_dtype(torch.float32)

print(f"\nDevice: {device}")

################
# Load dataset #
################
dynamics_parameters = {
    "game_of_life": {
        "universe_size": (20, 20),
        "gol_seed": "spaceship",
        "n_generations": 30,
        "quality": 100,
        "cmap": "binary",
        "interval": 300,
        "seed_position": (9, 9),
        "animate": False,
        "save": False,
    }
}



batch_size = 12  # FIX, batch is being ignore yet
#dataset = td.TemporalDynamicsGameofLife_Iterable(dynamics_parameters["game_of_life"])
#train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
dataset = td.TemporalDynamicsFrance('data/all_grid_population.pkl', dynamics_parameters)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
###########################
# Initilise the NCA model #
###########################
extra_hidden_channels = 12
target_channels = 1
input_channels = target_channels + extra_hidden_channels
# number of output features maps for each convolutional layer (!= number of kernels actually used in the convolutional layer), conv layers create a different per input channel
num_output_conv_features = 6
num_conv_layers = 1
num_extra_fc_layers = 0
bias = False
activation_conv = torch.nn.Identity()  # torch.nn.LeakyReLU(), torch.nn.Tanh(), torch.nn.Identity(), torch.nn.Sigmoid()
activation_fc = torch.nn.Identity()  # torch.nn.LeakyReLU(), torch.nn.Tanh(), torch.nn.Identity(), torch.nn.Sigmoid()
activation_last = torch.nn.Identity()  # torch.nn.LeakyReLU(), torch.nn.Tanh(), torch.nn.Identity(), torch.nn.Sigmoid()

# torch.nn.LeakyReLU(), torch.nn.ReLU6(), torch.nn.Tanh(), torch.nn.Identity(), torch.nn.Sigmoid(), lambda x: torch.clamp(x, 0, 1)
state_activation = torch.nn.Tanh()


# Initialize the model
nca_model = td.NCA_conv(
    input_channels=input_channels,
    num_output_conv_features=num_output_conv_features,
    num_conv_layers=num_conv_layers,
    num_extra_fc_layers=num_extra_fc_layers,
    bias=bias,
    activation_conv=activation_conv,
    activation_fc=activation_fc,
    activation_last=activation_last,
).to(device)

#######################
# Train the NCA model #
#######################

epochs = 200  # Number of training steps
learning_rate = 0.01  # Learning rate for the optimizer
nca_steps = dynamics_parameters["game_of_life"]["n_generations"]
nca_steps = 243
additive_update_delta = 1
clamp = True

print(f"\nNumber of NCA steps: {nca_steps}")
print(f"Additive update delta: {additive_update_delta}\n")


# Train the model
trained_model, losses = td.train_dynamics(
    model=nca_model,
    device=device,
    train_loader=train_loader,
    optimizer="adam",
    epochs=epochs,
    loss_function=torch.nn.MSELoss(),  # torch.nn.L1Loss(), torch.nn.MSELoss()
    lr=learning_rate,
    additive_update_delta=additive_update_delta*0.1,
    state_activation=state_activation,
    clamp=clamp,
    dynamic_steps=243-1,
    target_channels=target_channels,
    extra_hidden_channels=extra_hidden_channels,
)

torch.save(trained_model.state_dict(), path + "best_model.pth")
td.plot_loss(losses, path, "training")



# #############################
# # Visualise resulting model #
# #############################

# Evaluate the model
intermediate_states, test_losses = td.test_nca_dynamics(
    model=nca_model,
    device=device,
    test_loader=train_loader,
    loss_function=torch.nn.MSELoss(),  # torch.nn.L1Loss(), torch.nn.MSELoss()
    additive_update_delta=additive_update_delta,
    state_activation=state_activation,
    clamp=clamp,
    dynamic_steps=nca_steps,
    target_channels=target_channels,
    extra_hidden_channels=extra_hidden_channels,
)

# plot_loss(test_losses, path)
td.create_animation_celluloid(intermediate_states, path=path + "animation.mp4")

# Extract target sequence from the loader
target_sequence = [data[1].detach().numpy() for data in train_loader]
target_sequence = np.concatenate(target_sequence, axis=0)  # Form a continuous sequence

td.create_side_by_side_animation(intermediate_states[0][:-1,:1,:,:], target_sequence[0], "Comparison", path=path + "comparison_animation.mp4")


# ##############
# # Save model #
# ##############

# Parameters to be saved
saved_parameters = {
    "seed": seed,
    "nca_steps": nca_steps,
    "model_state_dict": trained_model.state_dict(),
    # "target_shape": target_image,
    # "downsample_factor": downsample_factor,
    "input_channels": input_channels,
    "num_output_conv_features": num_output_conv_features,
    "num_conv_layers": num_conv_layers,
    "num_extra_fc_layers": num_extra_fc_layers,
    "bias": bias,
    "state_activation": state_activation,
    "activation_conv": activation_conv,
    "activation_fc": activation_fc,
    "activation_last": activation_last,
    "target_channels": target_channels,
    "extra_hidden_channels": extra_hidden_channels,
    "additive_update_delta": additive_update_delta,
    "clamp": clamp,
    "device": device,
}

# Save the parameters
torch.save(saved_parameters, path + "/nca_model_and_params.pth")

print(f"\nModel saved at {path}")


