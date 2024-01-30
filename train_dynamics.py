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


def seed_python_numpy_torch_cuda(seed: int):
    if seed is None:
        rng = np.random.default_rng()
        seed = int(rng.integers(2**32, size=1)[0])
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"\nSeeded with {seed}")
    return seed


class TemporalDynamicsGameofLife_Iterable(IterableDataset):
    """
    An iterable Dataset.
    All datasets that represent an iterable of data samples should subclass it.
    All subclasses should overwrite __iter__(), which would return an iterator of samples in this dataset.
    next(iterable_dataset) is called on the iterator created by __iter__() method until it has built a full batch.
    """

    def __init__(self, config):
        from GoL import animate_life

        self.universe_size = config["universe_size"]
        self.n_generations = config["n_generations"]
        self.seed = config["gol_seed"]
        self.seed_position = config["seed_position"]

        gol_dynamics, initial_seed = animate_life(
            universe_size=self.universe_size, seed=self.seed, seed_position=self.seed_position, n_generations=self.n_generations
        )
        self.initial_state = torch.unsqueeze(torch.from_numpy(initial_seed), 0)  # channels, x, y
        self.target_dynamics = torch.unsqueeze(torch.from_numpy(gol_dynamics), 1)  # steps, channels, x, y
        print(f"Initial state shape: {self.initial_state.shape}")
        print(f"Target dynamics shape: {self.target_dynamics.shape}")

    def __iter__(self):
        yield self.initial_state, self.target_dynamics


class NCA_conv(nn.Module):
    def __init__(
        self,
        input_channels,
        num_output_conv_features,
        num_conv_layers,
        num_extra_fc_layers,
        bias,
        activation_conv,
        activation_fc,
        activation_last,
    ):
        super(NCA_conv, self).__init__()
        self.activation_conv = activation_conv
        self.activation_fc = activation_fc
        self.activation_last = activation_last

        # Define all convolutional layers (including the initial one)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=num_output_conv_features, kernel_size=3, padding=1, bias=bias),
            *[
                nn.Conv2d(in_channels=num_output_conv_features, out_channels=num_output_conv_features, kernel_size=3, padding=1, bias=bias)
                for _ in range(num_conv_layers - 1)
            ],
        )

        # Define fully connected layers as 1x1 convolutional layers
        # Number of parameters=(kernel height × kernel width × input channels+bias) × output channels
        self.fc_layers = nn.Sequential(
            *[
                nn.Conv2d(in_channels=num_output_conv_features, out_channels=num_output_conv_features, kernel_size=1, bias=bias)
                for _ in range(num_extra_fc_layers)
            ]
        )

        # Define the output layer to match input channel size
        self.output_layer = nn.Conv2d(in_channels=num_output_conv_features, out_channels=input_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        # Apply the convolutional layers
        x = self.activation_conv(self.conv_layers(x))

        # Apply the fully connected layers
        x = self.activation_fc(self.fc_layers(x))

        # Apply the final output layer and activation function
        x = self.activation_last(self.output_layer(x))

        return x


def train_dynamics(
    model,
    device,
    train_loader,
    optimizer,
    epochs,
    loss_function,
    lr,
    additive_update_delta,
    state_activation,
    clamp,
    dynamic_steps,
    target_channels,
    extra_hidden_channels,
):
    for param in model.parameters():
        param.requires_grad = True

    if optimizer == "adamW":
        print("Using adamW optimizer")
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == "adam":
        print("Using adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "adamax":
        print("Using adamax optimizer")
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    else:
        raise ValueError("Choose a valid optimizer")

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=500, threshold=0.1, cooldown=500, min_lr=0.0001, verbose=True)

    tic = time.time()
    loss_best_model = 1e10
    loss_tracking = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_cum = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            state_sequence = run_nca_dynamics(model, x, dynamic_steps, additive_update_delta, state_activation, clamp)

            loss = loss_function(state_sequence[:, :, :target_channels, :, :], y)
            loss.backward()
            optimizer.step()

            loss_cum += float(loss)

        toc = time.time()
        print(f"Epoch {epoch}: loss: {loss_cum:.8f} | lr: {optimizer.param_groups[0]['lr']:.8f} | Elapsed time: {toc - tic:.2f}s")

        scheduler.step(loss_cum)
        loss_tracking.append(loss_cum)

    if loss_cum < loss_best_model:
        best_model = model

    return best_model, loss_tracking


def run_nca_dynamics(model, initial_state, dynamic_steps, additive_update_delta, state_activation, clamp):
    states = [initial_state]

    for step in range(dynamic_steps):
        # USE RELEVANT ACTIVATION / CLAMPING
        next_state = states[-1] + additive_update_delta * model(states[-1])
        next_state = state_activation(next_state)
        if clamp:
            next_state = torch.clamp(next_state, 0, 1)
        states.append(next_state)

    return torch.stack(states, dim=1)


def test_nca_dynamics(
    model,
    device,
    test_loader,
    loss_function,
    dynamic_steps,
    additive_update_delta,
    state_activation,
    clamp,
    target_channels,
):
    model.eval()
    all_states = []
    all_losses = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            state_sequence = run_nca_dynamics(model, x, dynamic_steps, additive_update_delta, state_activation, clamp)

            # apply discretisation here

            # Compute losses for each step
            losses = [loss_function(state_sequence[:, step, :target_channels, :, :], y[:, step]).item() for step in range(dynamic_steps)]

            # Store the states and losses for all batches
            all_states.extend([s.cpu().numpy() for s in state_sequence])
            all_losses.append(losses)

    all_losses = np.array(all_losses)
    all_states = np.array(all_states)

    return all_states, all_losses


def plot_loss(losses, path, label):
    # sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    plt.plot(losses, color="black", linestyle="-", linewidth=2, marker="o", markersize=8, markerfacecolor="orange")
    plt.title("NCA Training Loss", fontsize=18, fontweight="bold", color="black")
    plt.xlabel("Epoch", fontsize=14, fontweight="regular")
    plt.ylabel("Loss", fontsize=14, fontweight="regular")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path + label + "_loss.png", format="png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    start_epoch = int(len(losses) / 2)
    x_values = range(start_epoch, start_epoch + len(losses[start_epoch:]))
    plt.plot(x_values, losses[start_epoch:], color="black", linestyle="-", linewidth=2, marker="o", markersize=8, markerfacecolor="orange")
    plt.title("NCA Training Loss", fontsize=18, fontweight="bold", color="black")
    plt.xlabel("Epoch", fontsize=14, fontweight="regular")
    plt.ylabel("Loss", fontsize=14, fontweight="regular")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path + label + "_loss_last_50.png", format="png", dpi=300)
    plt.close()


def create_side_by_side_animation(generated_states, target_states, title, path=None):
    assert generated_states.shape == target_states.shape, "Generated and target states must have the same shape"

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axs:
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    camera = Camera(fig)

    for gen_state, tgt_state in tqdm(
        zip(generated_states, target_states), desc=f"Creating animation: {title}", total=len(generated_states)
    ):
        gen_state = np.clip(gen_state, 0, 1)
        tgt_state = np.clip(tgt_state, 0, 1)

        axs[0].imshow(gen_state.transpose(2, 1, 0), cmap="binary")
        axs[1].imshow(tgt_state.transpose(2, 1, 0), cmap="binary")

        camera.snap()

    animation = camera.animate(interval=1)
    if path is not None:
        animation.save(path, fps=2)
    else:
        plt.show()


def create_animation_celluloid(states, path=None):
    # Create figure and adjust its settings
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Initialize Camera
    camera = Camera(fig)

    for state in tqdm(states[0], desc="Creating animation"):
        state = np.clip(state, 0, 1)
        ax.imshow(state.transpose(2, 1, 0), cmap="binary")
        camera.snap()

    # Create and save the animation
    animation = camera.animate(interval=1)
    if path is not None:
        animation.save(path, fps=2)
    else:
        plt.show()


if __name__ == "__main__":
    seed = seed_python_numpy_torch_cuda(seed=None)
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
    dataset = TemporalDynamicsGameofLife_Iterable(dynamics_parameters["game_of_life"])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    ###########################
    # Initilise the NCA model #
    ###########################
    extra_hidden_channels = 0
    target_channels = 1
    input_channels = target_channels + extra_hidden_channels
    # number of output features maps for each convolutional layer (!= number of kernels actually used in the convolutional layer), conv layers create a different per input channel
    num_output_conv_features = 3
    num_conv_layers = 1
    num_extra_fc_layers = 0
    bias = False
    activation_conv = torch.nn.Identity()  # torch.nn.LeakyReLU(), torch.nn.Tanh(), torch.nn.Identity(), torch.nn.Sigmoid()
    activation_fc = torch.nn.Identity()  # torch.nn.LeakyReLU(), torch.nn.Tanh(), torch.nn.Identity(), torch.nn.Sigmoid()
    activation_last = torch.nn.Identity()  # torch.nn.LeakyReLU(), torch.nn.Tanh(), torch.nn.Identity(), torch.nn.Sigmoid()

    # torch.nn.LeakyReLU(), torch.nn.ReLU6(), torch.nn.Tanh(), torch.nn.Identity(), torch.nn.Sigmoid(), lambda x: torch.clamp(x, 0, 1)
    state_activation = torch.nn.Tanh()

    # Initialize the model
    nca_model = NCA_conv(
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

    epochs = 100  # Number of training steps
    learning_rate = 0.05  # Learning rate for the optimizer
    nca_steps = dynamics_parameters["game_of_life"]["n_generations"]
    additive_update_delta = 1
    clamp = True

    print(f"\nNumber of NCA steps: {nca_steps}")
    print(f"Additive update delta: {additive_update_delta}\n")

    # Train the model
    trained_model, losses = train_dynamics(
        model=nca_model,
        device=device,
        train_loader=train_loader,
        optimizer="adam",
        epochs=epochs,
        loss_function=torch.nn.MSELoss(),  # torch.nn.L1Loss(), torch.nn.MSELoss()
        lr=learning_rate,
        additive_update_delta=additive_update_delta,
        state_activation=state_activation,
        clamp=clamp,
        dynamic_steps=nca_steps,
        target_channels=target_channels,
        extra_hidden_channels=extra_hidden_channels,
    )

    torch.save(trained_model.state_dict(), path + "best_model.pth")
    plot_loss(losses, path, "training")

    # #############################
    # # Visualise resulting model #
    # #############################

    # Evaluate the model
    intermediate_states, test_losses = test_nca_dynamics(
        model=nca_model,
        device=device,
        test_loader=train_loader,
        loss_function=torch.nn.MSELoss(),  # torch.nn.L1Loss(), torch.nn.MSELoss()
        additive_update_delta=additive_update_delta,
        state_activation=state_activation,
        clamp=clamp,
        dynamic_steps=nca_steps,
        target_channels=target_channels,
    )

    # plot_loss(test_losses, path)
    create_animation_celluloid(intermediate_states, path=path + "animation.mp4")

    # Extract target sequence from the loader
    target_sequence = [data[1].numpy() for data in train_loader]
    target_sequence = np.concatenate(target_sequence, axis=0)  # Form a continuous sequence

    create_side_by_side_animation(intermediate_states[0], target_sequence[0], "Comparison", path=path + "comparison_animation.mp4")

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
