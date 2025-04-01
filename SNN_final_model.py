import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loading import get_data
from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc
import json
from datetime import date

# Set device
cuda_kernel = 0
today = date.today()
torch.cuda.set_device(cuda_kernel)
num_inputs = 300 * 300
num_outputs = 64 * 64  # 4096
# tau_list = [120, 140, 160, 180, 200] #For trying different taus
nr_par_last_layer_list = [4000]
w_decay = 1e-4
lr = 8.5e-4
print_image = (
    False  # Set true to save current image for each loop, used to see progress during training
)
nr_data_files = 26
loss_function = nn.MSELoss()
tau_mem = 180
k = 0


def save_data(model, train_loss, validation_loss, tau, today, epoch, k):
    file_name = f"{today}_Tau_200_newest_version_{tau}_4"

    torch.save(model.state_dict(), f"{file_name}.pth")

    data = {
        "Epoch": epoch,
        "Tau": tau,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
    }
    k += 1

    with open(f"{file_name}.json", "w") as f:
        json.dump(data, f)

    print(f"Model has been svaed to \x1b[1m{file_name}.pth\x1b[22m")
    print(f"Training stats saved to \x1b[1m{file_name}.json\x1b[22m")
    return k


def save_current_result(output_frame, target_frame, frames, step):
    maximum_value = torch.max(output_frame).item()
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    axes[0].imshow(frames[0][step].cpu().detach().numpy(), cmap="gray")
    axes[1].imshow(target_frame[0].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=0.03)
    axes[2].imshow(output_frame[0].cpu().detach().numpy(), cmap="gray", vmax=maximum_value)

    axes[0].axis("off")
    axes[1].axis("off")
    axes[2].axis("off")
    fig.tight_layout()

    plt.savefig(f"test{cuda_kernel}.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def pretty_time(seconds):
    if not seconds:
        return f"0s"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    measures = (
        (hours, "h"),
        (minutes, "m"),
        (seconds, "s"),
    )
    return " ".join([f"{count}{unit}" for (count, unit) in measures if count])


k = 0
for layer_nr in nr_par_last_layer_list:

    class SNN(nn.Module):
        def __init__(self):
            super(SNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2)
            self.bn1 = nn.BatchNorm2d(8)
            self.lif1 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(8)
            self.lif2 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(8)
            self.lif3 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.fc1 = nn.Linear(5832, layer_nr)
            self.lif4 = LILinearCell(layer_nr, 4096)
            # self.lifbicycle = LILinearCell(layer_nr, 4096)
            # self.lifcar = LILinearCell(layer_nr, 4096)
            # self.lifbus = LILinearCell(layer_nr, 4096)
            # self.liftruck = LILinearCell(layer_nr, 4096)

            self.maxpool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x, mem_states):
            batch_size, C, W, H = x.shape
            x = (x != 0).float()  # Ensure binary data for this case

            mem1, mem2, mem3, mem4, mem5, mem6, mem7 = mem_states

            v1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(v1, mem1)

            v2 = self.dropout(self.bn2(self.conv2(self.maxpool(spk1))))
            spk2, mem2 = self.lif2(v2, mem2)

            v3 = self.dropout(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(v3, mem3)

            spk3_flat = spk3.view(batch_size, -1)
            v4 = self.dropout(self.fc1(spk3_flat))
            spk4, mem4 = self.lif4(v4, mem4)

            spk5, mem5 = self.lif4(v4, mem5)
            spk6, mem6 = self.lif4(v4, mem6)
            spk7, mem7 = self.lif4(v4, mem7)

            return torch.stack([spk4, spk5, spk6, spk7]), (
                mem1,
                mem2,
                mem3,
                mem4,
                mem5,
                mem6,
                mem7,
            )

    def loss_fn(output_frames, target_frames, step):
        mse_loss = 0
        for output_frame, target_frame in zip(output_frames, target_frames):
            mse_loss += loss_function(
                output_frame, target_frame * 1000
            )  # Multiplication due to the numbers being too small, should be fixed when creating the data

        # if print_image:
        #     save_current_result(output_frame, target_frame, frames, step)

        return mse_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print data of the model
    print(f"Total trainable parameters: {trainable_params}")
    trainable_weights = sum(
        p.numel() for name, p in model.named_parameters() if p.requires_grad and "weight" in name
    )

    print(f"Total trainable weights: {trainable_weights}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_type = "weights" if "weight" in name else "biases"
            print(f"Layer: {name} | Type: {param_type} | Number of Parameters: {param.numel()}")

    print("===================================================")
    num_epochs = 12
    frame_size = (64, 64)
    sequence_length = 75
    train_loss_list = []
    val_loss_list = []
    overlap = 25

    for epoch in range(num_epochs):
        train_loss = 0
        num_train_batches = 0
        model.train()
        start = time.time()

        for data_nr in range(5, 6):
            data = get_data(data_nr)

            if data is not None:
                train_data, val_data = data
                print(
                    f"\rRunning Epoch {epoch + 1} | File {data_nr} has been loaded. Training mode.",
                    end="",
                )
                train_loader = train_data

                for i, (frames, targets) in enumerate(train_loader):
                    # print("Frames")
                    # print(repr(np.array(frames).shape))
                    # print("Targets")
                    # print(repr(np.array(targets).shape))
                    mem_states = (None, None, None, None, None, None, None)

                    optimizer.zero_grad()

                    frames, targets = frames.to(device), targets.to(device)

                    loss = 0

                    for step in range(sequence_length):
                        input_frame = frames[:, step].unsqueeze(1)
                        output, mem_states = model(input_frame, mem_states)
                        # print(output.shape)

                        if step >= overlap:
                            final_output = output.view(16, 4, 64, 64)
                            # print(targets.shape)

                            loss += loss_fn(final_output, targets[:, :, step], step) / (
                                sequence_length - overlap
                            )  # only train on the last 50 frames

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    num_train_batches += 1

        model.eval()
        val_loss = 0
        num_test_batches = 0
        with torch.no_grad():
            for data_nr in range(5, 6):
                data = get_data(data_nr)

                if data is not None:
                    train_data, val_data = data
                    print(
                        f"\rRunning Epoch {epoch + 1} | File {data_nr} has been loaded. Validation mode.",
                        end="",
                    )
                    validation_loader = val_data

                    for i, (frames, targets) in enumerate(validation_loader):
                        mem_states = (None, None, None, None, None, None, None)

                        frames, targets = frames.to(device), targets.to(device)
                        loss = 0

                        for step in range(sequence_length):
                            input_frame = frames[:, step].unsqueeze(1)
                            output, mem_states = model(input_frame, mem_states)

                            final_output = output.view(16, 4, 64, 64)
                            loss += (
                                loss_fn(final_output, targets[:, :, step], step) / sequence_length
                            )

                        val_loss += loss.item()
                        num_test_batches += 1

        del data
        gc.collect()

        epoch_time = time.time() - start
        if i == 0:
            print(
                f"\x1b[0G\x1b[0KEpoch {epoch+1} | train loss: {train_loss/num_train_batches:.3f} | val loss {val_loss/num_test_batches:.3f} | finished in {pretty_time(epoch_time)}",
                end="\n",
            )
        else:
            print(
                f"\x1b[0F\x1b[0G\x1b[0KEpoch {epoch+1} | train loss: \x1b[1m{train_loss/num_train_batches:.3f}\x1b[22m | val loss \x1b[1m{val_loss/num_test_batches:.3f}\x1b[22m | finished in \x1b[1m{pretty_time(epoch_time)}\x1b[22m",
                end="\n",
            )

        train_loss_list.append(np.round(train_loss / num_train_batches, 4))
        val_loss_list.append(np.round(val_loss / num_test_batches, 4))

    k = save_data(model, train_loss_list, val_loss_list, tau_mem, today, epoch, k)
