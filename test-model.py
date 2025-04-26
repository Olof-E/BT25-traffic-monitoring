from functools import partial
from matplotlib import animation
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import date
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set device
cuda_kernel = 0
sequence_length, overlap, batch_size = 75, 25, 12
torch.cuda.set_device(cuda_kernel)
num_inputs = 256 * 256
num_outputs = 64 * 64  # 4096
# tau_list = [120, 140, 160, 180, 200] #For trying different taus
layer_nr = 1000
w_decay = 1e-3  # 6.5e-3
lr = 9e-4
print_image = (
    False  # Set true to save current image for each loop, used to see progress during training
)
loss_function = nn.MSELoss()
tau_mem = 180


class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.lif1 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.lif2 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.lif3 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.fcperson = nn.Linear(1568, layer_nr)
        self.lifperson = LILinearCell(layer_nr, 4096)

        self.fccar = nn.Linear(1568, layer_nr)
        self.lifcar = LILinearCell(layer_nr, 4096)

        self.fcbus = nn.Linear(1568, layer_nr)
        self.lifbus = LILinearCell(layer_nr, 4096)

        self.fctruck = nn.Linear(1568, layer_nr)
        self.liftruck = LILinearCell(layer_nr, 4096)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, mem_states):
        batch_size, C, W, H = x.shape
        x = (x != 0).float()  # Ensure binary data for this case

        mem1, mem2, mem3, mem4, mem5, mem6, mem7 = mem_states

        v1 = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(v1, mem1)

        v2 = self.dropout(self.bn2(self.conv2(self.maxpool(spk1))))
        # v2 = self.dropout(self.bn2(self.conv2(spk1)))
        spk2, mem2 = self.lif2(v2, mem2)

        # v3 = self.dropout(self.bn3(self.conv3(spk2)))
        v3 = self.dropout(self.maxpool(self.bn3(self.conv3(spk2))))
        spk3, mem3 = self.lif3(v3, mem3)

        spk3_flat = spk3.view(batch_size, -1)
        v4 = self.dropout(self.fcperson(spk3_flat))
        spk4, mem4 = self.lifperson(v4, mem4)

        v5 = self.dropout(self.fccar(spk3_flat))
        spk5, mem5 = self.lifcar(v5, mem5)

        v6 = self.dropout(self.fcbus(spk3_flat))
        spk6, mem6 = self.lifbus(v6, mem6)

        v7 = self.dropout(self.fctruck(spk3_flat))
        spk7, mem7 = self.liftruck(v7, mem7)

        return (
            spk4,
            spk5,
            spk6,
            spk7,
            (
                mem1,
                mem2,
                mem3,
                mem4,
                mem5,
                mem6,
                mem7,
            ),
        )


def loss_fn(output_frame, target_frame, step, class_id):
    mse_loss = 0
    mse_loss += loss_function(
        output_frame, target_frame * 1000
    )  # Multiplication due to the numbers being too small, should be fixed when creating the data

    # if print_image:
    #     save_current_result(output_frame, target_frame, frames, step, class_id)

    return mse_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNN()


model.load_state_dict(
    torch.load("models/new-model-6.6492.pth", weights_only=True, map_location="cuda:0")
)
model = model.to(device)
model.eval()


data = torch.load(f"training_data/event_frames_16.pt")
# f"./w31/box2/2-07-31/events/event_frames_14.pt"
# f"./w31/box2/2-08-01/events/event_frames_4.pt"
# f"./w38/box3/3-09-27/events/event_frames_6.pt"


mem_states = (None, None, None, None, None, None, None)

fig, ax = plt.subplots(1, 5, figsize=(30, 10))  # , layout="compressed")


ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")
ax[3].axis("off")
ax[4].axis("off")


event_img = ax[0].imshow([[0]], cmap="gray")

output_img1 = ax[1].imshow(
    [[0]],
    cmap="magma",
    vmin=0,
    vmax=1,
)
output_img2 = ax[2].imshow(
    [[0]],
    cmap="magma",
    vmin=0,
    vmax=1,
)
output_img3 = ax[3].imshow(
    [[0]],
    cmap="magma",
    vmin=0,
    vmax=1,
)
output_img4 = ax[4].imshow(
    [[0]],
    cmap="magma",
    vmin=0,
    vmax=1,
)


# cbar = fig.colorbar(
#     output_img4,
# )


FRAMES = len(data)


def animate(step):
    global mem_states
    if step == FRAMES - 1:
        plt.close(fig)
    # gc.collect()
    # torch.cuda.empty_cache()
    with torch.no_grad():
        frame = torch.tensor(
            np.array(np.array([[data[step].to_dense()[160 : 160 + 256, 50 : 50 + 256]]])),
            device=device,
        )  # [180:480, 100:400]
        frame.to(device)

        with torch.amp.autocast(device_type="cuda"):
            output1, output2, output3, output4, mem_states = model(frame, mem_states)
        # print(output.shape)
        final_output1 = output1.view(1, 64, 64)
        final_output2 = output2.view(1, 64, 64)
        final_output3 = output3.view(1, 64, 64)
        final_output4 = output4.view(1, 64, 64)

        event_img.set_data(frame[0][0].cpu().detach().numpy())
        event_img.autoscale()

        maximum = np.max(
            [
                torch.max(final_output1[0]).item(),
                torch.max(final_output2[0]).item(),
                torch.max(final_output3[0]).item(),
                torch.max(final_output4[0]).item(),
            ]
        )

        output_img1.set(
            data=final_output1[0].cpu().detach().numpy(),
            clim=(
                2,
                max(2, maximum),
            ),
        )

        output_img2.set(
            data=final_output2[0].cpu().detach().numpy(),
            clim=(2, max(2, maximum)),
        )

        output_img3.set(
            data=final_output3[0].cpu().detach().numpy(),
            clim=(2, max(2, maximum)),
        )

        output_img4.set(
            data=final_output4[0].cpu().detach().numpy(),
            clim=(2, max(2, maximum)),
        )

    # frame = None
    # output = None
    # final_output = None
    # del frame, output, final_output
    return (
        event_img,
        output_img1,
        output_img2,
        output_img3,
        output_img4,
    )
    # fig.colorbar(output, ax=ax[1], location="right")


ani = animation.FuncAnimation(
    fig,
    partial(animate),
    frames=FRAMES,
    interval=1,
    repeat=False,
    blit=True,
)

# plt.show()

save_dir = "test-anim-multi-full-dataset-maybework2.mp4"  # os.path.join(os.path.dirname(sys.argv[0]), 'MyVideo.mp4')
print(f"Saving video to {save_dir}...")
video_writer = animation.FFMpegWriter(fps=90, bitrate=-1)
update_func = lambda _i, _n: progress_bar.update(1)
with tqdm(total=len(data), ncols=86, desc="Saving video") as progress_bar:
    ani.save(save_dir, writer=video_writer, dpi=100, progress_callback=update_func)
print("Video saved successfully.")
