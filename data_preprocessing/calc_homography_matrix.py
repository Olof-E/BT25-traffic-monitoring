import os
from threading import Thread
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import patches
from skimage import color, exposure, feature, filters, transform, util


visualize = False

img = 0


def process_clip(event_path, normal_path):
    global img
    torch.cuda.set_device(device=0)
    model_matches = 0
    total_error = 0
    event_data = torch.load(event_path)

    normal_cap = cv2.VideoCapture(normal_path)

    matches = []
    for i in tqdm(
        range(0, 5000, 75),
        ncols=86,
        mininterval=0.75,
        leave=False,
    ):
        normal_cap.set(1, 0 + i)
        img_left2 = event_data[i + 1].to_dense()  # Read the Event frame
        _, img_right = normal_cap.read()  # Read the Normal frame
        # _, _ = normal_cap.read()  # Read the Normal frame
        _, img_right2 = normal_cap.read()  # Read the Normal frame

        img_left = exposure.adjust_gamma(filters.gaussian(img_left2, 12), 8)

        img_right = exposure.adjust_gamma(
            filters.gaussian(
                util.compare_images(
                    color.rgb2gray(img_right), color.rgb2gray(img_right2), method="diff"
                ),
                12,
            ),
            8,
        )

        img_left = exposure.rescale_intensity(img_left)
        img_right = exposure.rescale_intensity(img_right)

        blobs1 = feature.blob_doh(
            img_left, min_sigma=16, max_sigma=32, num_sigma=3, threshold_rel=0.45
        )
        blobs2 = feature.blob_doh(
            img_right, min_sigma=16, max_sigma=32, num_sigma=3, threshold_rel=0.55
        )

        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))

            axes[0].imshow(img_left, cmap="magma")
            axes[1].imshow(img_right, cmap="magma")

            for y1, x1, r1 in blobs1:
                cirk1 = patches.Circle((x1, y1), radius=r1, linewidth=2, edgecolor="r", fill=False)

                axes[0].add_patch(cirk1)

            for y2, x2, r2 in blobs2:

                cirk2 = patches.Circle((x2, y2), radius=r2, linewidth=2, edgecolor="r", fill=False)

                axes[1].add_patch(cirk2)
        for y1, x1, r1 in blobs1:
            best_match = 0.15
            best_index = -1
            for i, (y2, x2, r2) in enumerate(blobs2):
                error = math.dist((x1 / 640, y1 / 480), (x2 / 736, y2 / 460))
                if error < best_match:
                    best_match = error
                    best_index = i

            if best_index != -1:
                for i, (y2, x2, r2) in enumerate(blobs1):
                    error = math.dist(
                        (x2 / 640, y2 / 480),
                        (blobs2[best_index][1] / 736, blobs2[best_index][0] / 460),
                    )
                    if error < best_match:
                        break
                else:
                    model_matches += 1
                    total_error += best_match
                    matches.append([[x1, y1], [blobs2[best_index][1], blobs2[best_index][0]]])
                    if visualize:
                        conn = patches.ConnectionPatch(
                            xyA=(x1, y1),
                            xyB=(blobs2[best_index][1], blobs2[best_index][0]),
                            coordsA="data",
                            coordsB="data",
                            axesA=axes[0],
                            axesB=axes[1],
                            color="lightgreen",
                            linewidth=3,
                        )

                        fig.add_artist(conn)

                        cirk1 = patches.Circle((x1, y1), radius=2, color="lightgreen", fill=True)
                        cirk2 = patches.Circle(
                            (blobs2[best_index][1], blobs2[best_index][0]),
                            radius=2,
                            color="lightgreen",
                            fill=True,
                        )

                        axes[0].add_patch(cirk1)
                        axes[1].add_patch(cirk2)

        if visualize:
            axes[0].axis("off")
            axes[1].axis("off")
            fig.tight_layout()
            plt.show()
            # fig.savefig(f"clips/test/{img}.png", bbox_inches="tight", pad_inches=0.1)
            # img += 1

            # plt.close(fig)

    normal_cap.release()

    matches = np.array(matches)
    model = transform.SimilarityTransform()
    weight = 0.000001
    if len(matches) > 32:
        model.estimate(matches[:, 0], matches[:, 1])
        weight = 0.5 * model_matches + 0.5 * (1 - total_error)

    return model, weight, total_error


def get_targets(directory, target_length, file_nr):
    target_tensors = np.empty(target_length, dtype=object)

    for filename in sorted(filter(lambda f: f.endswith(".txt"), os.listdir(directory))):
        with open(os.path.join(directory, filename)) as file:
            filename = filename.split("_")
            if int(filename[1]) != file_nr:
                continue
            target_data = [
                float(value)
                for line in file
                if line.strip().split()[0] in str(("0", "2", "5", "7"))
                for value in line.split()[0:5]
            ]
            target_tensor = torch.tensor(target_data, dtype=torch.float16).view(-1, 5)
            target_tensors[int(filename[2].rsplit(".", maxsplit=1)[0]) - 1] = target_tensor
    return target_tensors


def calc_matrix(event_folder, normal_folder, start_clip, end_clip, results, res_weights):

    models = []
    weights = []
    for i in range(start_clip, end_clip + 1):
        model, weight, total_error = process_clip(
            f"{event_folder}event_frames_{i}.pt", f"{normal_folder}_{i}.mp4"
        )
        models.append(model.params)
        weights.append(weight)

    models = np.array(models)
    weights = np.array(weights)

    weights = weights / weights.sum()

    model = transform.SimilarityTransform()
    model.params = np.average(
        models,
        axis=0,
        weights=weights,
    )

    results.append(model.params)
    res_weights.append(1 - total_error)


mod_weights = []
models = []

curr_dir = "w31/box2/2-07-31"  # "w35/box1/1-09-04"

# "w31/box2/2-07-31"
# "w35/box1/1-09-04"

if not visualize:

    t1 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 22, 28, models, mod_weights],
    )
    t1.start()

    t2 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 16, 21, models, mod_weights],
    )
    t2.start()

    t3 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 10, 15, models, mod_weights],
    )
    t3.start()

calc_matrix(f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 5, 9, models, mod_weights)

if not visualize:
    t1.join()
    t2.join()
    t3.join()


model = transform.SimilarityTransform()


mod_weights = np.array(mod_weights)

model.params = np.average(
    models,
    axis=0,
    weights=[1 / len(models)] * len(models),
)

print(repr(model))

# model = transform.SimilarityTransform()
# model.params = np.array(
#     [
#         [8.32694435e-01, -1.24637729e-02, 1.32369730e02],
#         [1.24637729e-02, 8.32694435e-01, 6.27525193e01],
#         [0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )

event_data = torch.load(f"../{curr_dir}/events/event_frames_6.pt")

normal_cap = cv2.VideoCapture(f"../{curr_dir}/normal/_6.mp4")


target_tensors = get_targets(f"../{curr_dir}/track/labels/", 5400, 6)

# fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# axes[0].imshow(img_left, cmap="gray")
# axes[1].imshow(img_right, cmap="brg")

out = cv2.VideoWriter(
    filename=f"homography-vis.mp4",
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
    fps=90,
    frameSize=(
        640 + 736,
        480,
    ),
    isColor=False,
)


for frame_idx in tqdm(
    range(int(len(event_data))),
    desc=f"annotating frames",
    ncols=86,
    mininterval=0.25,
):
    frame = event_data[frame_idx].to_dense() * 255
    ret, img_right = normal_cap.read()
    combined_frame = np.concatenate(
        (
            frame,
            np.concatenate((cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY), np.full((20, 736), 255))),
        ),
        axis=1,
    )
    for t in range(len(target_tensors[frame_idx])):
        # ===========================
        # Event Image
        # ===========================

        center_x, center_y = target_tensors[frame_idx][t][1:3].numpy() * [736, 460]
        w, h = target_tensors[frame_idx][t][3:].numpy() * [736, 460]

        x_min, y_min = np.array([center_x, center_y]) - [w / 2, h / 2]
        x_max, y_max = np.array([center_x, center_y]) + [w / 2, h / 2]

        x_min, y_min = model._apply_mat((x_min, y_min), model.inverse.params)[0]
        x_max, y_max = model._apply_mat((x_max, y_max), model.inverse.params)[0]

        center_x, center_y = model._apply_mat((center_x, center_y), model.inverse.params)[0]

        cv2.rectangle(
            img=combined_frame,
            pt1=(int(x_min), int(y_min)),
            pt2=(int(x_max), int(y_max)),
            color=(255, 0, 0),
            thickness=2,
        )
        # rect = patches.Rectangle(
        #     (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor="r", facecolor="none"
        # )
        # cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
        # axes[0].add_patch(rect)
        # axes[0].add_patch(cirk)

        # ===========================
        # Normal Image
        # ===========================
        center_x, center_y = target_tensors[frame_idx][t][1:3].numpy() * [736, 460]
        w, h = target_tensors[frame_idx][t][3:].numpy() * [736, 460]

        x_min, y_min = np.array([center_x, center_y]) - [w / 2, h / 2]
        x_max, y_max = np.array([center_x, center_y]) + [w / 2, h / 2]
        cv2.rectangle(
            img=combined_frame,
            pt1=(int(x_min + 640), int(y_min)),
            pt2=(int(x_max + 640), int(y_max)),
            color=(255, 0, 0),
            thickness=2,
        )

        # rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none")
        # cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
        # axes[1].add_patch(rect)
        # axes[1].add_patch(cirk)
    out.write(np.uint8(combined_frame))


# plt.tight_layout()
# plt.show()

out.release()
normal_cap.release()
