import os
from threading import Thread
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import color, exposure, filters, transform, util


from skimage.feature import match_descriptors, plot_matched_features
from skimage.measure import ransac
import matplotlib.pyplot as plt


from skimage.feature import (
    match_descriptors,
    corner_peaks,
    plot_matched_features,
    BRIEF,
)
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

visualize = False


def process_clip(event_path, normal_path):
    torch.cuda.set_device(device=0)
    event_data = torch.load(event_path)

    normal_cap = cv2.VideoCapture(normal_path)

    extractor = BRIEF()

    for i in tqdm(
        range(0, 5000, 150),
        ncols=86,
        mininterval=0.75,
        leave=False,
    ):
        for j in range(2):
            normal_cap.set(1, i + 10 * j)
            img_left2 = (
                event_data[i + 10 * j + 1].to_dense() + event_data[i + 10 * j + 2].to_dense()
            )  # Read the Event frame
            _, img_right = normal_cap.read()  # Read the Normal frame
            _ = normal_cap.grab()  # Read the Normal frame
            _ = normal_cap.grab()  # Read the Normal frame
            _ = normal_cap.grab()  # Read the Normal frame

            _, img_right2 = normal_cap.read()  # Read the Normal frame

            img_left = filters.gaussian(img_left2, 0.8)

            img_right = filters.gaussian(
                util.compare_images(
                    color.rgb2gray(img_right), color.rgb2gray(img_right2), method="diff"
                ),
                0.6,
            )

            img_left = exposure.rescale_intensity(img_left)
            img_right = exposure.rescale_intensity(img_right)

            keypoints1 = corner_peaks(img_left, min_distance=2, threshold_rel=0.1)
            keypoints2 = corner_peaks(img_right, min_distance=2, threshold_rel=0.1)

            extractor.extract(img_left, keypoints1)
            keypoints1 = keypoints1[extractor.mask]
            descriptors1 = extractor.descriptors

            extractor.extract(img_right, keypoints2)
            keypoints2 = keypoints2[extractor.mask]
            descriptors2 = extractor.descriptors

            try:
                matches12 = match_descriptors(
                    descriptors1, descriptors2, cross_check=True, max_distance=2, max_ratio=0.8
                )

                # fig, ax = plt.subplots(nrows=1, ncols=1)

                # plot_matched_features(
                #     img_left,
                #     img_right,
                #     keypoints0=keypoints1,
                #     keypoints1=keypoints2,
                #     matches=matches12,
                #     ax=ax,
                #     only_matches=True,
                # )
                # plt.show()

                # tqdm.write(f"matches found {matches12.shape[0]}")

                new_model, inliers = ransac(
                    (keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]]),
                    transform.SimilarityTransform,
                    min_samples=8,
                    residual_threshold=0.6,
                    max_trials=1500,
                )

                if inliers is not None and inliers.sum() >= 2:

                    # fig, ax = plt.subplots(nrows=1, ncols=1)

                    # plot_matched_features(
                    #     img_left,
                    #     img_right,
                    #     keypoints0=keypoints1,
                    #     keypoints1=keypoints2,
                    #     matches=matches12[inliers],
                    #     ax=ax,
                    #     only_matches=True,
                    # )
                    # plt.show()

                    normal_cap.release()
                    return new_model

            except:
                continue

    normal_cap.release()

    return None


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


def calc_matrix(event_folder, normal_folder, start_clip, end_clip, results):

    models = []
    for i in range(start_clip, end_clip + 1):
        model = process_clip(f"{event_folder}event_frames_{i}.pt", f"{normal_folder}_{i}.mp4")
        if model is not None:
            models.append(model.params)

    models = np.array(models)

    model = transform.SimilarityTransform()
    model.params = np.average(
        models,
        axis=0,
    )
    print(repr(model))

    results.append(model.params)


models = []

curr_dir = "w31/box2/2-07-31"  # "w35/box1/1-09-04"

# "w31/box2/2-07-31"
# "w35/box1/1-09-04"

if not visualize:

    t1 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 22, 25, models],
    )
    t1.start()

    t2 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 16, 19, models],
    )
    t2.start()

    t3 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 10, 14, models],
    )
    t3.start()

calc_matrix(f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 5, 9, models)

if not visualize:
    t1.join()
    t2.join()
    t3.join()


model = transform.SimilarityTransform()


model.params = np.average(models, axis=0)

print(repr(model))

event_data = torch.load(f"../{curr_dir}/events/event_frames_6.pt")

normal_cap = cv2.VideoCapture(f"../{curr_dir}/normal/_6.mp4")


target_tensors = get_targets(f"../{curr_dir}/track/labels/", 5400, 6)


out = cv2.VideoWriter(
    filename=f"homography-vis2.mp4",
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
    fps=90,
    frameSize=(
        640 + 736,
        480,
    ),
    isColor=False,
)


for frame_idx in tqdm(
    range(int(len(event_data)) // 3),
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

        y_min, x_min = model._apply_mat((y_min, x_min), model.inverse.params)[0]
        y_max, x_max = model._apply_mat((y_max, x_max), model.inverse.params)[0]

        center_y, center_x = model._apply_mat((center_y, center_x), model.inverse.params)[0]

        cv2.rectangle(
            img=combined_frame,
            pt1=(int(x_min), int(y_min)),
            pt2=(int(x_max), int(y_max)),
            color=(255, 0, 0),
            thickness=2,
        )

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

    out.write(np.uint8(combined_frame))


out.release()
normal_cap.release()
