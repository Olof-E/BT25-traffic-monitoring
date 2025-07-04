<h1 align="center">
  Event Cameras and Spiking Neural Networks for Tracking and Classification in Traffic Monitoring
</h1>

> [!IMPORTANT]
> <h2 align="center"> README is not complete and up to date. Some Files in the source code also needs to be cleaned up from unecessary commented out code blocks. This will be done in a little while. </h2>


This repo contains the code for the Bachelor thesis work in the course IA150X with the title "_Event Cameras and Spiking Neural Networks for Tracking and Classification in Traffic Monitoring_". This work is also a continuation made to a previous Master thesis work, read the [Acknowledgements](#acknowledgements) section for more information.

# Acknowledgements

The code in this repository is an extended and modified version of the original code that was made by [**Emma Hagrot**](https://github.com/emmahagrot) and the original code can be found in this [**Repo**](https://github.com/emmahagrot/MT24-SNNs-for-Traffic-Observation).

# Preprocessing

![Preprocessing pipeline visualiztion](/UML_Chart_Data.png)


The raw data consists of both a normal camera recording together with an event representation captured with an Event camera.

**Outline of the preprocessing**

- Downscale the frame-based video
- Cut the video into smaller clips (1min)
- Bin the events into ~10ms frames
- Generate labels from the normal frames (using [YOLOv12](https://github.com/ultralytics/ultralytics))
- Transfer the labels using homography
- Transform bounding boxes to Gaussian “blobs”

The different stages of the preprocesing pipeline listed above are separated into individual files:

`cut_video.py`
Crops the given source video according to the given (Width, Height) and the top left corner (x_start, y_start) and also cuts the source into shorter clips for easier processing. The file can be run as follows:

<!-- ```shell
py cut_video.py path/to/footage path/to/output_dir 300
```

Where the required arguments are the `input_file`, `output_dir`, and the desired `length` of the produced clips in seconds. The default length is 5 minutes. -->

`create_event_frames.py`
This filame bins the events into frames. The events during a specific time interval are accumulated into frames, the time interval decided by the timestps.csv provided from the data recordings.

To generate the labels, [YOLOv11](https://github.com/ultralytics/ultralytics) is used and can be executed using the following command:

<!-- ```shell
yolo track model="yolo11n.pt" source="[path/to/footage]" conf=0.25, iou=0.5 project="yolo/results/" save_txt=true device="cuda:[deviceID]"
``` -->

where `[path/to/footage]` should be replaced with the filepath to the source footage that you want to generate the labels and bounding box data for and `[deviceID]` should be replaced with the ID of the gpu to be used for processing. If needed, the processing can be done on the cpu in which case `cuda:[deviceID]` can either be replaced by `cpu` or the entire `device` argument may be omitted.

`transfer_labels.py`
Manually check the diff between the event file and the video file and add that offset before transferring; it’s possible to see the result by using the plotting function.

Choose how much information you want to provide your target. You should include bounding box coordinates and target classification/ID.

# Network Architecture

[INSERT FINAL MODEL NETWORK IMAGE]

# Requirements

The main external packages and libraries needed are the following:

- ### **PyTorch**

  [PyTorch]() is

- ### **Norse**

  [Norse]() is

- ### **YOLOv12**

  [YOLOv12](https://github.com/ultralytics/ultralytics) is required since it is used in the preprocessing pipeline to generate the labels used to label the event frames. However, if the required training data has already been generated then YOLO is no longer required.

All required dependencies needed to run all aspects of the code are listed in the `requirements.txt` file and what verisions where used during the course of this thesis.

# Installation

# Usage

<!-- gcc -fPIC -shared -o event-reader.dll event_streamer.c -->



<!-- yolo track model=../../../yolo/yolo11l.pt source="./normal" conf=0.3, iou=0.7 project="./" save_txt=True device="cuda:0" batch=128 half verbose=True stream_buffer=False augment=True agnostic_nms=True -->