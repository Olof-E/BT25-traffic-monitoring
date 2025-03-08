# SNN and Event cameras for traffic monitoring
This repo contains the code for the Bachelor thesis work in the course IA150X with the title "*Event Cameras and Spiking Neural Networks for Tracking and Classification in Traffic Monitoring*". This work is also an extension made to a previous Master thesis work, read the [Acknowledgements](#acknowledgements) section for more information.


# Acknowledgements
The code in this repository is an extended and modified version of the original code that was made by [**Emma Hagrot**](https://github.com/emmahagrot) and the original code can be found in this [**Repo**](https://github.com/emmahagrot/MT24-SNNs-for-Traffic-Observation). 


# Preprocessing
**Outline of the preprocessing**
- Crop a smaller part of the whole frame-based video
- Cut the video into smaller clips (1min/5min)
- Get the event file into a streamable format
- Bin the events into ~10ms frames 
- Transfer the labels using homography
- Transform bounding boxes to Gaussian “blobs”

``cut_out_video.py``
This file is for cutting out a certain part of the frame. The idea is not to process the whole frame since it requires more space, which also helps YOLO run. The cutout size is adjusted case by case, depending on what part of the video you want to cut out. The same thing is true for the start of x and y. 

``cut_video.py``
Divide the video into shorter clips. Provide the input file, output directory and desired length of the file. I originally went with 5 minutes, so 60 * 5, but maybe 1 minute would be better for smaller file sizes.

``create_event_frames.py``
This file bins the events into frames. The events during a specific time interval are accumulated into frames, the time interval decided by the timestamps.csv provided from the data recordings. 

``transfer_labels.py``
Manually check the diff between the event file and the video file and add that offset before transferring; it’s possible to see the result by using the plotting function.

Choose how much information you want to provide your target. You should include bounding box coordinates and target classification/ID.

# Network Architecture
[INSERT FINAL MODEL NETWORK IMAGE]

# Prerequsites
- PyTorch
- Norse
- YOLOv8



*model_version*
The four different architectures tried in the thesis

*data_loading.py*
Loads the data into sequences 

*SNN_final_model.py*
The model used for the final tests 
