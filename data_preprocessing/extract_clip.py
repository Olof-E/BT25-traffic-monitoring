import cv2
import argparse
import os
from tqdm import tqdm

# Program to extract a portion of a video
# Might me able to use ffmpeg instead of this script tp levrage GPU acceleration and get better performance
# ffmpeg -hwaccel cuda -i videos/241201_1min_videos/w31.avi -ss 00:00:00 -t 00:10:00 -c:v h264_nvenc output_video_extract.avi

def get_cutout(input, length, frame_length, frame_number=0):
    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Does not seem to work for the large videos

    # Calculate the number of frames to extract
    if frame_length > 0:
        num_frames_to_extract = frame_length
    else:
        num_frames_to_extract = int(length * 60 * fps)

    # Ensure the number of frames to extract does not exceed the total frames
    # Does not seem to work for the large videos
    # if frame_number + num_frames_to_extract > total_frames:
    #    print(f"Error: The number of frames to extract ({num_frames_to_extract}) exceeds the total number of frames in the video ({total_frames}).")
    #    return

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Define the codec and create VideoWriter object to save the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Generate the output file name
    input_basename = os.path.basename(input)
    input_name, input_ext = os.path.splitext(input_basename)
    output_name = f"{input_name}_extract{input_ext}"

    out = cv2.VideoWriter(output_name, fourcc, fps, (frame_width, frame_height))

    print(f"Extracting {num_frames_to_extract} frames starting from frame {frame_number}")

    for i in tqdm(range(num_frames_to_extract), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Alternative frame check since CAP_PROP_FRAME_COUNT did not work for the large .avi files
    total_frames = i+1
    if num_frames_to_extract > total_frames:
        print(f"Error: the selected number of frames exceeds ({num_frames_to_extract}) the total number of frames in the video ({total_frames}).")
        print(f"Video size is now instead {total_frames} frames.")

    cap.release()
    out.release()
    print(f"Video extraction complete. Output saved to '{output_name}'")

parser = argparse.ArgumentParser(description="Extract a portion of a video.")
parser.add_argument("input", help="Input video file path.")
parser.add_argument("-l", type=int, default=10, metavar='Length (m)', help="Desired video length in minutes.")
parser.add_argument("-lf", type=int, default=0, metavar='Frame amount', help="Desired video length in number of frames.")
parser.add_argument("-f", type=int, default=0, metavar='Frame number', help="Starting frame number.")

args = parser.parse_args()

get_cutout(args.input, args.l, args.lf, args.f)