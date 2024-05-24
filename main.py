import cv2
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import streamlit as st

def extract_frames(video_path, output_dir, interval=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        success, frame = cap.read()
        if count % interval == 0 and success:
            cv2.imwrite(os.path.join(output_dir, f"frame_{count:05d}.jpg"), frame)
        count += 1
    cap.release()

# Usage
video_path = 'path/to/your/video.mp4'
output_dir = 'path/to/extracted/frames'
extract_frames(video_path, output_dir, interval=30)  # Extract a frame every 30 frames

# analyze frames

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def analyze_frame(frame_path):
    image = Image.open(frame_path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

# Usage
frame_path = 'path/to/extracted/frames/frame_00030.jpg'
outputs = analyze_frame(frame_path)

#segment video based on timestampes
# instead of actually segmenting the video, record the timestamps in a map

def segment_video(frame_dir, model, processor, interval=30):
    frames = sorted(os.listdir(frame_dir))
    segments = []
    for i, frame in enumerate(frames[::interval]):
        frame_path = os.path.join(frame_dir, frame)
        outputs = analyze_frame(frame_path)
        segments.append((i*interval, outputs))
    return segments

# Usage
segments = segment_video(output_dir, model, processor)
print(segments)


# aggregate similar/same segments together
# if cosine similarity > 80%

def aggregate_segments(segments):
    # Implement aggregation logic
    aggregated_segments = []
    current_segment = []
    for segment in segments:
        if not current_segment:
            current_segment.append(segment)
        else:
            # Assume simple heuristic for segment change
            if segment[1] != current_segment[-1][1]:
                aggregated_segments.append(current_segment)
                current_segment = [segment]
            else:
                current_segment.append(segment)
    if current_segment:
        aggregated_segments.append(current_segment)
    return aggregated_segments

# Usage
aggregated_segments = aggregate_segments(segments)
for segment in aggregated_segments:
    print(f"Segment from frame {segment[0][0]} to {segment[-1][0]}")
    

#UI option, will probably make an API with flask for it
st.title("Video Segmentation")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_file is not None:
    video_path = f"uploads/{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.video(video_path)
    
    extract_frames(video_path, output_dir)
    segments = segment_video(output_dir, model, processor)
    aggregated_segments = aggregate_segments(segments)
    st.write("Segments identified:")
    for segment in aggregated_segments:
        st.write(f"Segment from frame {segment[0][0]} to {segment[-1][0]}")