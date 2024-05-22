import cv2
import torch
from transformers import CLIPProcessor, CLIPModel  
# from swin_transformer import SwinTransformer  # Assuming SwinTransformer is properly installed

class ProcessVideo:
    """
    Takes Video input and appropriately pre-processes it
    """
    def __init__(self):
        self.frames = []
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.action_model = SwinTransformer.from_pretrained("your_swin_transformer_model_path")
    
    def process(self, path: str):
        """
        Main function to process the video.
        Extract frames, detect actions, and encode segments.
        """
        self.extract_frames(path)
        segments = self.detect_actions()
        embeddings = self.encode_segments(segments)
        return segments, embeddings
    
    def extract_frames(self, input_video, interval=10):
        """
        Extracts every `interval` frames from a video and stores them.
        """
        cap = cv2.VideoCapture(input_video)
        count = 0
        success = True
        while success:
            success, frame = cap.read()
            if count % interval == 0 and success:
                self.frames.append(frame)
            count += 1
        cap.release()

    def detect_actions(self):
        """
        Detects actions in the extracted frames using the Swin Transformer model.
        Groups consecutive frames with the same action.
        """
        segments = []
        current_action = None
        start_time = 0
        start_frame_idx = 0

        for i, frame in enumerate(self.frames):
            inputs = self.clip_processor(images=frame, return_tensors="pt")
            with torch.no_grad():
                logits = self.action_model(**inputs).logits
                action = logits.argmax(dim=-1).item()

            if action == current_action:
                end_time = (i + 1) * 10  # Assuming 10 frames per second
            else:
                if current_action is not None:
                    segments.append({
                        "action": current_action,
                        "start_time": start_time,
                        "end_time": end_time,
                        "frames": self.frames[start_frame_idx:i]
                    })
                current_action = action
                start_time = i * 10
                start_frame_idx = i
                end_time = (i + 1) * 10
        
        # Add the last segment
        if current_action is not None:
            segments.append({
                "action": current_action,
                "start_time": start_time,
                "end_time": end_time,
                "frames": self.frames[start_frame_idx:]
            })
        
        return segments
    
    def encode_segments(self, segments):
        """
        Encodes the video segments into embeddings.
        """
        embeddings = []
        for segment in segments:
            frames = segment["frames"]
            inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            embeddings.append({
                "action": segment["action"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "embedding": outputs.mean(dim=0).cpu().numpy()  # Aggregate embeddings
            })
        return embeddings


# Example usage
video_path = "path_to_video.mp4"
video_processor = ProcessVideo()
segments, embeddings = video_processor.process(video_path)

# Print the results
for segment, embedding in zip(segments, embeddings):
    print(f"Action: {segment['action']}, Start: {segment['start_time']}, End: {segment['end_time']}")
    print(f"Embedding shape: {embedding['embedding'].shape}")