def preprocess_video(video_path, target_size=(224, 224)):
    import cv2
    import numpy as np

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to the target size
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

    cap.release()
    # Convert frames to a numpy array
    return np.array(frames)

def extract_frames(video_path, frame_rate=1):
    import cv2
    import os

    # Create a directory to save frames
    frame_dir = os.path.join(os.path.dirname(video_path), 'frames')
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    success = True

    while success:
        success, image = cap.read()
        if count % frame_rate == 0 and success:
            frame_filename = os.path.join(frame_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, image)
        count += 1

    cap.release()