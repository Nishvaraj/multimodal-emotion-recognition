class VideoLoader:
    def __init__(self, video_path):
        self.video_path = video_path

    def load_video(self):
        # Implement video loading logic here
        pass

    def preprocess_video(self, video):
        # Implement video preprocessing logic here
        pass

    def get_video_data(self):
        video = self.load_video()
        return self.preprocess_video(video)