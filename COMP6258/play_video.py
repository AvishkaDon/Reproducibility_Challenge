import cv2
import numpy as np

video_frames_path = '/Users/aahd/Library/CloudStorage/OneDrive-UniversityofSouthampton/year_4/Deep Learning/cw/DL_MindVideo/wen_2018/video_test_256_3hz.npy'
video_frames = np.load(video_frames_path)

print("Shape of video frames:", video_frames.shape)

for video in range(video_frames.shape[0]):
    for set_index in range(video_frames.shape[1]):
        for frame_index in range(video_frames.shape[2]):
            current_frame = video_frames[video, set_index, frame_index]
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Video Playback', current_frame)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                cv2.destroyAllWindows()  
                exit()  

cv2.destroyAllWindows()
