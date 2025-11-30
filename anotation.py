from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector

video_path = "resultat_10s2.mp4"

video = open_video(video_path)

scene_manager_cuts = SceneManager()
scene_manager_cuts.add_detector(ContentDetector(threshold=30))
scene_manager_cuts.detect_scenes(video)

cuts = scene_manager_cuts.get_scene_list()

video.reset()
scene_manager_gradual = SceneManager()
scene_manager_gradual.add_detector(ThresholdDetector(min_scene_len=5))
scene_manager_gradual.detect_scenes(video)

graduals = scene_manager_gradual.get_scene_list()

annotations = []

for start, end in cuts:
    annotations.append({
        "start_frame": start.get_frames(),
        "end_frame": end.get_frames(),
        "start_time": start.get_timecode(),
        "end_time": end.get_timecode(),
        "transition_type": "cut"
    })

for start, end in graduals:
    annotations.append({
        "start_frame": start.get_frames(),
        "end_frame": end.get_frames(),
        "start_time": start.get_timecode(),
        "end_time": end.get_timecode(),
        "transition_type": "gradual"
    })

annotations.sort(key=lambda x: x["start_frame"])

for ann in annotations:
    print(ann)