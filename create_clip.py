from moviepy import VideoFileClip

clip = VideoFileClip("DataSet/drone_mouvement_canyon.mp4")

clip_coupe = clip.subclipped(32, 42)

clip_coupe.write_videofile("resultat_10s2.mp4", codec="libx264", audio_codec="aac")

clip.close()