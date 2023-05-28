import glob
import moviepy.editor as mp
from yt_dlp import YoutubeDL

# Reading video links from "video_links.txt" file
with open('data/video_links.txt') as f:
    links = f.readlines()
    links = [x.strip() for x in links]

'''Download videos from given youtube urls with "480p" resolution.
Audio is extracted by converting downloaded videos to "wav" format.
All videos, audios are stored in "data" folder.'''
with YoutubeDL({'height': 480, 'paths': {'home': './data/'}}) as ydl:
    ydl.download(links)

types = ('data/*.webm', 'data/*.mkv', 'data/*.mp4', 'data/*.avi') # the tuple of file types
videos = []
for files in types:
    videos.extend(glob.glob(files))
for vid_path in videos:
    video = mp.VideoFileClip(vid_path)
    video.audio.write_audiofile('.'.join(vid_path.split('.')[:-1])+'.wav')  
