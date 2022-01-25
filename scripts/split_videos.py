from pathlib import Path
from importlib_metadata import itertools
import skvideo.io
import os
import numpy as np
import skvideo.datasets
from tqdm import tqdm
import gc
import sys
from copy import deepcopy as d
import itertools

VIDEO_LENGTH = 45
OVERLAP = 5

VIDEO_DIR = Path()
files = VIDEO_DIR.glob("*.avi")

current_video_frames = []
next_video_frames = []

for file in files:
    video_reader = skvideo.io.FFmpegReader(str(file))
    videometadata = skvideo.io.ffprobe(file)
    frame_rate = videometadata['video']['@avg_frame_rate'] 
    frame_rate =  float(frame_rate.split("/")[0]) / float(frame_rate.split("/")[1]) 
    if not (VIDEO_DIR/file).is_dir():
        (VIDEO_DIR/file).with_suffix('').mkdir(parents=True, exist_ok=True)
    folder_path = (VIDEO_DIR/file).with_suffix('')
    batch_length = int(VIDEO_LENGTH*frame_rate)
    overlap = int(OVERLAP*frame_rate)
    gen = video_reader.nextFrame()

    for i in tqdm(range(0,video_reader.getShape()[0]-batch_length, batch_length-overlap)):
        gen2 = None
        #print(i,i+batch_length-overlap,video_reader.getShape())
        video_save_path = folder_path.resolve()/f"{int(i/batch_length)}.avi"
        fps = frame_rate
        writer = skvideo.io.FFmpegWriter(video_save_path, 
                                inputdict={'-r': str(fps)},
                                outputdict={'-r': str(fps), '-c:v': 'libx264', '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'})
        for frame_idx in range(batch_length):
            if frame_idx >= batch_length-overlap: 
                if gen2 == None:
                    gen, gen2 = itertools.tee(gen)
                temp_arr = next(gen2)
                writer.writeFrame(temp_arr)

            else:
                writer.writeFrame(next(gen))

        writer.close()
        del writer

        gc.collect()

    del video_reader

    gc.collect()
            
        


    