from pathlib import Path

import imageio.v2 as imageio
import skvideo.io as io

import numpy as np
from tqdm import tqdm

DATA_SOURCE = Path("./out")

VIS_DIR = DATA_SOURCE / 'visualizations'
RGB_DIR = DATA_SOURCE / 'rgb'

visual_dirs = list(VIS_DIR.iterdir())
visual_dirs.append(RGB_DIR)

all_items = [sorted(dir.iterdir()) for dir in visual_dirs]

writer = io.FFmpegWriter("./video.mp4", outputdict={"-pix_fmt": "yuv420p"})

frame_numbers = [int(path.stem) for path in RGB_DIR.iterdir()]

for num in tqdm(sorted(frame_numbers)):
    try:
        rgb = RGB_DIR / f"{num}.png"
        flow = VIS_DIR / 'optical_flow' / f"vis_{num}.png"
        events = VIS_DIR / 'dvs' / f"{num}.png"
        segmentation = VIS_DIR / 'semantic_segmentation' / f"{num}.png"
        
        frame_items = [rgb, flow, events, segmentation]
        frame_items = [imageio.imread(item)[:,:,:3] for item in frame_items]

        row1 = np.hstack(frame_items[:2])
        row2 = np.hstack(frame_items[-2:])
        frame = np.vstack([row1, row2])


    except FileNotFoundError:
        continue
    
    writer.writeFrame(frame.astype('uint8'))

writer.close()

