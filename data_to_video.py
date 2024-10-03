from pathlib import Path

import imageio.v2 as imageio
import skvideo.io as io

import numpy as np
from tqdm import tqdm


class VideoDataGenerator():
    def __init__(self, data_dir:Path, output_dir:Path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sequences = [seq for seq in self.data_dir.iterdir() if seq.is_dir()]

    def generate(self):
        
        for seq in self.sequences:
            rgb_dir = seq / 'rgb'
            writer = io.FFmpegWriter(self.output_dir / f"./{seq.name}.mp4", outputdict={"-pix_fmt": "yuv420p"})
            frame_numbers = [int(path.stem) for path in rgb_dir.iterdir()]

            for num in tqdm(sorted(frame_numbers), desc=seq.name):
                try:
                    rgb = rgb_dir / '{}.png'.format(num)
                    flow = seq / 'visualizations/optical_flow/vis_{}.png'.format(num)
                    events = seq / 'visualizations/dvs/{}.png'.format(num)
                    segmentation = seq / 'visualizations/semantic_segmentation/{}.png'.format(num)
                    
                    frame_items = [rgb, flow, events, segmentation]
                    frame_items = [imageio.imread(item)[:,:,:3] for item in frame_items]

                    row1 = np.hstack(frame_items[:2])
                    row2 = np.hstack(frame_items[-2:])
                    frame = np.vstack([row1, row2])
                
                except FileNotFoundError:
                    continue

                writer.writeFrame(frame.astype('uint8'))

            writer.close()

    def generate_superposed(self):
         for seq in self.sequences:
            rgb_dir = seq / 'rgb'
            writer = io.FFmpegWriter(self.output_dir / f"./{seq.name}_s.mp4", outputdict={"-pix_fmt": "yuv420p"})
            frame_numbers = [int(path.stem) for path in rgb_dir.iterdir()]

            for num in tqdm(sorted(frame_numbers), desc=seq.name):
                try:
                    rgb = rgb_dir / '{}.png'.format(num)
                    flow = seq / 'visualizations/optical_flow/vis_{}.png'.format(num)
                    events = seq / 'visualizations/dvs/{}.png'.format(num)
                    segmentation = seq / 'visualizations/semantic_segmentation/{}.png'.format(num)
                    
                    rgb_img = imageio.imread(rgb)[:,:,:3]
                    flow_img = imageio.imread(flow)[:,:,:3]
                    events_img = imageio.imread(events)[:,:,:3]
                    segmentation_img = imageio.imread(segmentation)[:,:,:3]

                    frame_items = [0.4*rgb_img + 0.6 * events_img,
                                   flow_img,
                                   events_img,
                                   0.4*rgb_img + 0.6 * segmentation_img]

                    row1 = np.hstack(frame_items[:2])
                    row2 = np.hstack(frame_items[-2:])
                    frame = np.vstack([row1, row2])
                
                except FileNotFoundError:
                    continue

                writer.writeFrame(frame.astype('uint8'))

            writer.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="Dataset directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory")

    args = parser.parse_args()

    input = Path(args.input)
    assert input.is_dir()

    output = Path(args.output)

    generator = VideoDataGenerator(input, output)

    print(f"Generating videos from {input}")
    # generator.generate()
    generator.generate_superposed()
    print(f"Done generating.. videos are saved in : {output}")
