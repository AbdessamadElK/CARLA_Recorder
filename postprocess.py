from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import shutil
import os
from tqdm import tqdm

def get_indices_to_include(data_dirs:list):
    # Returns a list of frame numbers that are present in every data directory
    # so that additional frames (mostly the last or the first) in some data directories will be ignored
    sorted_dirs = data_dirs.copy()
    sorted_dirs.sort(key=lambda dir : len(list(dir.iterdir())))
    reference_dir = sorted_dirs[0]

    return [int(file.stem) for file in reference_dir.iterdir()]

def process_sequence(input_path:Path, output_path:Path):
    # Input directories
    events_dir = input_path / 'dvs'
    flow_dir = input_path / 'optical_flow'
    rgb_dir = input_path / 'rgb'
    segmentation_dir = input_path / 'semantic_segmentation'

    data_dirs = [events_dir, flow_dir, rgb_dir, segmentation_dir]

    # Output directories
    output_path = output_path / input_path.name
    output_path.mkdir(parents=True, exist_ok=True)
    
    rgb_destination = output_path / 'images'
    rgb_destination.mkdir(parents=True, exist_ok=True)

    seg_destination = output_path / 'segmentation'
    seg_destination.mkdir(parents=True, exist_ok=True)

    # Export processed data
    indices_to_include = sorted(get_indices_to_include(data_dirs))
    for i, idx in tqdm(enumerate(indices_to_include), desc=input_path.name):
        if i == 0:
            # skip the first frame (only used in events)
            continue
        try:
            # Save each two consecutive event streams in one file
            events0 = np.load(events_dir / "{}.npy".format(idx-1))
            events1 = np.load(events_dir / "{}.npy".format(idx))
            np.savez(output_path / "{:06d}".format(i-1), events_prev = events0, events_curr = events1)

            # Save flow files
            flow_16bit = imageio.imread(flow_dir / f"{idx}.png", format='PNG-FI')
            np.save(output_path / 'flow_{:06d}.npy'.format(i-1), flow_16bit)

            # Copy the rest of data
            shutil.copy(rgb_dir / "{}.png".format(idx), rgb_destination / "{:06d}.png".format(i-1))
            shutil.copy(segmentation_dir / "{}.png".format(idx), seg_destination / "{:06d}.png".format(i-1))
        
        except FileNotFoundError:
            print(f"Missing file : {idx}")
            continue

def process_dataset(source:Path, output_path:Path):
    print("Post processing the dataset {}...".format(source.name))
    sequences = [seq for seq in source.iterdir() if seq.is_dir()]

    for seq in sequences:
        process_sequence(seq, output_path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--source", "-s", type=str, help="Path to raw data")
    parser.add_argument("--destination", "-d", type=str, help="Path where to save the processed data")

    args = parser.parse_args()

    source = Path(args.source)
    assert source.is_dir()

    destination = Path(args.destination)

    process_dataset(source, destination)
