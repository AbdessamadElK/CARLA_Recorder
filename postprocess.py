from pathlib import Path
import numpy as np
import shutil
import os

def process_sequence(input_path:Path, output_path:Path):
    data_dirs = [dir for dir in input_path.iterdir() if dir.name != "visualizations"]
    vis_dir = input_path / 'visualizations'

    rgb_dir = input_path / 'rgb'

    # Remove the files that are present in some data directories and not in others
    # mostly the first or the last frame
    ref_dir = get_reference_dir(data_dirs)
    for data_dir in data_dirs:
        remove_extras(data_dir, ref_dir)

    for visual_dir in vis_dir.iterdir():
        remove_extras(visual_dir, ref_dir)

    # Assemble each two event streams in a single npz file
    event_files = (data_dir/'dvs').glob("*.npy")
    


    pass

def get_reference_dir(data_dirs:list):
    sorted_dirs = data_dirs.copy()
    sorted_dirs.sort(key=lambda dir : len(list(dir.iterdir())))
    return sorted_dirs[0]


def remove_extras(target_dir:Path, reference_dir:Path):
    # Removes files from target dir that are not present in the reference dir
    # This removal is only based on the frame numbers (x.npy is the same as x.png)
    if target_dir == reference_dir:
        return
    
    indices_to_include = [int(file.stem) for file in reference_dir.iterdir()]
    for target_file in target_dir.iterdir():
        if not int(target_file.stem) in indices_to_include:
            os.remove(target_file)
    
    # return indices_to_include

if __name__ == "__main__":
    
    pass