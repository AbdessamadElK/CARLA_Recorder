from pathlib import Path
import numpy as np
import shutil
import os

def process_sequence(input_path:Path, output_path:Path):
    data_dirs = [dir for dir in input_path.iterdir() if dir.name != "visualizations"]
    vis_dir = input_path / 'visualizations'

    rgb_dir = input_path / 'rgb'

    indices_to_include = sorted(get_indices_to_include(data_dirs))

    events_dir = input_path / 'dvs'


    for i, idx in enumerate(indices_to_include):
        if i == 0:
            # skip the first frame (only used in events)
            continue
        


        pass
    


    pass

def get_indices_to_include(data_dirs:list):
    # Returns a list of frame numbers that are present in every data directory
    sorted_dirs = data_dirs.copy()
    sorted_dirs.sort(key=lambda dir : len(list(dir.iterdir())))
    reference_dir = sorted_dirs[0]

    return [int(file.stem) for file in reference_dir.iterdir()]


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