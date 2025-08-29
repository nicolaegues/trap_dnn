import numpy as np
import os
import numpy as np
from utils.binary_traps import get_random_trap_coords, make_binary_trap_target, two_traps_moving_closer, symmetric_traps
import yaml
    

def generate_traps(no_samples, output_dir, shape=(64, 64), n_traps=2, trap_coords = None):
    """
    Generates and saves synthetic binary trap images.

    Args:
        no_samples (int): Number of samples to generate.
        output_dir (str): Path to directory where the .npy files will be saved.
        shape (tuple): Shape of each 2D trap image.
        n_traps (int): Number of traps per image (used if trap_coords is None).
        trap_coords (list of list of tuples): Optional fixed trap coordinates per sample.
    """

    #safety overwrite
    if trap_coords is None: 
        no_samples = no_samples
    else: 
        no_samples = len(trap_coords)

    traps = np.zeros((int(no_samples), *shape))
    coords = []


    for i in range(int(no_samples)):
        
        # randomly place n_traps if no coordinates are given
        if trap_coords is None or trap_coords[i] is None:
            sample_coords = get_random_trap_coords(n_traps, shape)
        else: 
            sample_coords = trap_coords[i]

        #target = make_gaussian_trap_target(shape, trap_coords[i])
        radius_pixels = 2.5

        target = make_binary_trap_target(shape, sample_coords, radius_pixels=radius_pixels)


        traps[i] = target
        coords.append(sample_coords)

    np.save(os.path.join(output_dir, "acoustic_traps.npy"), traps)
    np.save(os.path.join(output_dir, "trap_coords.npy"), np.array(coords))



def load_config(path: str):
    """Loads YAML configuration file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":   

    C = load_config("configs/datagen.yaml")

    output_dir  = C["paths"]["output_dir"]
    trap_coords = C["run"]["trap_coords"]
    no_samples = C["run"]["no_samples"]
    no_traps = C["run"]["no_traps"]


    #trap_coords = two_traps_moving_closer(20)
    #trap_coords = symmetric_traps(no_samples, no_traps )
    

    generate_traps(no_samples=no_samples, output_dir = output_dir, shape = (64, 64), n_traps = no_traps, trap_coords = trap_coords)

