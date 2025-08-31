import numpy as np

def make_binary_trap_target(shape, trap_coords, radius_pixels=2.5, weights = None):
    """
    Creates a binary target map with circular trap regions set to 1.
    Args:
        shape (tuple): Shape of the 2D output array (height, width).
        trap_coords (list of tuples): List of the (x, y) trap centers.
        radius_pixels (float): Radius of circular trap area in pixels.
        weights (list of floats): Optional weights for each of the traps - if None, then they are all set to 1

    Returns:
        ndarray: Binary target image.
    """
    if weights == None: 
        weights = np.ones(shape = len(trap_coords))

    target = np.zeros(shape, dtype=np.float32)
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for weight, (cx, cy) in zip(weights, trap_coords):
        mask = (X - cx)**2 + (Y - cy)**2 <= radius_pixels**2
        target[mask] = 1*weight

    return target

def get_random_trap_coords(n_traps, shape, edge_lim = 2, min_trap_distance = 12): 
    """
    Randomly generates non-overlapping trap coordinates within a 64x64 grid.
    Ensures traps are spaced apart by at least 12 pixels.

    Args:
        n_traps (int): Number of trap-coordinates to generate.
        shape (tuple of int): Grid size (height, width).
        edge_lim (int, optional): Minimum distance from edges. Default is 2.
        min_trap_distance (int, optional): Minimum distance between traps. Default is 12.

    Returns:
        list of tuples: Trap coordinates.
    """
    lower_lim = edge_lim 
    upper_lim = shape[0] - edge_lim

    rng = np.random.default_rng()
    trap_coords = []

    too_near = True
    for t in range(n_traps): 
        fp = (rng.integers(lower_lim, upper_lim), rng.integers(lower_lim, upper_lim))

        if t != 0:
            while too_near == True: 
                for e_fp in trap_coords: 
                    if np.sqrt(np.sum((np.array(e_fp) - np.array(fp))**2)) >= min_trap_distance:  
                        too_near = False
                    else: 
                        fp = (rng.integers(lower_lim, upper_lim), rng.integers(lower_lim, upper_lim))
                        
        too_near = True
        trap_coords.append(fp)
    
    return trap_coords



def two_traps_moving_closer(no_samples):
    """
    Creates a series of 2-trap coordinate pairs that gradually move closer  along the x-axis.

    Args:
        no_samples (int): Number of samples to generate.

    Returns:
        list: List of the two coordinate pairs [(x1, y1), (x2, y2)] per sample.
    """
    all_coords = []
    for i in range(int(no_samples)): 
        coords = [(4+i, 32), (60-i, 32)]
        all_coords.append(coords)

    return all_coords


def symmetric_traps(n_samples, n_traps, centre =(32, 32), radius=24):
    """
    Generate trap coordinates.
    
    The first sample is a symmetric arrangement of "n_traps" on a circle 
    around "centre" with radius "radius". 
    The remaining samples are None so that their trap locations are chosen randomly.

    Args:
        n_samples (int): total number of samples
        n_traps (int): number of traps in the first sample
        center (tuple): (x, y) center of the circle
        radius (int): radius of the circle

    Returns:
        list: trap coordinates
    """
    traps = []
    
    # symmetric traps for first sample
    coords = []
    for i in range(n_traps):
        angle = 2 * np.pi * i / n_traps
        x = int(centre[0] + radius * np.cos(angle))
        y = int(centre[1] + radius * np.sin(angle))
        coords.append((x, y))
    
    traps.append(coords)
    
    # remaining samples are None
    traps.extend([None] * (n_samples - 1))
    
    return traps

