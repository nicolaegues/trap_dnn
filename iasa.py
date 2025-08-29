
import numpy as np
import os
import json
import time
import yaml

from utils.plots import plot_iasa_comp
from utils.binary_traps import get_random_trap_coords, make_binary_trap_target
from utils.eval import evaluate_sample
from utils.plots import plot_preds
from physics.asm import ASM
from physics.gorkov import AcousticTrapAnalyser



def element_projector(P0, N=7):
    """
    Downsamples a complex field onto an NxN element grid by calculating the complex mean from each block, 
    extracting the phase, and replacing the values in the block with a unit-magnitude complex field with that mean phase.

    Args:
        P0 (ndarray): 2D complex input field 
        N (int): Number of grid divisions per axis. Default is 7.

    Returns:
        ndarray: Discretised complex field.
    """
    H, W = P0.shape
    ys = np.round(np.linspace(0, H, N+1)).astype(int)
    xs = np.round(np.linspace(0, W, N+1)).astype(int)

    P0_proj = np.zeros_like(P0, dtype=np.complex128)
    for iy in range(N):
        for ix in range(N):
            y0, y1 = ys[iy], ys[iy+1]
            x0, x1 = xs[ix], xs[ix+1]
            if y1 <= y0 or x1 <= x0:
                continue

            # Complex average avoids phase-wrap problems
            cmean = P0[y0:y1, x0:x1].mean()
            phi = np.angle(cmean)

            P0_proj[y0:y1, x0:x1] = np.exp(1j * phi)

    return P0_proj


#================================== IASA Core Algorithm ==================================

def IASA(target, shape, focal_points, asm, ATA, sample_coords, iterations = 200, reduce_elements = False, N_elements_per_side = 11): 
    """
    Iterative Angular Spectrum Algorithm (IASA) to compute a phase-only pattern 
    for trapping particles at given focal points.
    
    The algorithm iteratively enforces constraints in the transducer and focal 
    planes by forward and backward propagating fields using the Angular Spectrum Method.
    
    After a warm-up period (default: 50 iterations), the algorithm evaluates the 
    acoustic intensity at the trap coordinates and adjusts the target field by 
    reweighting the amplitudes to balance trap strength. The new target is used 
    for the remaining iterations to help converge to uniform trap intensity.

    Args:
        target (ndarray): Target amplitude distribution in focal plane.
        shape (tuple): Shape of the phase/intensity map.
        focal_points (list): List of (x, y) coordinates for traps.
        dx (float): Pixel spacing in source plane.
        iterations (int): Number of IASA iterations.

    Returns:
        tuple: (Final phase pattern, intermediate result at iter=0, at iter=1)
    """
    # Backward popagate the target to transducer plane 
    P0 = asm(target)
    P0 = np.conj(P0)

    #Phase-only constraint
    phase = np.angle(P0)
    P0 = np.exp(1j*phase)

    if reduce_elements: 

        P0 = element_projector(P0, N=N_elements_per_side)

    for i in range(iterations):

        # Forward propagate
        P_z = asm(P0)

        # Store for visualisation later and print final trap magnitudes
        if i == 0: 
            i_0 = np.angle(P0), np.abs(P_z)
            #i_1 = np.angle(P0), np.abs(P_z)


        if i == iterations-1:
            trap_magns = []
            for pt in focal_points:
                magn = np.abs(P_z) 
                trap_magn = magn[pt[1], pt[0]]
                trap_magns.append(trap_magn)
            # print(trap_magns)
            # print("")

 
        #============ Applying weights to the traps ============
        if i == 50:  # warm-up period until then!

            # Measure field magnitude at each trap
            magn = np.abs(P_z)
            trap_magns = [magn[pt[1], pt[0]] for pt in focal_points]
            #print(trap_magns)

            # Normalize so that the strongest trap has weight = 1
            max_magn = max(trap_magns)
            weights = [max_magn / m / 1.1 if m > 0 else 1.0 for m in trap_magns]
            #weights = [1, 1.5]

            # Rebuild weighted target
            target = make_binary_trap_target(shape, focal_points, weights=weights)


        P_z_angle = np.angle(P_z)


        P_z_sugg =  target* np.exp(1j*P_z_angle)

        #backpropagate
        P0 = asm(P_z_sugg)
        P0 = np.conj(P0)

        #Phase-only constraint
        phase = np.angle(P0)
        P0 = np.exp(1j*phase)

        if reduce_elements: 
            # # Downsample to given size 
            P0 = element_projector(P0, N=N_elements_per_side)


    return P0, i_0

#================================== Data Generator ==================================
def generate_IASA(no_samples, output_dir, trap_coords, iterations, shape=(64, 64), n_traps=2, reduce_elements = False, N_elements_per_side = 11):
    """
    Generates synthetic acoustic phase data using IASA.

    Args:
        no_samples (int): Number of samples to generate.
        output_dir (str): Directory to save .npy files.
        shape (tuple): Shape of output arrays.
        n_traps (int): Number of acoustic traps per target.
    """
    #safety overwrite
    if trap_coords is None: 
        no_samples = no_samples
    else: 
        no_samples = min(len(trap_coords), no_samples)

    asm = ASM(resolution = shape)
    ATA = AcousticTrapAnalyser(particle_material="air")

    phases = np.zeros((int(no_samples), *shape))
    traps = np.zeros((int(no_samples), *shape))
    coords = []

    # safety over-write
    if not reduce_elements: 
        N_elements_per_side = traps.shape[-1]

    metrics = {i: {"reconstruction_error": 0, "trap_amplitude_variance": 0, "PSBR": 0, "trap_calcs": 0} for i in range(no_samples)}

    #store sample metrics to compute the mean later
    rec_errors = np.zeros(shape = int(no_samples))
    trap_vars = np.zeros(shape = int(no_samples))
    psbrs = np.zeros(shape = int(no_samples))
    amplitudes = []
    potentials = []
    laplacians = []

    for i in range(int(no_samples)):

        # randomly place n_traps if no coordinates are given
        if trap_coords is None or trap_coords[i] is None:
            sample_coords = get_random_trap_coords(n_traps, shape)
        else: 
            sample_coords = trap_coords[i]


        target = make_binary_trap_target(shape, sample_coords)

        P0, i_0 = IASA(target, shape, sample_coords, asm, ATA, sample_coords, iterations = iterations, reduce_elements=reduce_elements, N_elements_per_side=N_elements_per_side)
        pressure = asm(P0)
        P0_phase = np.angle(P0)
        i_final = P0_phase, np.abs(pressure)


        plot_preds(target,i_final[1], P0_phase, sample_coords, f"{output_dir}{i}")
        plot_iasa_comp( i_0, i_final, f"{output_dir}{i}_comp")

        phases[i] = P0_phase
        traps[i] = np.abs(pressure)
        coords.append(sample_coords)

        mae_recon, trap_amps_var, psbr, trap_dict, U, Laplacian, interim_lists = evaluate_sample(asm, ATA, sample_coords, target, np.abs(pressure), pressure)
        metrics[i]["reconstruction_error"] = f"{mae_recon:.6}"
        metrics[i]["trap_amplitude_variance"] = f"{trap_amps_var:.6}"
        metrics[i]["PSBR"] = f"{psbr:.6}"
        metrics[i]["trap_calcs"] = trap_dict

        rec_errors[i] = mae_recon
        trap_vars[i] = trap_amps_var
        psbrs[i] = psbr

        all_amps, all_pots, all_laps = interim_lists
        amplitudes.extend(all_amps)
        potentials.extend(all_pots)
        laplacians.extend(all_laps)

        arr = np.array([target, np.zeros(shape = target.shape), 
                     np.abs(pressure), P0_phase])

        np.save(f"{output_dir}{i}", arr)
        np.save(f"{output_dir}coords_{i}", sample_coords)



    means = {
         "mean_reconstruction_error": f"{np.mean(rec_errors):.6}",
         "mean_trap_amplitude_variance": f"{np.mean(trap_vars):.6}",
         "mean_PSBR": f"{np.mean(psbrs):.6}",
         "mean_amplitude": f"{np.mean(amplitudes):.6}",
         "mean_gorkov_potential": f"{np.mean(potentials):.6e}",
         "mean_gorkov_laplacian": f"{np.mean(laplacians):.6e}"
      }

    metrics["means"] = means

    with open(f"{output_dir}metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def load_config(path: str):
    """Loads YAML configuration file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":   

    C = load_config("configs/iasa.yaml")

    output_dir  = C["paths"]["output_dir"]
    coords_dir = C["paths"]["coords_dir"]
    load_coords = C["run"]["load_coords"]
    trap_coords = C["run"]["trap_coords"]
    no_samples = C["run"]["no_samples"]
    iterations = C["run"]["iterations"]
    reduce_elements = C["run"]["reduce_elements"]
    N_elements_per_side = C["run"]["N_elements_per_side"]

    os.makedirs(output_dir, exist_ok=True)

    if load_coords == True:
        trap_coords = np.load(coords_dir)
    else: 
        trap_coords = trap_coords


    start = time.time()
    generate_IASA(no_samples, output_dir = output_dir, trap_coords = trap_coords, iterations = iterations, reduce_elements=reduce_elements, N_elements_per_side=N_elements_per_side)
    end = time.time()
    print(end-start)


