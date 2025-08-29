
import numpy as np
import os
import re
from physics.asm import ASM
from physics.gorkov import AcousticTrapAnalyser
from utils.plots import plot_magn_phase_ims, plot_Gorkov_ims_signature
from utils.eval import trap_analysis
import json
import yaml


#================================== Signature Functions ==================================

def wrap_to_pi(phi):
    #Map to (-pi, pi]
    return (phi + np.pi) % (2*np.pi) - np.pi


def add_vortex_signature(P_phase):
    """
    Args:
        P_phase (ndarray): 2D array of phase values (radians).

    Returns:
        ndarray: Phase field with vortex pattern added, wrapped to [-π, π].
    """
    m = 1
    H, W = P_phase.shape
    cy, cx = ((H-1)/2, (W-1)/2) 
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    X, Y = np.meshgrid(x, y)
    sig = m * np.arctan2(Y, X)
    return wrap_to_pi(P_phase + sig)



def main():
    #================================== File Extraction ==================================
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    files = sorted(os.listdir(trap_dir), key=natural_sort_key)
    np_files = []
    coord_files = []
    for file in files: 
        coords_match = re.search(r"coords", file)
        match = re.search(r".npy", file)
        if match and not coords_match: 
            np_files.append(file)
        if coords_match: 
            coord_files.append(file)


    #================================== Discretisation and Forward-Propagation ==================================
    probe = np.load(f"{trap_dir}{np_files[0]}")[0]
    asm = ASM(resolution=(probe.shape[0], probe.shape[1]))
    ATA= AcousticTrapAnalyser(particle_material="polysterene") # e.g. bubble (air) or polysterene

    metrics = {i: {"trap_calcs": 0} for i in range(len(np_files))}

    for i, (file, coords_f) in enumerate(zip(np_files, coord_files)): 

        arr = np.load(f"{trap_dir}{file}")
        coords = np.load(f"{trap_dir}{coords_f}")
        og_magn, pred_magn_norm, pred_magn_raw, pred_phases = arr

        P0_phase = add_vortex_signature(pred_phases)

        #Forward Propagation using ASM
        amp = 1
        # Create the complex number
        P0 = amp * np.exp(1j * P0_phase)

        P_z_ASM = asm(P0)
        P_z_magn = np.abs(P_z_ASM)

        trap_dict, U, Laplacian, *_ = trap_analysis(asm, ATA, P_z_magn, coords, P_z_ASM)

        metrics[i]["trap_calcs"] = trap_dict


        plot_magn_phase_ims(P_z_magn, P0_phase, coords, dir = f"{signature_dir}{i}")
        arr = np.array([og_magn, np.zeros(shape = P_z_magn.shape), 
                        P_z_magn, P0_phase])
        np.save(f"{signature_dir}{i}", arr)

        plot_Gorkov_ims_signature(pred_magn_raw, P_z_magn,pred_phases,P0_phase, U, Laplacian, dir = f"{signature_dir}{i}_Gorkov")


    with open(f"{signature_dir}metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)




def load_config(path: str):
    """Loads YAML configuration file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":   

    C = load_config("configs/iasa.yaml")

    trap_dir = C["paths"]["trap_dir"]

    signature_dir = f"{trap_dir}vortex_traps/"
    os.makedirs(signature_dir, exist_ok=True)
            





