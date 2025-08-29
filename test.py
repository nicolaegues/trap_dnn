"""
Acoustic Autoencoder Testing Script

Description:
------------
This script evaluates a trained acoustic autoencoder model on test data.
It:
  - Loads and normalises acoustic trap patterns and trap coordinates
  - Runs the model to predict phases and pressure fields 
  - Saves visualisations of the results and numpy arrays
  - Computes metrics for each sample: 
         MAE reconstruction error, trap amplitude variance, PSBR, 
         amplitudes at the trap centres, and the Gor'kov potential and the Laplacian at the trap centres
  - Optionally saves visualisations of the Gor'kov potential and the Laplacian across the whole array
  - Stores results and a copy of the config file for reproducibility

"""

#================================== Imports ==================================
import torch
from torch.utils.data import Dataset

import numpy as np
from torch.utils.data import DataLoader
import os
import json
import time
import yaml

# Local modules
from models.acoustic_autoencoder import recon_model
from utils.plots import plot_preds, plot_Gorkov_ims
from utils.eval import evaluate_sample
from physics.asm import ASM
from physics.gorkov import AcousticTrapAnalyser


def ensure_dir(path):
   """Creates directory if it does not exist."""
   os.makedirs(path, exist_ok=True)

def load_config(path: str):
    """Loads YAML configuration file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
class CustomDataset(Dataset):
   def __init__(self, X, y, transform=None):
      self.X = X
      self.y = y
      self.transform = transform

   def __len__(self):
      return len(self.X)

   def __getitem__(self, idx):
      sample_x = self.X[idx]
      sample_y = self.y[idx]

      # Apply transformation if provided
      if self.transform:
         sample_x = self.transform(sample_x)
      
      return sample_x, sample_y
    

def main():
   #================================== Experiment Configuration ==================================

   C = load_config("configs/test.yaml")

   exp_dir  = C["paths"]["experiment_dir"]
   output_subdir = C["paths"]["output_subdir"]
   data_dir = C["paths"]["data_dir"]
   out_dir = f"{exp_dir}/{output_subdir}"

   ensure_dir(out_dir)

   print("-"*100)
   print(f"Experiment directory: {exp_dir}")
   print(f"Output subdirectory: {output_subdir}")
   print("-"*100)

   #save config snapshot
   with open(out_dir + "/used_config.yaml", "w") as f:
      yaml.safe_dump(C, f, sort_keys=False)

   #================================== Load and Normalise Data ==================================
   max_figs_to_test = C["run"]["max_figs_to_test"]


   #Load target field magnitudes and normalise per sample (in case they are not normalised already)
   target_pattern = np.load(data_dir + "acoustic_traps.npy")[:max_figs_to_test]
   max_vals = np.amax(np.abs(target_pattern), axis=(1, 2), keepdims=True)
   target_pattern = target_pattern/max_vals
   target_pattern = torch.Tensor(target_pattern[:, np.newaxis ])

   # Load trap coordinates (for loss computation later) and source phases (only for visualisation later)
   trap_coords = np.load(data_dir + "trap_coords.npy")[:max_figs_to_test]
   #trap_coords = np.zeros(shape = (target_pattern.shape[0], 2, 2))


   # Create DataLoader
   batch_size = 1
   dataset = CustomDataset(target_pattern, trap_coords)
   test_loader = DataLoader(dataset, batch_size=batch_size, shuffle = False)

   #================================== Initialise Simulation Classes  ==================================

   asm = ASM(resolution=(target_pattern.shape[-1], target_pattern.shape[-1]))
   ATA= AcousticTrapAnalyser(particle_material="air") # e.g. bubble (air) or polysterene

   #================================== Initialise Model ==================================
   # Initialise model and load its state dictionary in from the experiment's folder

   reduce_elements = C["run"]["reduce_elements"]
   if reduce_elements: 
      no_elements_per_side = C["run"]["no_elements_per_side"]
   else:
      no_elements_per_side = target_pattern.shape[-1]

   model = recon_model(reduce_elements, no_elements_per_side)
   model.load_state_dict(torch.load(f"{exp_dir}final_model.pth"))
   model.eval()

   #================================== Metric Containers ==================================

   metrics = {i: {"reconstruction_error": 0, "trap_amplitude_variance": 0, "PSBR": 0, "trap_calcs": 0} for i in range(len(target_pattern))}

   #store sample metrics to compute the mean later
   rec_errors = np.zeros(shape = len(trap_coords))
   trap_vars = np.zeros(shape = len(trap_coords))
   psbrs = np.zeros(shape = len(trap_coords))
   amplitudes = []
   potentials = []
   laplacians = []

   #================================== Evaluation Loop ==================================
   start = time.time()
   for i, (og_magn, og_trap_coords) in enumerate(test_loader): 
      
      with torch.no_grad(): 
         P_z, pred_magn_norm, pred_magn_raw, pred_phases = model(og_magn)


         og_magn = og_magn.detach().numpy()[0][0]
         P_z = P_z.detach().numpy()[0][0]
         pred_magn_norm = pred_magn_norm.detach().numpy()[0][0]
         pred_magn_raw = pred_magn_raw.detach().numpy()[0][0]
         pred_phases = pred_phases.detach().numpy()[0][0]
         og_trap_coords = og_trap_coords.detach().numpy()[0]

         plot_preds(og_magn, 
                     pred_magn_raw, pred_phases, 
                     og_trap_coords, dir = f"{out_dir}{i}")
         
         arr = np.array([np.array(og_magn), np.array(pred_magn_norm), 
                        np.array(pred_magn_raw), np.array(pred_phases)])
         
         np.save(f"{out_dir}{i}", arr)
         np.save(f"{out_dir}coords_{i}", og_trap_coords)


         #======================= Further Analysis of the resulting Pressure Field =======================

         
         mae_recon, trap_amps_var, psbr, trap_dict, U, Laplacian, interim_lists = evaluate_sample(asm, ATA, og_trap_coords, og_magn, pred_magn_raw, P_z)

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

         #Optional Gor'kov visualisation
         if C["run"]["plot_gorkov"] == True:

            plot_Gorkov_ims(og_magn, pred_magn_raw, U, Laplacian, dir = f"{out_dir}{i}_Gorkov")
                  
            gorkov_arr = np.array([np.array(og_magn), np.array(pred_magn_raw), 
                           np.array(U), np.array(Laplacian)])
            
            np.save(f"{out_dir}{i}_Gorkov", gorkov_arr)

            #1D plots
            #p2, v2, v2x, v2y = U_decom
            #plot_p2_and_v2(U, p2, v2, v2x, v2y)

   end = time.time()
   print(f"{end-start:.6}")

   #================================== Store Metrics ==================================


   means = {
         "mean_reconstruction_error": f"{np.mean(rec_errors):.6}",
         "mean_trap_amplitude_variance": f"{np.mean(trap_vars):.6}",
         "mean_PSBR": f"{np.mean(psbrs):.6}",
         "mean_amplitude": f"{np.mean(amplitudes):.6}",
         "mean_gorkov_potential": f"{np.mean(potentials):.6e}",
         "mean_gorkov_laplacian": f"{np.mean(laplacians):.6e}"
      }
   
   stds = {
      "std_reconstruction_error": f"{np.std(rec_errors, ddof=1):.6}",
      "std_trap_amplitude_variance": f"{np.std(trap_vars, ddof=1):.6}",
      "std_PSBR": f"{np.std(psbrs, ddof=1):.6}",
      "std_amplitude": f"{np.std(amplitudes, ddof=1):.6}",
      "std_gorkov_potential": f"{np.std(potentials, ddof=1):.6e}",
      "std_gorkov_laplacian": f"{np.std(laplacians, ddof=1):.6e}"
   }

   metrics["means"] = {**means, **stds}
   with open(f"{out_dir}metrics.json", "w") as f:
      json.dump(metrics, f, indent=4)


if __name__ == "__main__":
   main()



