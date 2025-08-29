"""
Acoustic Autoencoder Training Script

"""

#================================== Imports ==================================
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.data import Dataset



import numpy as np
import datetime
import os
import time
import json
import yaml

# Local modules
from models.acoustic_autoencoder import recon_model
from utils.loss import trap_amplitude_loss, raw_trap_amplitude_loss
from utils.plots import plot_preds

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

    C = load_config("configs/train.yaml")

    data_dir = C["paths"]["data_dir"]

    # Experiment output folder
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    exp_dir = f"experiments/exp_{current_datetime}/"

    eval_figs_dir = f"{exp_dir}train_eval_figs/"
    ensure_dir(eval_figs_dir)

    progression_figs_dir = f"{exp_dir}progression_figs/"
    plot_progression = C["run"]["plot_progression"]
    if plot_progression:
        ensure_dir(progression_figs_dir)

    print("-"*100)
    print(f"Experiment directory: {exp_dir}")
    print("-"*100)


    # TensorBoard writer
    writer = SummaryWriter() # Launch with: python -m tensorboard.main --logdir=runs
    #writer = None

    # Training hyperparameters
    epochs = C["hyperparameters"]["epochs"]
    batch_size = C["hyperparameters"]["batch_size"]
    lr =  C["hyperparameters"]["learning_rate"]
    trap_loss_weight =  C["hyperparameters"]["trap_loss_weight"]
    raw_trap_loss_weight = C["hyperparameters"]["raw_trap_loss_weight"]

    #================================== Load and Normalise Data ==================================

    #Load target field magnitudes and normalise per sample (in case they are not normalised already)
    target_pattern = np.load(os.path.join(data_dir, "acoustic_traps.npy"))
    max_vals = np.amax(np.abs(target_pattern), axis=(1, 2), keepdims=True)
    target_pattern = target_pattern/max_vals
    target_pattern = torch.Tensor(target_pattern[:, np.newaxis ])

    # Load trap coordinates (for loss computation later) and source phases (only for visualisation later)
    trap_coords = np.load(os.path.join(data_dir,"trap_coords.npy"), allow_pickle=True)

    # Create DataLoaders
    dataset = CustomDataset(target_pattern, trap_coords)
    train_set, val_set = train_test_split(dataset, test_size=0.2, shuffle = True)
    train_loader  = DataLoader(train_set, batch_size=batch_size, shuffle = True )
    val_loader  = DataLoader(val_set, batch_size=batch_size, shuffle = True)


    #================================== Model, Optimiser, and Loss Function Initialiation ==================================
    reduce_elements = C["run"]["reduce_elements"]
    if reduce_elements: 
        no_elements_per_side = C["run"]["no_elements_per_side"]
    else: 
        no_elements_per_side = target_pattern.shape[-1]

    model = recon_model(reduce_elements, no_elements_per_side)

    #look at shape of typical batches in data loaders
    for idx, (X_, Y_) in enumerate(train_loader):
        print("X: ", X_.shape)
        print("Y: ", Y_.shape)
        if idx >= 0:
            break

    #model architecture summary
    summary(model,
            input_data = X_,
            col_names=["input_size",
                        "output_size",
                        "num_params"])



    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    criterion = nn.L1Loss() # MAE

    #================================== Metric Tracking ==================================
    experiment_summary = {
        "data_dir": data_dir,
        "reduce_elements": reduce_elements, 
        "no_elements_per_side": no_elements_per_side,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "trap loss weight (norm)": trap_loss_weight,
        "raw trap loss weight": raw_trap_loss_weight,
        "total_time": 0,
        "time_per_train_epoch": 0
    }

    metrics = dict([(f"Epoch {epoch+1}", {"training_total_loss": [], "training_recon_loss": [], "training_amp_loss": [], "training_raw_amp_loss": [], "validation_total_loss": [], "validation_recon_loss": [], "validation_amp_loss": [], "validation_raw_amp_loss":[]}) for epoch in range(epochs)])


    # Desired number of progression plots per epoch
    no_desired_figs_per_epoch = 25
    train_batches_per_epoch = len(train_set)//batch_size
    train_plot_every_n_batches = train_batches_per_epoch // no_desired_figs_per_epoch

    val_batches_per_epoch = len(val_set)//batch_size
    val_plot_every_n_batches = val_batches_per_epoch // 2


    #================================== Training Loop ==================================
    start_time = time.time()
    for epoch in range(1, epochs + 1):

        print(f"\n Epoch {epoch}:")

        train_total_loss = []
        train_recon_loss = []
        train_amp_loss = []
        train_raw_amp_loss = []

        train_count = 0

        val_total_loss = []
        val_recon_loss = []
        val_amp_loss = []
        val_raw_amp_loss = []


        for b, (og_magn, og_trap_coords) in enumerate(train_loader, start = 1): 

            # Forward pass
            P_z, pred_magn_norm, pred_magn_raw, pred_phase = model(og_magn)
            
            # MAE Reconstruction Loss
            recon_loss = criterion(pred_magn_norm, og_magn)

            amp_loss = trap_amplitude_loss( pred_magn_norm, og_trap_coords)
            raw_amp_loss = raw_trap_amplitude_loss(pred_magn_raw, og_trap_coords)


            loss = recon_loss + trap_loss_weight * amp_loss + raw_trap_loss_weight*raw_amp_loss

    
            # Backpropagation 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # Generate progression plots/store data every few batches
            if plot_progression == True: #and 1 < epoch < 5: 
                if (b % train_plot_every_n_batches == 0):

                    # Store the results (plot and numpy array) for the first item in the current batch
                    # ----------------------------------------------------------------------------------
                    # plot_preds(og_magn.detach()[0][0], pred_magn_raw.detach()[0][0], 
                    #            pred_phase.detach()[0][0], og_trap_coords.detach()[0][0], 
                    #            dir = f"{progression_figs_dir}train_epoch{epoch}_batch{b}")
                    # arr = np.array([np.array(og_magn.detach()[0][0]), np.array(pred_magn_norm.detach()[0][0]),
                    #                 np.array(pred_magn_raw.detach()[0][0]), np.array(pred_phase.detach()[0][0])])
                    # np.save(f"{progression_figs_dir}train_epoch_{epoch}_batch{b}", arr)


                    # this is to track the progression (loss and plots) of the SAME input at each stage. 
                    # ----------------------------------------------------------------------------------
                    model.eval()
                    test = val_set[6][0][0]
                    test = test[np.newaxis, np.newaxis, :]
                    with torch.no_grad(): 
                        
                        P_z, pred_magn_norm, pred_magn_raw, pred_phase = model(test)
                        temp_loss = criterion(pred_magn_norm, test)

                        plot_preds(test.detach()[0][0], pred_magn_raw.detach()[0][0], 
                                pred_phase.detach()[0][0], og_trap_coords.detach()[0], 
                                dir = f"{progression_figs_dir}loss_{temp_loss.detach():.4f}_train_epoch{epoch}_batch{b}.jpg")
                    model.train()


            train_total_loss.append(loss.item())
            train_recon_loss.append(recon_loss.item())
            train_amp_loss.append(amp_loss.item()*trap_loss_weight)
            train_raw_amp_loss.append(raw_amp_loss.item()*raw_trap_loss_weight)

            if writer is not None: 
                writer.add_scalar("Loss/Train", loss.item(), b+train_batches_per_epoch*(epoch-1))

            # Estimate epoch duration
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time-start_time
            if epoch == 1:
                experiment_summary["time_per_train_epoch"] = f"{epoch_duration/60:.4f} minutes"


        #====================== Validation Loop ======================
        model.eval()
        for b, (og_magn, og_trap_coords) in enumerate(val_loader): 

            with torch.no_grad(): 
                P_z, pred_magn_norm, pred_magn_raw, pred_phase = model(og_magn)

                recon_loss = criterion(pred_magn_norm, og_magn)
                amp_loss = trap_amplitude_loss( pred_magn_norm, og_trap_coords)
                raw_amp_loss = raw_trap_amplitude_loss(pred_magn_raw, og_trap_coords)
            
                loss = recon_loss + trap_loss_weight * amp_loss + raw_trap_loss_weight*raw_amp_loss


                # Store results (plots and arrays) of last batch of the last epoch
                if epoch == epochs and b == len(val_loader)-1: #and b >= len(val_loader)-2:
                    for i in range(len(og_magn)):


                        plot_preds(og_magn.detach()[i][0], pred_magn_raw.detach()[i][0], 
                                pred_phase.detach()[i][0], og_trap_coords.detach()[i], dir = f"{eval_figs_dir}{i}")

                        arr = np.array([np.array(og_magn.detach()[i][0]), np.array(pred_magn_norm.detach()[i][0]), 
                                        np.array(pred_magn_raw.detach()[i][0]), np.array(pred_phase.detach()[i][0])])
                        
                        coords_arr = og_trap_coords.detach()[i]
                        np.save(f"{eval_figs_dir}{i}", arr)
                        np.save(f"{eval_figs_dir}coords_{i}", coords_arr)

                val_total_loss.append(loss.item())
                val_recon_loss.append(recon_loss.item())
                val_amp_loss.append(amp_loss.item()*trap_loss_weight)
                val_raw_amp_loss.append(raw_amp_loss.item()*raw_trap_loss_weight)

                if writer is not None: 
                    writer.add_scalar("Loss/Validation", loss.item(),  b+train_batches_per_epoch*(epoch-1))



        #====================== Epoch Data Collection  ======================
        torch.save(model.state_dict(),f"{exp_dir}final_model.pth")

        metrics[f"Epoch {epoch}"]["training_total_loss"].append(train_total_loss)
        metrics[f"Epoch {epoch}"]["validation_total_loss"].append(val_total_loss)
        epoch_train_total_loss = np.array(train_total_loss).mean()
        print(f"Epoch {epoch} Mean Train Total Loss: {epoch_train_total_loss}")

        metrics[f"Epoch {epoch}"]["training_recon_loss"].append(train_recon_loss)
        metrics[f"Epoch {epoch}"]["training_amp_loss"].append(train_amp_loss)
        metrics[f"Epoch {epoch}"]["training_raw_amp_loss"].append(train_raw_amp_loss)

        metrics[f"Epoch {epoch}"]["validation_recon_loss"].append(val_recon_loss)
        metrics[f"Epoch {epoch}"]["validation_amp_loss"].append(val_amp_loss)
        metrics[f"Epoch {epoch}"]["validation_raw_amp_loss"].append(val_raw_amp_loss)


        epoch_train_recon_loss = np.array(train_recon_loss).mean()
        epoch_train_amp_loss = np.array(train_amp_loss).mean()
        epoch_train_raw_amp_loss = np.array(train_raw_amp_loss).mean()

        print(f"Epoch {epoch} Mean Train Reconstruction Loss: {epoch_train_recon_loss}")
        print(f"Epoch {epoch} Mean Train Trap Amplitude Loss (weighted): {epoch_train_amp_loss}")
        print(f"Epoch {epoch} Mean Train Raw Trap Amplitude Loss (weighted): {epoch_train_raw_amp_loss}")

        
        epoch_val_total_loss = np.array(val_total_loss).mean()
        print(f"\nEpoch {epoch} Mean Val Total Loss: {epoch_val_total_loss.item()}")

        epoch_val_recon_loss = np.array(val_recon_loss).mean()
        epoch_val_amp_loss = np.array(val_amp_loss).mean()
        epoch_val_raw_amp_loss = np.array(val_raw_amp_loss).mean()

        print(f"Epoch {epoch} Mean Val Reconstruction Loss: {epoch_val_recon_loss}")
        print(f"Epoch {epoch} Mean Val Trap Amplitude Loss (weighted): {epoch_val_amp_loss}")
        print(f"Epoch {epoch} Mean Val Raw Trap Amplitude Loss (weighted): {epoch_val_raw_amp_loss}")



    #================================== Save Results ==================================
    end_time = time.time()
    duration = end_time - start_time
    experiment_summary["total_time"] = f"{duration/60:.4f} minutes"

    with open(f"{exp_dir}metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(f"{exp_dir}experiment_summary.json", "w") as f:
        json.dump(experiment_summary, f, indent=4)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
   main()

