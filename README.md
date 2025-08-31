# Trap DNN


## Repository Structure
```
.
├── train.py               # Training loop for the autoencoder
├── test.py                # Test/Use the trained model
├── iasa.py                # Classical Iterative Angular Spectrum Algorithm 
├── generate_binary_traps.py # Generate synthetic binary trap targets
├── add_signature.py        # Turns focal point traps into vortex traps
├── configs/               # YAML configs for data, training, testing, signatures
│   ├── datagen.yaml
│   ├── train.yaml
│   ├── test.yaml
│   ├── iasa.yaml
│   └── signatures.yaml
├── models/                
│   └── acoustic_autoencoder.py   # Autoencoder model with a spectral layer at the bottleneck
├── physics/
│   ├── asm.py             # Angular Spectrum Method 
│   └── gorkov.py          # Gor'kov potential and Laplacian calculation
├── utils/
│   ├── binary_traps.py    # Target generation and trap coordinate utilities
│   ├── eval.py            # Metrics: MAE, PSBR, Trap Amplitude Variance, and other parameters
│   ├── loss.py            # Custom losses 
│   └── plots.py           # Plotting utilities for fields, phases, Gorkov maps
├── requirements.txt       # Dependencies
└── README.md
```

---

## Installation

### Prerequisites

- Dependencies in `requirements.txt`

### Setup
```bash
# Clone repository
git clone https://github.com/nicolaegues/trap_dnn.git
cd trap_dnn

# Create & activate virtual environment
conda create --name trap_dnn
conda activate trap_dnn

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Training Data
Before running, edit `configs/datagen.yaml` to set output path, number of samples and number of traps. Set trap_coords to None so that random traps will be generated for each sample.

Then run:
```bash
python generate_binary_traps.py
```
This will create and fill the specified `data/` folder with NumPy arrays containing the binary target patterns and trap coordinates.

### 2. Train the Autoencoder
Edit `configs/train.yaml` to set output path, model hyperparameters, and other flags. 

Then run:
```bash
python train.py
```
- Stores all results in an experiment folder: `experiments/exp_[date and time]/`
  - `final_model.pth`: the model's saved state dictinary (weights and biases)
  - `experiment_summary.json`: the experimental training parameters to look back on
  - `metrics.json`: the loss data points for training and validation
  - `train_eval_figs`: plots of the target, predicted phase and acoustic field magnitude for a subset fo the validation data, as well as the corresponding NumPy arrays for each sample.


### 3. Test a Trained Model
Edit `configs/test.yaml` to specify the path to the test data, to the saved model's state dictionary, and the output path, as well as some testing parameters.

Then run:
```bash
python test.py 
```
- Stores test results in a subdirectory of the original training folder (with the model)
  `metrics.json`: Contains the main metrics (Reconstruction MAE, PSBR, Trap Amplitude variance) for each sample, as well as other trap-specific outputs. The means and standard deviation for each metric over the entire test set is also given. 
- Stores plots of the target, predicted phases, and acoustic field magnitude for each sample. NumPy arrays containing those images are also stored for each sample. 
- Stores a used_config.yaml file for reproducibility.


### 4. Compare with IASA Baseline
Edit `configs/iasa.yaml` to adjust the IASA settings and test dataset path.  

Then run:
```bash
python iasa.py
```

### 5. Optional: Add the Signature for a Vortex Trap to the Phases 
Edit `configs/signatures.yaml` to specify the path to the phases the signature wants to be applied to. 

Then run:
```bash
python add_signature.py
```
---

