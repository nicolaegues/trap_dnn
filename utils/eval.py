import numpy as np

def mae(pred, target):
    """
    Compute mean absolute error.

    Args:
        pred (ndarray): Predicted array.
        target (ndarray): Ground truth array.

    Returns:
        float: Mean absolute error.
    """
    err = np.abs(pred - target)
    return err.mean()


def psbr(field, trap_coords, trap_radius=2.5):
    """
    Compute peak to sideband ratio (PSBR).

    Args:
        field (ndarray): Magnitude field [H, W].
        trap_coords (list of tuple): List of (x, y) trap coordinates.
        trap_radius (float): Radius around each trap to exclude from background.

    Returns:
        float: Peak to sideband ratio.
    """
    H, W = field.shape
    Y, X = np.ogrid[:H, :W]

    # Exclusion mask for trap disks
    mask = np.zeros((H, W), dtype=bool)

    peaks = []
    for (x, y) in trap_coords:
        iy, ix = int(round(y)), int(round(x))
        peaks.append(field[iy, ix])
        disk = (X - x)**2 + (Y - y)**2 <= trap_radius**2
        mask |= disk

    mean_peak_val = float(np.mean(peaks)) if peaks else np.nan
    bg_level = np.mean(field[~mask])

    return mean_peak_val / bg_level


def trap_analysis(asm, ATA, pred_magn_raw, og_trap_coords, P_z):
    """
    Analyse traps by amplitude, Gor'kov potential, and Laplacian.

    Args:
        asm (ASM): Angular Spectrum Method object.
        ATA (AcousticTrapAnalyser): Gor'kov analyser.
        pred_magn_raw (ndarray): Predicted raw magnitude field.
        og_trap_coords (list of tuple): Trap coordinates (x, y).
        P_z (ndarray): Complex propagated pressure field.

    Returns:
        tuple:
            trap_dict (dict): Trap metrics at each coordinate.
            U (ndarray): Gor'kov potential map.
            Laplacian (ndarray): Laplacian of potential.
            all_amps (ndarray): Amplitude values at traps.
            all_pots (ndarray): Potential values at traps.
            all_laps (ndarray): Laplacian values at traps.
    """
    trap_dict = {f"{trap}": {"amplitude": 0, "gorkov_potential": 0, "gorkov_laplacian": 0} 
                 for trap in og_trap_coords}

    dx = asm.dx
    U, *_ = ATA.gorkov_potential(P_z, dx)
    Laplacian, *_ = ATA.laplacian(U, dx)

    all_amps = np.zeros(len(og_trap_coords))
    all_pots = np.zeros(len(og_trap_coords))
    all_laps = np.zeros(len(og_trap_coords))

    for t, trap in enumerate(og_trap_coords):
        amp_at_trap = pred_magn_raw[trap[1], trap[0]]
        U_at_trap = U[trap[1], trap[0]]
        lap_at_trap = Laplacian[trap[1], trap[0]]

        trap_dict[f"{trap}"]["amplitude"] = f"{amp_at_trap:.6}"
        trap_dict[f"{trap}"]["gorkov_potential"] = f"{U_at_trap:.6e}"
        trap_dict[f"{trap}"]["gorkov_laplacian"] = f"{lap_at_trap:.6e}"

        all_amps[t] = amp_at_trap
        all_pots[t] = U_at_trap
        all_laps[t] = lap_at_trap
    
    return trap_dict, U, Laplacian, all_amps, all_pots, all_laps


def evaluate_sample(asm, ATA, og_trap_coords, og_magn, pred_magn_raw, P_z):
    """
    Evaluate reconstruction quality for one sample.

    Args:
        asm (ASM): Angular Spectrum Method object.
        ATA (AcousticTrapAnalyser): Gor'kov analyser.
        og_trap_coords (list of tuple): Ground truth trap coordinates.
        og_magn (ndarray): Ground truth magnitude field.
        pred_magn_raw (ndarray): Predicted raw magnitude field.
        P_z (ndarray): Complex propagated pressure field.

    Returns:
        tuple:
            mae_recon (float): Mean absolute error of reconstruction.
            trap_amps_var (float): Variance of trap amplitudes.
            psbr_val (float): Peak to sideband ratio.
            trap_dict (dict): Trap metrics.
            U (ndarray): Gor'kov potential map.
            Laplacian (ndarray): Laplacian of potential.
            lists (tuple): (all_amps, all_pots, all_laps).
    """
    pred_magn_norm = pred_magn_raw / np.max(np.abs(pred_magn_raw))

    trap_dict, U, Laplacian, all_amps, all_pots, all_laps = trap_analysis(
        asm, ATA, pred_magn_raw, og_trap_coords, P_z
    )

    mae_recon = mae(pred_magn_norm, og_magn)
    trap_amps_var = np.var(all_amps)
    psbr_val = psbr(pred_magn_raw, og_trap_coords)

    lists = all_amps, all_pots, all_laps

    return mae_recon, trap_amps_var, psbr_val, trap_dict, U, Laplacian, lists
