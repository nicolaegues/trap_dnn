
import torch


def trap_amplitude_loss(pred_magn, coords):
    """
    Computes the MSE between normalised predicted amplitude and ideal target (1.0) at trap coordinates.

    Args:
        pred_magn (Tensor): Normalised predicted field magnitudes of shape (Batchsize, 1, H, W)
        coords (Tensor): shape (Batchsize, no_traps, 2) with x,y coordinates of the traps

    Returns:
        Scalar loss 
    """
    B = pred_magn.shape[0]
    loss = 0.0
    no_traps = coords.shape[1]
    
    for i in range(B):
        for t in range(no_traps):

            a = pred_magn[i][..., coords[i][t][1], coords[i][t][0]]
            # loss += F.mse_loss(a, torch.tensor([1.0]))
            per_trap = (a - 1)**2
            loss += per_trap.mean()


    return loss/ (B*no_traps)
    #return loss


def raw_trap_amplitude_loss( pred_magn_raw, coords ): 
    """
    Computes the MSE between raw predicted amplitude and ideal target (1.0) at trap coordinates.

    Args:
        pred_magn (Tensor): Raw predicted field magnitudes of shape (Batchsize, 1, H, W)
        coords (Tensor): shape (Batchsize, no_traps, 2) with x,y coordinates of the traps

    Returns:
        Scalar loss 
    """

    B = pred_magn_raw.shape[0]
    loss = 0.0
    no_traps = coords.shape[1]

    for i in range(B):

        for t in range(no_traps):
            raw_amp_at_trap = pred_magn_raw[i][..., coords[i][t][1], coords[i][t][0]]
            loss += raw_amp_at_trap

    return - torch.abs(loss/ (B*no_traps))

