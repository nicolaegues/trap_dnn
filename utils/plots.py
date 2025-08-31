
import matplotlib.pyplot as plt
import numpy as np


def plot_preds(og_magn, pred_magn_raw, pred_phase, coords, dir):

    fig, axes = plt.subplots(1, 3, figsize = (10, 3.2))
    im1 = axes[0].imshow(og_magn)
    axes[0].set_title("Target")

    im2 = axes[1].imshow(pred_phase, cmap = "twilight")
    axes[1].set_title("Predicted Phases")

    im3 = axes[2].imshow(pred_magn_raw)
    axes[2].set_title("Predicted Acoustic Field Magnitude")
    for p in coords: 
        axes[2].plot(p[0], p[1], "ro", markersize = 2)
    

    for ax in axes: 
        ax.set_axis_off()

    fig.colorbar(im1, ax = axes[0], shrink = 0.85)
    fig.colorbar(im2, ax = axes[1], shrink = 0.85)
    fig.colorbar(im3, ax = axes[2], shrink = 0.85)

    fig.tight_layout()


    fig.savefig(dir, dpi = 300)
    plt.close()


def plot_Gorkov_ims(og_magn, pred_magn_raw, U, Laplacian, dir): 

    font = {'fontname':'Times New Roman'}
    fsize = 12

    fig, axes = plt.subplots(1, 4, figsize = (13.2, 3))

    im0 = axes[0].imshow(og_magn)
    axes[0].set_title("Target", fontsize = fsize, **font)
    im1 = axes[1].imshow(pred_magn_raw)
    axes[1].set_title("Acoustic Field Magnitude", fontsize = fsize, **font)
    im2 = axes[2].imshow(U, cmap = "seismic")
    axes[2].set_title("Gor'kov Potential", fontsize = fsize, **font)
    im3 = axes[3].imshow(Laplacian, cmap = "seismic")
    axes[3].set_title("Gor'kov Laplacian", fontsize = fsize, **font)


    for ax in axes: 
        ax.set_axis_off()

    fig.colorbar(im0, ax = axes[0], shrink = 0.85)
    fig.colorbar(im1, ax = axes[1], shrink = 0.85)
    fig.colorbar(im2, ax = axes[2], shrink = 0.85)
    fig.colorbar(im3, ax = axes[3], shrink = 0.85)


    fig.tight_layout()


    fig.savefig(dir, dpi = 300)
    plt.close()


def plot_Gorkov_ims_signature(pred_magn_raw, P_z_sign, pred_phases, sign_phases, U, Lap, dir): 

    font = {'fontname':'Times New Roman'}
    fsize = 12

    fig, axes = plt.subplots(3, 2, figsize = (9, 11))

    im0 = axes[0][0].imshow(pred_magn_raw)
    axes[0][0].set_title("Predicted traps", fontsize = fsize, **font)
    im1 = axes[0][1].imshow(P_z_sign)
    axes[0][1].set_title("Traps with signature", fontsize = fsize, **font)
    im2 = axes[1][0].imshow(pred_phases, cmap = "twilight")
    axes[1][0].set_title("Predicted Phases", fontsize = fsize, **font)
    im3 = axes[1][1].imshow(sign_phases, cmap = "twilight")
    axes[1][1].set_title("Phases with signature", fontsize = fsize, **font)
    im4 = axes[2][0].imshow(U, cmap = "seismic")
    axes[2][0].set_title("Gor'kov Potential", fontsize = fsize, **font)
    im5 = axes[2][1].imshow(Lap, cmap = "seismic")
    axes[2][1].set_title("Gor'kov Laplacian", fontsize = fsize, **font)



    for i in range(2):
        for ax in axes[i]: 
            ax.set_axis_off()
    
    shrink = 1

    fig.colorbar(im0, ax = axes[0][0], shrink = shrink )
    fig.colorbar(im1, ax = axes[0][1], shrink = shrink )
    fig.colorbar(im2, ax = axes[1][0], shrink = shrink )
    fig.colorbar(im3, ax = axes[1][1], shrink =  shrink )
    fig.colorbar(im4, ax = axes[2][0], shrink =  shrink )
    fig.colorbar(im5, ax = axes[2][1], shrink =  shrink )



    fig.tight_layout()


    fig.savefig(dir, dpi = 300)
    plt.close()



def plot_iasa_comp( i_0, i_final, dir): 

    P0_angle_0, Pz_magn_0 = i_0
    P0_angle_f, Pz_magn_f = i_final

    fig, axes = plt.subplots(2, 2, figsize = (10, 8))

    pos1 = axes[0][0].imshow(P0_angle_0, cmap = "twilight")
    axes[0][0].set_title("Phase field (iter = 1)")
    pos2 = axes[0][1].imshow(Pz_magn_0)
    axes[0][1].set_title("Acoustic Field Magnitude (iter = 1)")

    pos3 = axes[1][0].imshow(P0_angle_f, cmap = "twilight")
    axes[1][0].set_title("Phase field (iter = 200)")
    pos4 = axes[1][1].imshow(Pz_magn_f)
    axes[1][1].set_title("Acoustic Field Magnitude (iter = 200)")

    for ax in axes:
        for ax_ in ax: 
            ax_.set_axis_off()

    fig.colorbar(pos1, ax = axes[0][0])
    fig.colorbar(pos2, ax = axes[0][1])
    fig.colorbar(pos3, ax = axes[1][0])
    fig.colorbar(pos4, ax = axes[1][1])

    plt.tight_layout()
    plt.savefig(dir)
    #plt.show()
    plt.close()

