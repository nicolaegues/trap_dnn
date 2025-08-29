
"""
Autoencoder with a spectral layer for trapping field reconstruction.
Spectral layer inspired by:
 - GedankenNet: https://github.com/PORPHURA/GedankenNet/blob/main/GedankenNet_Phase/networks/fno.py
 - Neural Operator: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/spectral_convolution.py
"""

import torch 
from torch import nn
import numpy as np
import torch.nn.functional as F
from physics.asm import TorchASM

#================================== Constants ==================================


asm = TorchASM()

nconv = 64
H = 64

class SpectralConv2d(nn.Module):
    """2D Fourier layer: keeps low-frequency modes, multiplies with learned weights."""
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.modes = modes
        scale = 1 / in_channels
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes // 2 + 1, 2))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(batchsize, x.size(1), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        x_ft = x_ft[:, :, :self.modes, :self.modes]
        w = torch.view_as_complex(self.weight)
        out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(x_ft, w)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
        return x

class SpectralBlock(nn.Module):
    """Spectral + Conv block with residual connection."""

    def __init__(self, channels, modes=16):
        super().__init__()
        self.spec_conv = SpectralConv2d(channels, channels, modes)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.prelu = nn.PReLU(channels) #try GELU?

    def forward(self, x):
        x_spec = self.spec_conv(x)
        x_conv = self.conv(x)
        return self.prelu(x + x_spec + x_conv)
    
class In(nn.Module):
    """Input convolution block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            #nn.PReLU(),
        )

    def forward(self, x):
        return self.conv(x) 

class Down(nn.Module):
    """Downsampling convolution block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            #nn.PReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            #nn.PReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module): 
    """Upsampling convolution block (bilinear interpolation + convolution)."""
   
    def __init__(self, in_channels, out_channels): 
        super().__init__()

        self.deconv_= nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

        )

        self.deconv= nn.Sequential(
    
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.PReLU(),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.PReLU(),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
    
    def forward(self, x):
        return self.deconv(x)
    
class Out(nn.Module): 
    """Output convolution layer"""
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)            
        

    def forward(self, x):
        return self.out(x)

class recon_model(nn.Module):
  """
  Autoencoder network for the reconstruction of an optimal trapping field.

  Args:
    reduce_elements (bool): If True, reduces the phase mask to a 
        discrete grid of transducer elements and upsamples back to 
        the original resolution. 
    N_elements_per_side (int): Number of elements per side of the 
        reduced grid when "reduce_elements" is True.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (B, 1, H, W) -  a target trap mask.

    Forward Returns:
        P_z_ASM (torch.Tensor): Complex propagated field at target plane.
        P_z_magn_norm (torch.Tensor): Normalised magnitude of the 
            propagated field (per sample).
        P_z_magn_raw (torch.Tensor): Raw normalised magnitude of the 
            propagated field.
        P0_phase (torch.Tensor): Predicted source phase mask in radians.
  
  """

  def __init__(self, reduce_elements = False, N_elements_per_side = 11):
    super(recon_model, self).__init__()

    self.reduce_elements = reduce_elements
    self.N_elements_per_side = N_elements_per_side

    self.inc = In(1, nconv) 
    self.down1 = Down(nconv, nconv*2)
    self.down2= Down(nconv*2, nconv*4)
    self.spec_block = SpectralBlock(nconv * 4, modes=16)
    self.up2 = Up(nconv*4,  nconv*2)
    self.up3 = Up(nconv*2,  nconv)

    self.outc = Out(nconv, 1)            


  def forward(self,x):

    x = self.inc(x)
    x = self.down1(x)
    x = self.down2(x)
    x = self.spec_block(x)
    x = self.up2(x)
    x = self.up3(x)

    logits = self.outc(x)

    P0_phase = logits
    P0_phase = torch.tanh(P0_phase) # tanh activation (-1 to 1) 
    P0_phase = P0_phase*np.pi # restore to (-pi, pi) range


    if self.reduce_elements == True:

        phase_elem = F.interpolate(P0_phase, size=(self.N_elements_per_side, self.N_elements_per_side), mode='area')
        P0_phase = F.interpolate(phase_elem, size=(P0_phase.shape[2], P0_phase.shape[2]), mode='nearest')


    #Create the complex number
    P0 = torch.exp(1j * P0_phase)     

    #==================== Forward Propagation ====================
    P_z_ASM = asm(P0)

    # Normalise
    max_vals = torch.amax(torch.abs(P_z_ASM), dim=(2, 3), keepdim=True)
    P_z_ASM_norm = P_z_ASM / max_vals
    P_z_magn_norm = torch.abs(P_z_ASM_norm)

    P_z_magn_raw = torch.abs(P_z_ASM)

    return P_z_ASM, P_z_magn_norm, P_z_magn_raw, P0_phase
  

