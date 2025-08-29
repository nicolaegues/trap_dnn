

""" 
Note: The code for the Angular Spectrum Method (ASM) originally belongs to Barney Emmens, 
but has been slightly modified/restructured. 
"""


import numpy as np
import torch

class ASM(): 
    """
    Propagates a 2D complex pressure field from z = 0 to z = z using 
    the Angular Spectrum Method (ASM).

    The input field P0 is first transformed into the spatial frequency domain,
    multiplied by a propagation filter (transfer function), and then transformed
    back via inverse FFT. High-angle components (evanescent waves) are filtered out.

    Args:
        P0 (ndarray): Complex-valued input pressure field at the source plane.
        dx (float): Spatial resolution [m] of the input grid.
        z (float): Propagation distance [m].

    Returns:
        ndarray: Complex-valued propagated field at distance z,
                cropped to the original shape of P0.
    """


    def __init__(self, resolution = (64, 64)):

        self.resolution = resolution

        
        self.c_medium = 1480.0 # speed of sound in water [m/s]

        # Acoustic wave parameters
        self.f = 1e6 # frequency of the emitted waves [Hz]
        self.wavelength = self.c_medium/self.f
        self.k = 2*np.pi*self.f/self.c_medium
        self.w = 2*np.pi*self.f

        # This part below is mainly to get a realistic aperture size, and in case a discrete grid wants to be generated instead. 
        # The main implementation, however, currently just simulates a square-shaped continuous source field (not discrete).
        # Source Parameters 
        self.element_width = 3e-3 # Width of one transducer element [m]
        self.kerf = 0.1e-3 # Gap between elements [m]
        self.N_elements_per_side = 7 # Square grid (NxN elements)
        self.pitch = self.element_width + self.kerf # Distance between centers of adjacent elements 
        self.aperture = self.N_elements_per_side*self.pitch - 2*self.kerf # Full aperture width
        self.Lx = 1.1*self.aperture  #size of () (slighlty larger than aperture)

        # Focal plane distance (propagation depth)
        self.z = 1.5*self.aperture
        self.dx = self.Lx/self.resolution[0]


        
    def __call__(self, P0, z = None):
        
        if z == None: 
            z = self.z

        # Padded size for FFT
        Nk = 2**int(np.ceil(np.log2(P0.shape[0]))+1)

        # Compute spatial frequency grids
        kv = 2*np.pi*np.fft.fftfreq(Nk, d = self.dx)
        kx, ky = np.meshgrid(kv, kv)
        kz =  np.emath.sqrt(self.k**2 - kx**2 - ky**2)
        
        # Propagator
        H = np.exp(-1j*kz*z)

        # Limit angular spectrum to propagating waves only
        D = (Nk-1)*self.dx
        kc = self.k*np.sqrt(0.5*(D**2)/(0.5*D**2 + z**2)) # Angular cutoff
        H[np.sqrt(kx**2 + ky**2) > kc] = 0 # Wavelengths greater than kc cannot propogate - Zero out evanescent components

        # Propagate the field
        P0_fourier = np.fft.fft2(P0,[Nk,Nk]) # Compute the 2D Fourier Transform of the input field
        P_z_fourier = P0_fourier * H
        P_z = np.fft.ifft2(P_z_fourier,[Nk,Nk]) # Compute the inverse 2D Fourier Transform of the field

        P_z = P_z[:P0.shape[0],:P0.shape[1]]

        return P_z

    
class TorchASM(): 
    
    """
        Propagates a 2D complex pressure field from z = 0 to z = z using 
        the Angular Spectrum Method (ASM) implemented in PyTorch.

        The input field P0 is transformed into the spatial frequency domain
        using torch.fft, multiplied by a propagation filter (transfer function), 
        and then transformed back via inverse FFT. High-angle components 
        (evanescent waves) are filtered out.

        Args:
            P0 (torch.Tensor): Complex-valued input pressure field at the source plane, 
                of shape (..., H, W).
            z (float, optional): Propagation distance [m]. Defaults to the focal plane 
                distance set during initialisation.

        Returns:
            torch.Tensor: Complex-valued propagated field at distance z, 
                cropped to the original shape of P0.
        """
    

    def __init__(self, resolution = (64, 64)):

        self.resolution = torch.tensor(resolution)

        
        self.c_medium = torch.tensor(1480.0)# speed of sound in water [m/s]

        # Acoustic wave parameters
        self.f = torch.tensor(1e6) # frequency of the emitted waves [Hz]
        self.wavelength = self.c_medium/self.f
        self.k = 2*np.pi*self.f/self.c_medium
        self.w = 2*np.pi*self.f

        # This part below is mainly to get a realistic aperture size, and in case a discrete grid wants to be generated instead. 
        # The main implementation, however, currently just simulates a square-shaped continuous source field (not discrete).
        # Source Parameters 
        self.element_midth = torch.tensor(3e-3) # Width of one transducer element [m]
        self.kerf = torch.tensor(0.1e-3) # Gap between elements [m]
        self.N_elements_per_side = torch.tensor(7) # Square grid (NxN elements)
        self.pitch = self.element_midth + self.kerf # Distance between centers of adjacent elements 
        self.aperture = self.N_elements_per_side*self.pitch - 2*self.kerf # Full aperture width
        self.Lx = 1.1*self.aperture  #size of () (slighlty larger than aperture)

        # Focal plane distance (propagation depth)
        self.z = 1.5*self.aperture
        self.dx = self.Lx/self.resolution[0] #what actually is dx? spacing of the sampling?

        
    def __call__(self, P0, z = None):
        
        if z == None: 
            z = self.z

        # Padded size for FFT
        Nk = 2**int(np.ceil(np.log2(P0.shape[-1]))+1)
        kmax = 2*np.pi/self.dx

        # Compute spatial frequency grids
        kv = torch.fft.fftfreq(Nk)*kmax 
        kx, ky = torch.meshgrid(kv, kv, indexing='ij')
        kz = torch.sqrt((self.k**2 - kx**2 - ky**2).to(torch.complex64))# Allow for complex values

        # Transfer function
        H = torch.exp(-1j*kz*z)

        D = (Nk-1)*self.dx
        kc = self.k*torch.sqrt(0.5*(D**2)/(0.5*D**2 + z**2))  # Angular cutoff
        H[torch.sqrt(kx**2 + ky**2) > kc] = 0 

        # Propagate the field
        P0_fourier = torch.fft.fft2(P0,[Nk,Nk]) # Compute the 2D Fourier Transform of the input field
        P_z_fourier = P0_fourier * H

        P_z = torch.fft.ifft2(P_z_fourier,[Nk,Nk]) # Compute the inverse 2D Fourier Transform of the field
        P_z = P_z[..., :P0.shape[-1], :P0.shape[-1]]

        return P_z