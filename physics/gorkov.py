import numpy as np
import torch

class AcousticTrapAnalyser:
    """
    Analyse acoustic traps using Gor'kov potential in NumPy.
    Computes pressure gradients, Gor'kov potential, and Laplacian maps
    for given medium and particle properties.
    """

    def __init__(self, particle_material="polysterene", medium="water"):
        """
        Args:
            particle_material (str): Material of the particle 
                ("polysterene" or "air").
            medium (str): Medium ("water" or "air").
        """
        # Medium properties
        if medium == "water":
            self.c_m = 1480.0   # sound speed [m/s]
            self.rho_m = 997.0  # density [kg/m^3]
        elif medium == "air":
            self.c_m = 343
            self.rho_m = 1.204

        # Particle properties
        if particle_material == "polysterene":
            self.c_p = 2340
            self.rho_p = 1050.0
        elif particle_material == "air":
            self.c_p = 343
            self.rho_p = 1.204

        # Acoustic parameters
        self.f = 1e6                        # frequency [Hz]
        self.k = 2*np.pi*self.f/self.c_m    # wavenumber
        self.w = 2*np.pi*self.f             # angular frequency

        # Particle size
        self.a = 1e-5                       # radius [m]
        self.V = (4/3)*np.pi*self.a**3      # volume

    def spectral_derivs(self, p, dx):
        """
        Compute spatial derivatives of pressure using FFT.

        Args:
            p (ndarray): Complex pressure field of shape (..., H, W).
            dx (float): Spatial grid spacing [m].

        Returns:
            tuple of ndarray: (dpdx, dpdy) arrays of same shape as p.
        """
        *lead, H, W = p.shape
        Nk = 2**int(np.ceil(np.log2(W))+1)  # padded size

        kv = 2*np.pi*np.fft.fftfreq(Nk, d=dx)
        kx, ky = np.meshgrid(kv, kv)

        P = np.fft.fft2(p, [Nk, Nk])
        dpdx = np.fft.ifft2(1j*kx*P, [Nk, Nk])
        dpdy = np.fft.ifft2(1j*ky*P, [Nk, Nk])

        return dpdx[..., :W, :W], dpdy[..., :W, :W]

    def gorkov_potential(self, p, dx):
        """
        Compute Gor'kov potential and energy density maps.

        Args:
            p (ndarray): Complex harmonic pressure field [Pa].
            dx (float): Spatial grid spacing [m].

        Returns:
            tuple: (U, p2_avg, v2_avg, vx2_avg, vy2_avg) where
                U (ndarray): Gor'kov potential [J].
                p2_avg (ndarray): Time-averaged pressure squared.
                v2_avg (ndarray): Time-averaged velocity squared.
                vx2_avg (ndarray): Time-averaged x-velocity squared.
                vy2_avg (ndarray): Time-averaged y-velocity squared.
        """
        rho0, c0 = self.rho_m, self.c_m
        rho_p, c_p = self.rho_p, self.c_p
        omega = self.w

        # Contrast factors
        kappa0 = 1.0/(rho0*c0**2)
        kappap = 1.0/(rho_p*c_p**2)
        f1 = 1.0 - kappap/kappa0
        f2 = 2.0*(rho_p - rho0)/(2.0*rho_p + rho0)

        dpdx, dpdy = self.spectral_derivs(p, dx)
        vx = dpdx/(-1j*omega*rho0)
        vy = dpdy/(-1j*omega*rho0)

        p2_avg = 0.5*np.abs(p)**2
        v2_avg = 0.5*(np.abs(vx)**2 + np.abs(vy)**2)

        U = self.V*(f1*0.5*kappa0*p2_avg - f2*0.75*rho0*v2_avg)

        return np.real(U), p2_avg, v2_avg, 0.5*np.abs(vx)**2, 0.5*np.abs(vy)**2

    def laplacian(self, U, dx):
        """
        Compute Laplacian of Gor'kov potential using FFT.

        Args:
            U (ndarray): Gor'kov potential map [J].
            dx (float): Spatial grid spacing [m].

        Returns:
            tuple: (lap, Uxx, Uyy) where
                lap (ndarray): Laplacian of U.
                Uxx (ndarray): Second derivative in x.
                Uyy (ndarray): Second derivative in y.
        """
        *lead, H, W = U.shape
        Nk = 2**int(np.ceil(np.log2(W))+1)

        kv = 2*np.pi*np.fft.fftfreq(Nk, d=dx)
        kx, ky = np.meshgrid(kv, kv)

        Uhat = np.fft.fft2(U, [Nk, Nk])
        Uxx = np.fft.ifft2(-(kx**2)*Uhat, [Nk, Nk])
        Uyy = np.fft.ifft2(-(ky**2)*Uhat, [Nk, Nk])

        Uxx = np.real(Uxx[..., :W, :W])
        Uyy = np.real(Uyy[..., :W, :W])

        return Uxx + Uyy, Uxx, Uyy


class TorchAcousticTrapAnalyser:
    """
    Analyse acoustic traps using Gor'kov potential in PyTorch.

    GPU-friendly version of AcousticTrapAnalyser.
    """

    def __init__(self):
        """Initialise with water medium and polystyrene particle."""
        self.c_m = torch.tensor(1480.0)
        self.c_p = torch.tensor(2340.0)
        self.rho_m = torch.tensor(997.0)
        self.rho_p = torch.tensor(1050.0)

        self.f = torch.tensor(1e6)
        self.k = 2*np.pi*self.f/self.c_m
        self.w = 2*np.pi*self.f

        self.a = torch.tensor(1e-5)
        self.V = (4/3)*np.pi*self.a**3

    def _complex(self, x): 
        """Cast tensor to complex64."""
        return x.to(torch.complex64)

    def spectral_derivs(self, p, dx):
        """
        Compute spatial derivatives of pressure using FFT (torch).

        Args:
            p (torch.Tensor): Complex pressure field (..., H, W).
            dx (float): Spatial grid spacing [m].

        Returns:
            tuple of torch.Tensor: (dpdx, dpdy).
        """
        *lead, H, W = p.shape
        p = p.to(torch.complex64)
        Nk = 2**int(np.ceil(np.log2(W))+1)

        kv = 2*np.pi*torch.fft.fftfreq(Nk, d=dx)
        kx, ky = torch.meshgrid(kv, kv, indexing='ij')

        P = torch.fft.fft2(p, [Nk, Nk])
        dpdx = torch.fft.ifft2(1j*self._complex(kx)*P, [Nk, Nk])
        dpdy = torch.fft.ifft2(1j*self._complex(ky)*P, [Nk, Nk])

        return dpdx[..., :W, :W], dpdy[..., :W, :W]

    def gorkov_potential(self, p, dx):
        """
        Compute Gor'kov potential and energy density maps (torch).

        Args:
            p (torch.Tensor): Complex harmonic pressure [Pa].
            dx (float): Spatial grid spacing [m].

        Returns:
            tuple: (U, p2_avg, v2_avg, vx2_avg, vy2_avg).
        """
        rho0, c0 = self.rho_m, self.c_m
        rho_p, c_p = self.rho_p, self.c_p
        omega = self.w
        p = p.to(torch.complex64)

        kappa0 = 1.0/(rho0*c0**2)
        kappap = 1.0/(rho_p*c_p**2)
        f1 = 1.0 - kappap/kappa0
        f2 = 2.0*(rho_p - rho0)/(2.0*rho_p + rho0)

        dpdx, dpdy = self.spectral_derivs(p, dx)
        vx = dpdx/(-1j*omega*rho0)
        vy = dpdy/(-1j*omega*rho0)

        p2_avg = 0.5*torch.abs(p)**2
        v2_avg = 0.5*(torch.abs(vx)**2 + torch.abs(vy)**2)

        U = self.V*(f1*0.5*kappa0*p2_avg - f2*0.75*rho0*v2_avg)

        return torch.real(U), p2_avg, v2_avg, 0.5*torch.abs(vx)**2, 0.5*torch.abs(vy)**2

    def laplacian(self, U, dx):
        """
        Compute Laplacian of Gor'kov potential using FFT (torch).

        Args:
            U (torch.Tensor): Gor'kov potential map [J].
            dx (float): Spatial grid spacing [m].

        Returns:
            tuple: (lap, Uxx, Uyy).
        """
        *lead, H, W = U.shape
        Nk = 2**int(np.ceil(np.log2(W))+1)

        kv = 2*np.pi*torch.fft.fftfreq(Nk, d=dx)
        kx, ky = torch.meshgrid(kv, kv, indexing='ij')

        Uhat = torch.fft.fft2(U, [Nk, Nk])
        Uxx = torch.fft.ifft2(-(self._complex(kx**2))*Uhat, [Nk, Nk])
        Uyy = torch.fft.ifft2(-(self._complex(ky**2))*Uhat, [Nk, Nk])

        Uxx = torch.real(Uxx[..., :W, :W])
        Uyy = torch.real(Uyy[..., :W, :W])

        return Uxx + Uyy, Uxx, Uyy
