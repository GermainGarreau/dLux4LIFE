import numpy as nnp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy import modeling
from scipy import special
import random

# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp

# Optimisation imports
import zodiax as zdx
import optax

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
from tqdm.notebook import tqdm

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72

import Aspheres

class PIAAforLIFE:
    """
    The PIAAforLIFE class combines all the tools to simulate the shape of mirrors to perform achromatic phase-induced amplitude apodization (PIAA), 
    and calculate the coupling of the apodized beam in a fiber with predefined mode field diameter (MFD).

    Notes
    -----

    Attributes
    ----------
    En_diam : float
        The diameter in meters of the entrance pupil with top-hat shape.
    Ex_diam : float
        The diameter in meters of the exit pupil. If apodized, the diameter is defined by 1/e in amplitude or 1/e2 in intensity.
    Apodized : bool
        True if the exit pupil should be apodized into a gaussian, False otherwise.
    n : int
        Sampling of the 1D-radius in the pupil plane.
    wave_mono : float
        The wavelength of the light in meters.
    mfd : float
        Mode field diameter of the fiber at wave_mono in meter. The diameter is defined by 1/e in amplitude or 1/e2 in intensity.
    focal_length : float
        Effective focal length of the imaging system in meters. If 0, the focal length is automatically calculated to maximize the theoretical coupling efficiency.
    f_psf_size : int
        Multiplication factor to define the size of the image plane in pixels, in units of psf size. 10 is better to plot the psf, 50 is better to calculate the coupling efficiency.
    wf_npix : int
        Number of pixels in the pupil wavefront
    psf_pixel_scale : int
        Size of the pixels in the image plane [microns]
    oversample : int
        Oversampling factor for the PSF
    n_asphere_term : int
        Number of modes to fit the surfaces of M1 and M2 with a mild asphere surface
    D_M1_M2 : float
        Distance between M1 and M2. Default is 5x the entrance pupil diameter
    pointing : float
        Pointing error [rad] for the injection in the fiber 
    shear : float
        Shear error [m] for the injection in the fiber
    unzoom : float
        Unzoom factor for the exit pupil. Allows to get most of the light from the gaussian beam
    """
    
    def __init__(self,
                 En_diam: float,
                 Ex_diam: float,
                 n: int,
                 wave_mono: float,
                 mfd: float,
                 focal_length: float = 0,
                 apodized: bool = True,
                 f_psf_size: int = 50,
                 wf_npix: int = 512,
                 psf_pixel_scale: int = 1,
                 oversample: int = 3,
                 n_asphere_term: int = 12,
                 pointing: float = 0,
                 shear: float = 0,
                 unzoom: float = 2,
                ):
        
        self.En_diam = En_diam
        self.Ex_diam = Ex_diam
        self.apodized = apodized
        self.n = n
        self.wave_mono = wave_mono
        self.mfd = mfd
        self.focal_length = focal_length
        self.f_psf_size = f_psf_size
        self.wf_npix = wf_npix
        self.psf_pixel_scale = psf_pixel_scale
        self.oversample = oversample
        self.n_asphere_term = n_asphere_term
        self.pointing = pointing
        self.shear = shear
        self.unzoom = unzoom

        self.En_rad = En_diam/2
        self.Ex_rad = Ex_diam/2
        self.D_M1_M2 = 5*self.En_diam

    def get_f(self):
        """
        Calculates the best focal length of the imaging system to maximize the coupling efficiency depending if the exit pupil is apodized or not.
        """
        self.focal_length = nnp.pi * self.Ex_diam * self.mfd / (4 * self.w_mono)
        if self.apodized == False:
            self.focal_length *= 1/1.1209
        return self.focal_length
    
    def r1_(self, t):
        """
        Radius function of the entrance pupil with regards to the entrance radius and the intensity fraction
        """
        return self.En_rad*nnp.sqrt(t) 

    def dr1_(self, t):
        """
        Derivative function of r1     
        """
        return self.En_rad/(2*nnp.sqrt(t))

    def f1_(self, t):
        """
        Brightness distribution of the entrance pupil
        """
        return 1/(nnp.pi*self.En_rad**2)*nnp.ones(len(t)) 

    def r2_(self, t):
        """
        Radius function of the exit pupil with regards to the entrance radius, the intensity fraction, the exit radius, and if the beam is apodized
        """
        if self.apodized == True:
            return nnp.sqrt(-0.5*(self.Ex_rad)**2*nnp.log(1-(self.r1_(t)/self.En_rad)**2))
        if self.apodized == False:
            return self.Ex_rad*nnp.sqrt(t)

    def dr2_(self, t):
        """
        Derivative function of r2
        """
        r_1 = self.r1_(t)
        if self.apodized == True:
            g = 1-(r_1/self.En_rad)**2
            dg_dt = -2/(self.En_rad**2)*r_1*self.dr1_(t)
            f = -0.5*self.Ex_rad**2*nnp.log(g)
            df_dt = -0.5*self.Ex_rad**2*dg_dt/g
            return df_dt * 1/(2*nnp.sqrt(f))
        if self.apodized == False:
            return self.Ex_rad/(2*nnp.sqrt(t))

    def f2_(self, t, r2):
        """
        Brightness distribution of the exit apodized pupil
        """
        if self.apodized == True:
            return nnp.exp(-2*r2**2/self.Ex_rad**2)*2/(nnp.pi*self.Ex_rad**2)     
        if self.apodized == False:
            return 1/(nnp.pi*self.Ex_rad**2)*nnp.ones(len(t))

    def dM_dr(self, M, t):
        """
        Differential equation function for M1 and M2
        """
        M1, M2 = M
        A = self.r2_(t) - self.r1_(t)
        B = M2 - M1
        if A == 0:
            return [0, 0]
        else:
            a = (nnp.sqrt(A**2+B**2)-B)/A
            #We obtain dM1/dr1 and dM2/dr2, we need to multiply them with dr1/dt and dr2/dt to obtain dM1/dt and dM2/dt, which is necessary for odeint
            return [a*self.dr1_(t), a*self.dr2_(t)]

    def get_M1_M2(self):
        """
        Calculate the shape of the mirrors M1 and M2 to perform the apodization.
        """
        
        #Fraction of total intensity in the pupil plane
        t = nnp.linspace(0, 1, self.n)[:-1]

        r1, r2 = self.r1_(t), self.r2_(t)
        f1, f2 = self.f1_(t), self.f2_(t, r2)

        #Initial values for M1 and M2
        M_0 = [0, self.D_M1_M2]  #Initial values are the absolute positions of the mirrors, including their distance, default distance is 5x entrance pupil diameter    

        # Solving of the differential equation
        results = nnp.array(integrate.odeint(self.dM_dr, M_0, t))
        M1_results, M2_results = results[:,0], results[:,1]
        M2_results = M2_results - M_0[1]

        # Save mirror shapes
        self.r1, self.r2 = r1, r2
        self.M1, self.M2 = M1_results, M2_results

    def Mild_asphere_fit(self):
        """
        Fit the mirror shapes M1 and M2 with mild asphere model using the n_asphere_term number of modes
        """
        # Fit of M1 and M2, and definition of the M1 and M2 objects
        piaa_profile_1 = Aspheres.MildAsphere.from_target_sag(
                        target_rhos = self.r1,
                        target_sags = self.M1,
                        n_terms = self.n_asphere_term,
                        scale=1,
                    )
        
        piaa_profile_2=Aspheres.MildAsphere.from_target_sag(
                        target_rhos = self.r2,
                        target_sags = -self.M2,
                        n_terms =  self.n_asphere_term,
                        scale=1,
                    )
        self.piaa_profile_1 = piaa_profile_1
        self.piaa_profile_2 = piaa_profile_2

    def do_PIAA(self):
        """
        Core function, which performs the PIAA using the calculated M1 and M2
        """
        
        # Entrance pupil definition
        coords = dlu.pixel_coords(self.wf_npix, self.En_diam)
        aperture = dlu.circle(coords, 0.5 * self.En_diam)

        # Define focal length for the imaging
        if self.focal_length == 0:
            self.focal_length = nnp.pi * self.Ex_diam * self.mfd / (4 * self.wave_mono)
            if self.apodized == False:
                self.focal_length *= 1/1.1209

        # Define our image plane properties before apodization
        self.psf_size_en = 1.22 * self.focal_length * self.wave_mono / self.En_diam
        self.psf_en_npix = int(self.f_psf_size*self.psf_size_en/(self.psf_pixel_scale*1e-6))  # Number of pixels in the PSF

        # Define the optics object for the entrance pupil
        layers_en = [("aperture", dl.layers.TransmissiveLayer(transmission=aperture, normalise=True))]
        optics_entrance = dl.LayeredOpticalSystem(wf_npixels=self.wf_npix, diameter=self.En_diam, layers=layers_en)
        optics_entrance_psf = dl.CartesianOpticalSystem(self.wf_npix, self.En_diam, layers_en, self.focal_length, self.psf_en_npix, self.psf_pixel_scale, self.oversample)

        # Define our image plane properties after apodization
        if self.apodized == False: 
            self.psf_size_ex = 1.22 * self.focal_length * self.wave_mono / self.Ex_diam
        if self.apodized == True:  #1/e2 size of a gaussian
            self.psf_size_ex = 2*self.focal_length*self.wave_mono/(np.pi*self.Ex_diam)
        self.psf_ex_npix = int(self.f_psf_size*self.psf_size_ex/(self.psf_pixel_scale*1e-6))  # Number of pixels in the PSF

        # Extract the entrance pupil and psf electric field
        pupil_entrance = optics_entrance.propagate_mono(self.wave_mono, return_wf = True)
        self.pupil_entrance = pupil_entrance.real
        psf_entrance = optics_entrance_psf.propagate_mono(self.wave_mono, return_wf = True)
        self.psf_entrance = psf_entrance.real

        # Exit pupil definition
        self.wf_npix_exit = self.unzoom * self.wf_npix
        self.diam_exit_pupil = self.unzoom * self.En_diam
        coords = dlu.pixel_coords(self.wf_npix_exit, self.diam_exit_pupil)
        aperture = dlu.circle(coords, 0.5 * self.En_diam)

        # Construct the PIAA model from the fitted M1 and M2
        PIAA = Aspheres.PIAAset(
                piaa_optic_1=Aspheres.AsphericMirror(
                    self.piaa_profile_1,
                    coords=coords,
                ),
                piaa_optic_2=Aspheres.AsphericMirror(
                    self.piaa_profile_2,
                    coords=coords,
                ),
                distance=self.D_M1_M2,
            )
        
        layers_ex = [("aperture", dl.TransmissiveLayer(aperture, normalise=True)), PIAA,]

        # Define the optics object for the exit pupil
        optics_exit = dl.LayeredOpticalSystem(wf_npixels=self.wf_npix_exit, diameter=self.diam_exit_pupil, layers=layers_ex,)
        optics_exit_psf = dl.CartesianOpticalSystem(self.wf_npix_exit, self.diam_exit_pupil, layers_ex, self.focal_length, self.psf_ex_npix, self.psf_pixel_scale, self.oversample)

        # Extract the exit pupil and psf electric field
        pupil_exit = optics_exit.propagate_mono(self.wave_mono, return_wf = True)
        self.pupil_exit = pupil_exit.real
        self.pupil_exit_phase = pupil_exit.phase
        psf_exit = optics_exit_psf.propagate_mono(self.wave_mono, return_wf = True)
        self.psf_exit = psf_exit.real     

        # Check for vignetting
        En_sum_pupil = np.sum(self.pupil_entrance**2)
        Ex_sum_pupil = np.sum(self.pupil_exit**2)
        En_sum_psf = np.sum(self.psf_entrance**2)
        Ex_sum_psf = np.sum(self.psf_exit**2)

        #if En_sum_pupil <= 0.95 or Ex_sum_pupil <= 0.95 or En_sum_psf <= 0.95 or Ex_sum_psf <= 0.95:
            #print("Warning: you may have some vignetting. Check with self.print_intensity()")

        # Cross sections of the different outputs
        self.pupil_exit_cs = self.pupil_exit[int(0.5*self.wf_npix_exit),:]
        self.pupil_exit_phase_cs = self.pupil_exit_phase[int(0.5*self.wf_npix_exit),:]
        self.psf_exit_cs = self.psf_exit[int(0.5*self.psf_ex_npix*self.oversample),:]
        self.pupil_entrance_cs = self.pupil_entrance[int(0.5*self.wf_npix),:]
        self.psf_entrance_cs = self.psf_entrance[int(0.5*self.psf_en_npix*self.oversample),:]

    def plot_im(self):
        """
        Plots the 2D images of the entrance pupil, exit pupil, their PSFs, and the wavefront of the exit pupil
        """
        #Pupil plane
        plt.figure(figsize=(14, 4))
        #plt.suptitle("Pupil plane")
        
        plt.subplot(1, 3, 1)
        plt.title("Before PIAA")
        plt.imshow(self.pupil_entrance, norm=colors.Normalize(vmin=np.min(self.pupil_entrance), vmax=np.max(self.pupil_entrance)))
        plt.colorbar(label="Amplitude")
        
        plt.subplot(1, 3, 2)
        plt.title("After PIAA")
        plt.imshow(self.pupil_exit, norm=colors.Normalize(vmin=np.min(self.pupil_exit), vmax=np.max(self.pupil_exit)))
        plt.colorbar(label="Amplitude")
        
        plt.subplot(1, 3, 3)
        plt.title("After PIAA")
        plt.imshow(self.pupil_exit_phase)
        plt.colorbar(label="Phase [rad]")
        
        plt.tight_layout()
        plt.savefig("images/2D_im_pupil.png", dpi=300, bbox_inches = "tight")
        plt.show()
        
        #Image plane
        plt.figure(figsize=(10, 4))
        #plt.suptitle("PSF")
        
        plt.subplot(1, 2, 1)
        plt.title("PSF Before PIAA")
        plt.imshow(self.psf_entrance, norm=colors.Normalize(vmin=np.min(self.psf_entrance), vmax=np.max(self.psf_entrance)))
        plt.colorbar(label="Amplitude")
        
        plt.subplot(1, 2, 2)
        plt.title("PSF After PIAA")
        plt.imshow(self.psf_exit, norm=colors.Normalize(vmin=np.min(self.psf_exit), vmax=np.max(self.psf_exit)))
        plt.colorbar(label="Amplitude")
        
        plt.tight_layout()
        plt.savefig("images/2D_im_psf.png", dpi=300, bbox_inches = "tight")
        plt.show()

    def plot_cross_sec(self):
        """
        Plots the 1D profile of the entrance pupil, exit pupil, their PSFs, and the wavefront of the exit pupil
        """
        # X-axes in length for the image and pupil plane
        x_axis_pupil = np.linspace(0, self.wf_npix, self.wf_npix)*self.En_diam/self.wf_npix - self.En_diam/2
        x_axis_pupil_exit = np.linspace(0, self.wf_npix_exit, self.wf_npix_exit)*self.diam_exit_pupil/self.wf_npix_exit - self.diam_exit_pupil/2
        x_axis_psf_en = np.linspace(0, self.psf_en_npix, self.psf_en_npix*self.oversample)*self.psf_pixel_scale - self.psf_en_npix*self.psf_pixel_scale/2
        x_axis_psf_en *= 1e-6
        x_axis_psf_ex = np.linspace(0, self.psf_ex_npix, self.psf_ex_npix*self.oversample)*self.psf_pixel_scale - self.psf_ex_npix*self.psf_pixel_scale/2
        x_axis_psf_ex *= 1e-6
        
        # Fit with a gaussian with the defined waist
        if self.apodized == True:
            A=np.max(self.pupil_exit_cs**2)
            G_fit = A*np.exp(-2*self.r2**2/(self.Ex_rad)**2)
        
        plt.figure(figsize=(10, 4))
        #plt.suptitle("Beam profiles")
        plt.subplot(1, 2, 1)
        plt.title("Wavefront amplitude")
        plt.plot(x_axis_pupil*1000, self.pupil_entrance_cs**2, label='before PIAA')
        plt.plot(x_axis_pupil_exit*1000, self.pupil_exit_cs**2, label='after PIAA')
        if self.apodized == True:
            plt.plot(self.r2*1000, G_fit, color="k", label="Requested")
        plt.ylabel('Intensity')
        plt.xlabel("Radii [mm]")
        plt.xlim(0, np.max(x_axis_pupil_exit*1000))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Wavefront phase")
        plt.plot(x_axis_pupil_exit*1e3, self.pupil_exit_phase_cs, label='after PIAA')
        plt.xlim(0, np.max(x_axis_pupil_exit*1e3))
        plt.yscale("linear")
        plt.ylabel('Phase [rad]')
        plt.xlabel("Radii [mm]")   

        plt.tight_layout()
        plt.savefig("images/1D_sections.png", dpi=300, bbox_inches = "tight")
        plt.show()

        plt.figure(figsize=(5, 4))
        plt.title("PSF")
        plt.plot(x_axis_psf_en*1e6, self.psf_entrance_cs**2, label='before PIAA') 
        plt.plot(x_axis_psf_ex*1e6, self.psf_exit_cs**2, label='after PIAA')
        #plt.vlines(self.psf_size_ex*1e6, 1e-15, 1e-0, ls="--", color="k", label="Requested")
        plt.yscale("log")
        plt.xlim(0, min([np.max(x_axis_psf_ex), np.max(x_axis_psf_en)])*1e6)
        plt.ylim(np.min(self.psf_entrance_cs**2), np.max(self.psf_exit_cs**2))
        plt.ylabel('Intensity')
        plt.xlabel("Radii [microns]")
        plt.legend()
        plt.tight_layout()
        plt.savefig("images/1D_psf_section.png", dpi=300, bbox_inches = "tight")
        plt.show()


    def print_intensity(self):
        """
        Print the total intensity of the entrance pupil, exit pupil, and their PSFs
        """
        En_sum_pupil = np.sum(self.pupil_entrance**2)
        Ex_sum_pupil = np.sum(self.pupil_exit**2)
        En_sum_psf = np.sum(self.psf_entrance**2)
        Ex_sum_psf = np.sum(self.psf_exit**2)
        print("Entrance pupil = ", En_sum_pupil)
        print("Exit pupil = ", Ex_sum_pupil)
        print("Entrance PSF = ", En_sum_psf)
        print("Exit PSF = ", Ex_sum_psf)

    def Ceff_gaussian(self):
        """
        Calculate the coupling efficiency between a Gaussian beam and a single-mode fiber.
        """
        w1 = self.mfd/2 # mode field radius (m)
        w0 = self.Ex_rad # beam radius (m)
        factor = 4 * (self.focal_length * np.pi * w0 * w1 * self.wave_mono)**2 / (np.pi**2 * w0**2 * w1**2 + self.focal_length**2 * self.wave_mono**2)**2
        exponent = - 2 * np.pi * (self.shear**2 * np.pi * w1**2 + self.focal_length**2 * np.pi * w0**2 * self.pointing**2) / (np.pi**2 * w0**2 * w1**2 + self.focal_length**2 * self.wave_mono**2)
        return factor * np.exp(exponent)

    def get_Ceff(self):
        """
        Calculates the coupling efficiency of the exit PSF from PIAA with the mode of the fiber defined by mfd. The mode of the fiber is assumed ot be an ideal centrosymmetric 2D gaussian.

        Returns
        ----------
        Ceff_i, Ceff_airy, Ceff_dLux :
            Ceff_i corresponds to the ideal coupling efficiency calculated from the focal length, beam diameter, mfd, and wavelength, regardless of apodization.
            Ceff_airy corresponds to the ideal coupling efficiency between an airy disk modelled by the different parameters of the system, and the fiber.
            Ceff_dLux corresponds to the coupling efficiency between the output PSF from dLux, and the fiber. If apodization = False, this corresponds to Ceff_airy.
        """
        # Construct the 2D grid for the fiber 2D model
        x_axis_psf_ex = np.linspace(0, self.psf_ex_npix, self.psf_ex_npix*self.oversample)*self.psf_pixel_scale - self.psf_ex_npix*self.psf_pixel_scale/2
        x_axis_psf_ex *= 1e-6
        X, Y = np.meshgrid(x_axis_psf_ex, x_axis_psf_ex)

        # Model of fiber mode
        Mode_std = float(self.mfd/4)
        Fiber_model = modeling.functional_models.Gaussian2D(amplitude = float(1/(Mode_std*np.sqrt(2*np.pi))), x_stddev = Mode_std, y_stddev = Mode_std)
        Fiber_2D = Fiber_model(X,Y) # in Intensity

        # PSF from dLux (apodized or not)
        PSF_2D_dLux = self.psf_exit #in Amplitude

        # Model for Airy disk PSF in 2D
        R = np.sqrt(X**2 + Y**2) # New grid defined as radius 
        X_array = np.pi * R * self.Ex_diam / self.wave_mono / self.focal_length
        PSF_2D = special.j1(X_array)/X_array
        PSF_2D = PSF_2D.at[nnp.isnan(PSF_2D)].set(0.5)  #in Amplitude

        # Ideal coupling efficiency
        NA = self.Ex_diam/(2*self.focal_length)
        beta = np.pi*self.mfd/self.wave_mono*NA/2
        if self.apodized == False:
            Ceff_i = 2*((np.exp(-beta**2)-1)/beta)**2
        if self.apodized == True:
            Ceff_i = self.Ceff_gaussian()
        
        # Coupling efficiency from Airy disk model
        E_psf, E_g = PSF_2D, np.sqrt(Fiber_2D)
        Ceff_airy = (np.sum(E_psf * E_g))**2/(np.sum(E_psf**2) * np.sum(E_g**2)) 
        
        # Coupling efficiency from dLux
        E_psf, E_g = self.psf_exit, np.sqrt(Fiber_2D)
        Ceff_dLux = (np.sum(E_psf * E_g))**2/(np.sum(E_psf**2) * np.sum(E_g**2)) 

        return Ceff_i, Ceff_airy, Ceff_dLux

    def plot_fiber_modes(self):
        """
        1D and 2D plots of the fiber mode, the psf from dLux, and the equivalent Airy disk for comparison
        """
        # Construct the 2D grid for the fiber 2D model
        x_axis_psf_ex = np.linspace(0, self.psf_ex_npix, self.psf_ex_npix*self.oversample)*self.psf_pixel_scale - self.psf_ex_npix*self.psf_pixel_scale/2
        x_axis_psf_ex *= 1e-6
        X, Y = np.meshgrid(x_axis_psf_ex, x_axis_psf_ex)

        # Model of fiber mode
        Mode_std = float(self.mfd/4)
        Fiber_model = modeling.functional_models.Gaussian2D(amplitude = float(1/(Mode_std*np.sqrt(2*np.pi))), x_stddev = Mode_std, y_stddev = Mode_std)
        Fiber_1D = Fiber_model(x_axis_psf_ex, 0) #Intensity
        Fiber_2D = Fiber_model(X,Y) #Intensity

        # PSF from dLux (apodized or not)
        PSF_1D_dLux = self.psf_exit_cs**2/nnp.trapezoid(self.psf_exit_cs**2, x_axis_psf_ex) #Intensity
        PSF_2D_dLux = self.psf_exit #Amplitude

        # Model for Airy disk in 1D
        x = np.pi * x_axis_psf_ex * self.Ex_diam / self.wave_mono / self.focal_length
        PSF_model = (special.j1(x)/x)**2
        PSF_model *= 4*np.max(PSF_1D_dLux)  
        PSF_1D = PSF_model   #Intensity

        # Model for Airy disk PSF in 2D
        R = np.sqrt(X**2 + Y**2) # New grid defined as radius 
        X_array = np.pi * R * self.Ex_diam / self.wave_mono / self.focal_length
        PSF_2D = special.j1(X_array)/X_array
        PSF_2D = PSF_2D.at[nnp.isnan(PSF_2D)].set(0.5)  #in Amplitude

        # Plot of the 1D profiles
        plt.figure(figsize=(10, 4))
        #plt.subplot(1, 2, 1)
        plt.plot(x_axis_psf_ex*1e6, PSF_1D_dLux, label="PSF dLux",color="tab:orange")
        plt.plot(x_axis_psf_ex*1e6, Fiber_1D, color="k",ls="dotted", label="Fiber mode")
        plt.plot(x_axis_psf_ex*1e6, PSF_1D, color="tab:blue", label="Airy disk")
        #plt.vlines(mfd/2*1e6, 1e-2, np.max(Fiber_1D), color="tab:orange", ls="--")
        #plt.vlines(psf_size_exit*1e6, np.min(PSF_1D_dLux), np.max(Fiber_1D), ls="--", color="tab:blue")
        #plt.hlines(np.max(Fiber_1D)*np.exp(-2), 0, np.max(x_axis*1e6)/2, color="tab:orange", ls="--")
        plt.xlabel("Radii [microns]")
        plt.ylabel("Intensity")
        plt.xlim(0, np.max(x_axis_psf_ex*1e6))
        plt.ylim(np.min(PSF_1D_dLux), max([np.max(PSF_1D_dLux),np.max(Fiber_1D)]))
        plt.yscale("log")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_axis_psf_ex*1e6, PSF_1D_dLux, label="PSF dLux")
        plt.plot(x_axis_psf_ex*1e6, Fiber_1D, color="tab:orange", label="Fiber mode")
        plt.plot(x_axis_psf_ex*1e6, PSF_1D, color="k",ls="dotted", label="Airy disk model")
        #plt.vlines(mfd/2*1e6, 1e-2, np.max(Fiber_1D), color="tab:orange", ls="--")
        #plt.vlines(psf_size_exit*1e6, np.min(PSF_1D_dLux), np.max(Fiber_1D), ls="--", color="tab:blue")
        #plt.hlines(np.max(Fiber_1D)*np.exp(-2), 0, np.max(x_axis*1e6)/2, color="tab:orange", ls="--")
        plt.xlabel("Radii [microns]")
        plt.ylabel("Intensity")
        plt.xlim(0, np.max(x_axis_psf_ex*1e6))
        plt.ylim(np.min(PSF_1D_dLux), max([np.max(PSF_1D_dLux),np.max(Fiber_1D)]))
        plt.yscale("linear")
        plt.legend()
        plt.savefig("images/1D_fiber_psf.png", dpi=300, bbox_inches = "tight")
        plt.show()

        # PLot of the 2D images
        plt.figure(figsize=(14, 4))
        #plt.suptitle("Modes at the fiber input")
        
        plt.subplot(1, 3, 1)
        plt.title("Fiber mode")
        plt.imshow(Fiber_2D**0.5, norm=colors.Normalize(vmin=np.min(Fiber_2D**0.5), vmax=np.max(Fiber_2D**0.5)))
        plt.colorbar(label="Amplitude")
        
        plt.subplot(1, 3, 2)
        plt.title("Airy disk")
        plt.imshow(PSF_2D, norm=colors.Normalize(vmin=np.min(PSF_2D), vmax=np.max(PSF_2D)))
        plt.colorbar(label="Amplitude")
        
        plt.subplot(1, 3, 3)
        plt.title("DLux output")
        plt.imshow(PSF_2D_dLux, norm=colors.Normalize(vmin=np.min(PSF_2D_dLux), vmax=np.max(PSF_2D_dLux)))
        plt.colorbar(label="Amplitude")
        
        plt.tight_layout()
        plt.savefig("images/2D_fiber_psf.png", dpi=300, bbox_inches = "tight")
        plt.show()

    def M_slope_err(self, error_mag):
        """
        Calculate the coupling efficiency for perturbed shapes of M1 and M2. The perturbation here is proportionnal to the sag of the mirrors, and results in an error in the slope.

        Attributes
        ----------
        error_mag : float
            Maximum error in meter between the ideal shapes and the perturbed shape

        Returns
        ----------
        Ceff_low, Ceff_high :
            The coupling efficiencies from the dLux perturbed outputs, Ceff_low corresponding to the result of a negative sag perturbation, and Ceff_high a positive sag perturbation.
        """
        # Reassign M1 and M2 in the PIAA to their optimal values in case they were changed
        self.get_M1_M2()
        
        # Implement slope perturbation on the mirror surfaces
        pert_f = error_mag/np.max(np.abs(self.M1))
        M1_pert_low, M1_pert_high = self.M1*(1-pert_f), self.M1*(1+pert_f)
        pert_f = error_mag/np.max(np.abs(self.M2))
        M2_pert_low, M2_pert_high = self.M2*(1-pert_f), self.M2*(1+pert_f)

        # Implement the lower sag in the PIAA and calculate the coupling efficiency
        self.M1, self.M2 = M1_pert_low, M2_pert_low
        self.Mild_asphere_fit()
        self.do_PIAA()
        Ceff_low = self.get_Ceff()[2]

        # Implement the lower sag in the PIAA and calculate the coupling efficiency
        self.M1, self.M2 = M1_pert_high, M2_pert_high
        self.Mild_asphere_fit()
        self.do_PIAA()
        Ceff_high = self.get_Ceff()[2]

        # Reassign again the original unperturbed values for M1 and M2
        self.get_M1_M2()
        self.Mild_asphere_fit()

        return Ceff_low, Ceff_high


    def M_spot_err(self, err_mag, spot_size):
        """
        Simulate centrosymmetric manufacturing errors on the surface of M1 and M2.

        Attributes
        ----------
        err_mag : float
            Standard deviation in meter of the error spots added to the ideal surface.
        spot_size : float
            Width of the centrosymmetric error spots in meter
        """
        # Reassign M1 and M2 in the PIAA to their optimal values in case they were changed
        self.get_M1_M2()

        # Calculate random error spots that are centrosymmetric
        error_M1 = nnp.zeros(len(self.M1))
        error_M2 = nnp.zeros(len(self.M2))

        # For M1
        k, ref = 0, 0
        while k < len(self.r1)-1:
            err = random.gauss(0, err_mag)
            while self.r1[k] - ref < spot_size and k < len(self.r1)-1:
                error_M1[k] = err
                k += 1
            ref = self.r1[k]

        # For M2
        k, ref = 0, 0
        while k < len(self.r2)-1:
            err = random.gauss(0, err_mag)
            while self.r2[k] - ref < spot_size and k < len(self.r2)-1:
                error_M2[k] = err
                k += 1
            ref = self.r2[k]

        # Add the error to the surfaces of M1 and M2
        self.M1 += error_M1
        self.M2 += error_M2