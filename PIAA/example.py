# %%
import dLux as dl
import dLux.utils as dlu
import Aspheres
import matplotlib.pyplot as plt
import jax.numpy as np

# %%
wf_npix = 1024
wf_diam = 10e-3
aperture_diameter = 8e-3
wavels = np.array([10e-6])

piaa_distance = 10e-3

n_asphere_terms = 8

# %%
coords = dlu.pixel_coords(wf_npix, wf_diam)
aperture = dlu.circle(coords, radius=aperture_diameter / 2)
plt.figure()
plt.imshow(aperture)


# %%

# if you already know what your sag is (roughly) meant to be,
# use Aspheres.MildAsphere.fit_from_sag

layers = [
    dl.TransmissiveLayer(aperture, normalise=True),
    Aspheres.PIAAset(
        piaa_optic_1=Aspheres.AsphericMirror(
            asphere_profile=Aspheres.MildAsphere(
                aperture_radius=aperture_diameter / 2,
                sag_at_edge=0.0,
                n_terms=n_asphere_terms,
                scale=1e-6,
            ),
            coords=coords,
        ),
        piaa_optic_2=Aspheres.AsphericMirror(
            asphere_profile=Aspheres.MildAsphere(
                aperture_radius=aperture_diameter / 2,
                sag_at_edge=0.0,
                n_terms=n_asphere_terms,
                scale=1e-6,
            ),
            coords=coords,
        ),
        distance=piaa_distance,
    ),
    # if you want to model to the focal plane, add a dl.MFT layer here 
]

optics = dl.LayeredOpticalSystem(wf_npixels=wf_npix, diameter=wf_diam, layers=layers)
source = dl.PointSource(wavelengths=wavels)

# %%
plt.figure()
plt.imshow(optics.model(source))
plt.show()
