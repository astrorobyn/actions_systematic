import numpy as np
import agama
import utilities as ut

import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

textwidth = 7.10000594991
columnwidth = 3.35224200913

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

Rcut = 0.2
xrng = [0.7, 1.15]
yrng = [0.0, 0.005]
res = [128, 128]

off1 = np.array([0, 0, 100./1000.0, 0, 0, 0])
# off2 = np.array([0, 0, 100./1000.0, 0, 0, 0])

vmin=10
vmax=50

vmin_diff = -1
vmax_diff = 1

agama.setUnits(mass=1, length=1, velocity=1)
G = 4.30092e-06

bulge = agama.Potential(type='Dehnen', gamma=1, mass=5E9, scaleRadius=1.0)
nucleus = agama.Potential(type='Dehnen', gamma=1, mass=1.71E09, scaleRadius=0.07)
disk = agama.Potential(type='MiyamotoNagai', mass=6.80e+10, scaleRadius=3.0, scaleHeight=0.28)
halo = agama.Potential(type='NFW', mass=5.4E11, scaleRadius=15.62)
mwpot = agama.Potential(bulge, nucleus, disk, halo)

R0 = 8.135
z0 = 20.3/1000.
v0 = np.sqrt(-R0*mwpot.force([R0, 0, 0])[0])
J0 = R0*v0

Rd = 2.4
q = 0.45
Sigma0_thin_Binney = 36.42
Sigma0_thick_Binney = 4.05

Sigma0_thin = np.exp(R0/Rd)*Sigma0_thin_Binney
Sigma0_thick = np.exp(R0/Rd)*Sigma0_thick_Binney
# Sigma0_thin  = Sigma0_thin_Binney
# Sigma0_thick = Sigma0_thick_Binney

coefJr = 1
coefJz = 1

hdisk_thin = 0.36
hdisk_thick = 1.0

Rsigmar = Rd/q

sigmar0_thin_binney = 27.
sigmar0_thick_binney = 48.

sigmar0_thin = sigmar0_thin_binney * np.exp(R0/Rsigmar)
sigmar0_thick = sigmar0_thick_binney * np.exp(R0/Rsigmar)

sigmamin_thin = 0.05*sigmar0_thin
sigmamin_thick = 0.05*sigmar0_thick
Jmin = 0.05*J0

df_thin = agama.DistributionFunction(type='QuasiIsothermal', potential=mwpot, coefJr=coefJr, coefJz=coefJz, 
                                     Sigma0=Sigma0_thin, Rdisk=Rd, Hdisk=hdisk_thin, sigmar0=sigmar0_thin, 
                                     Rsigmar=Rsigmar, sigmamin=sigmamin_thin, Jmin=Jmin)

df_thick = agama.DistributionFunction(type='QuasiIsothermal', potential=mwpot, coefJr=coefJr, coefJz=coefJz, 
                                      Sigma0=Sigma0_thick, Rdisk=Rd, Hdisk=hdisk_thick, sigmar0=sigmar0_thick, 
                                      Rsigmar=Rsigmar, sigmamin=sigmamin_thick, Jmin=Jmin)

df = agama.DistributionFunction(df_thin, df_thick)
# df = df_thin
gm = agama.GalaxyModel(mwpot, df)

try:
    f = h5.File('posvel_mass.h5', 'r')
    posvel = np.array(f['posvel'])
    mass = np.array(f['mass'])
    f.close()
except:
    posvel, mass = gm.sample(int(4E7))
    f = h5.File('posvel_mass.h5', 'w')
    f.create_dataset('posvel', data=posvel)
    f.create_dataset('mass', data=mass)
    f.close()

Rstar = np.linalg.norm(posvel[:,:2], axis=1)
zstar = posvel[:,2]

Rdiff = np.subtract(Rstar, R0)
zdiff = np.subtract(zstar, z0)
diff = np.sqrt(np.add(np.square(Rdiff), np.square(zdiff))) 

keys = np.where(diff < Rcut)[0]
print(keys[:10])
zoff = np.random.uniform(0, 100, len(keys))/1000.0

off2 = np.array([[0, 0, z, 0, 0, 0] for z in zoff])

posvel, mass = posvel[keys], mass[keys]

posvel_off1 = np.subtract(posvel, off1)
posvel_off2 = np.subtract(posvel, off2)

actions = gm.af(posvel)
actions[:,[1, 2]] = actions[:,[2, 1]] # transpose Jphi and Jz

actions_off1 = gm.af(posvel_off1)
actions_off1[:,[1, 2]] = actions_off1[:,[2, 1]] # transpose Jphi and Jz

actions_off2 = gm.af(posvel_off2)
actions_off2[:,[1, 2]] = actions_off2[:,[2, 1]] # transpose Jphi and Jz

fig, (ax, ax_off1, ax_off2) = plt.subplots(1, 3, sharey=True, figsize=(10,4))
rng = [xrng, yrng]

heatmap, xedges, yedges = np.histogram2d(actions[:,1]/J0, actions[:,2]/J0, bins=res, range=rng)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax.imshow(heatmap.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax, aspect=1.5*(xrng[1]-xrng[0])/(yrng[1]-yrng[0]))
ax.set_title('no offset')

heatmap, xedges, yedges = np.histogram2d(actions_off1[:,1]/J0, actions_off1[:,2]/J0, bins=res, range=rng)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax_off1.imshow(heatmap.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax, aspect=1.5*(xrng[1]-xrng[0])/(yrng[1]-yrng[0]))
ax_off1.set_title(r'$z_{\text{offset}} = 100\,\text{pc}$')

heatmap, xedges, yedges = np.histogram2d(actions_off2[:,1]/J0, actions_off2[:,2]/J0, bins=res, range=rng)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax_off2.imshow(heatmap.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax, aspect=1.5*(xrng[1]-xrng[0])/(yrng[1]-yrng[0]))
fig.colorbar(im, label='number')
ax_off2.set_title(r'$z_{\text{offset}} = \mathcal{N}(0, (100\,\text{pc})^2)$')

ax.set_xlabel(r'$J_{\phi} / J_0$')
ax_off1.set_xlabel(r'$J_{\phi} / J_0$')
ax_off2.set_xlabel(r'$J_{\phi} / J_0$')
ax.set_ylabel(r'$J_z / J_0$')

ax.set_xlim(xrng)
ax_off1.set_xlim(xrng)
ax_off2.set_xlim(xrng)
ax.set_ylim(yrng)

fig.tight_layout()
fig.savefig('action_df.pdf')



# now plot the difference between the two

fig, (ax_off1, ax_off2) = plt.subplots(1, 2, sharey=True, figsize=(6,4))
# fig.set_size_inches(10, 4)

heatmap, xedges, yedges = np.histogram2d(actions[:,1]/J0, actions[:,2]/J0, bins=res, range=rng)
heatmap_off1, xedges, yedges = np.histogram2d(actions_off1[:,1]/J0, actions_off1[:,2]/J0, bins=res, range=rng)
heatmap_off2, xedges, yedges = np.histogram2d(actions_off2[:,1]/J0, actions_off2[:,2]/J0, bins=res, range=rng)

heatmap_diff1 = (heatmap_off1 - heatmap)/heatmap
heatmap_diff2 = (heatmap_off2 - heatmap)/heatmap

heatmap_diff1.T[heatmap_diff1.T == 0] = np.nan
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax_off1.imshow(heatmap_diff1.T, extent=extent, origin='lower', cmap='seismic', vmin=vmin_diff, vmax=vmax_diff, aspect=1.5*(xrng[1]-xrng[0])/(yrng[1]-yrng[0]))
ax_off1.set_title(r'$z_{\text{offset}} = 100\,\text{pc}$')

heatmap_diff2.T[heatmap_diff2.T == 0] = np.nan
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax_off2.imshow(heatmap_diff2.T, extent=extent, origin='lower', cmap='seismic', vmin=vmin_diff, vmax=vmax_diff, aspect=1.5*(xrng[1]-xrng[0])/(yrng[1]-yrng[0]))
# fig.colorbar(im, label='number')
ax_off2.set_title(r'$z_{\text{offset}} = \mathcal{N}(0, (100\,\text{pc})^2)$')

# fig.colorbar(im, fraction=0.06, pad=0.15)
# fig.colorbar(im, shrink=.5, pad=.2, aspect=10)
fig.colorbar(im, label='fractional difference')

ax.set_xlabel(r'$J_{\phi} / J_0$')
ax_off1.set_xlabel(r'$J_{\phi} / J_0$')
ax_off2.set_xlabel(r'$J_{\phi} / J_0$')
ax.set_ylabel(r'$J_z / J_0$')

ax.set_xlim(xrng)
ax_off1.set_xlim(xrng)
ax_off2.set_xlim(xrng)
ax.set_ylim(yrng)

fig.tight_layout()
fig.savefig('action_df_diff.pdf')

# fig, ax = plt.subplots(1,1)
# cylpos = ut.coordinate.get_positions_in_coordinate_system(posvel[:,:3])
# cylvel = ut.coordinate.get_velocities_in_coordinate_system(posvel[:,3:], posvel[:,:3])

# ax.scatter(cylvel[:,0], cylvel[:,2], s=0.1)

# ax.set_xlabel('vR [km/s]')
# ax.set_ylabel('vphi [km/s]')

# fig.tight_layout()
# fig.savefig('vR_vphi.png')

