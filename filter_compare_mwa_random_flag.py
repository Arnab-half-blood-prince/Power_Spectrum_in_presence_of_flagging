## introduce flag, FFT, then use CLEAN and estimate PS ##
from functions import *
#import uvtools.dspec as dspec
try:
    import pyfftw
    fft = lambda *args, **kargs: pyfftw.interfaces.numpy_fft.fft(*args, threads=NUM_POOL, **kargs)
    fft2 = lambda *args, **kargs: pyfftw.interfaces.numpy_fft.fft2(*args, threads=NUM_POOL, **kargs)
    ifft2 = lambda *args, **kargs: pyfftw.interfaces.numpy_fft.ifft2(*args, threads=NUM_POOL, **kargs)
    # print("Using FFTW with %s threads" % NUM_POOL)
except Exception:
    fft = np.fft.fft
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
    print("Warning: using slower numpy fft's functions. Consider installing pyfftw.")



# HI line frequency
freq21cm = 1420.405751  # [MHz]
# Adopted cosmology
H0 = 71.0  # [km/s/Mpc]
OmegaM0 = 0.27
cosmo = Planck15


#############################
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s:%(lineno)d] %(message)s")
logger = logging.getLogger()

#### read image cube ###

f = fits.open('eor_diffuse_cube.fits') 
cube_data = f[0].data
header_data = f[0].header
logger.info("Cube shape: %dx%dx%d" % cube_data.shape)
bunit = header_data.get("BUNIT", "???")
logger.info("Cube data unit: %s" % bunit)
if bunit.upper() not in ["K", "KELVIN", "MK"]:
     logger.warning("input cube in unknown unit: %s" % bunit)


pixelsize = get_pixelsize(header_data) # in arcsec
logger.info("Image pixel size: %.2f [arcsec]" % pixelsize)

cube_image = np.array(cube_data, dtype=float)
unit = bunit
logger.info("Data unit: %s" % unit)

Nx = cube_image.shape[2]  ## The image size is 126*300*300. 
Ny = cube_image.shape[1]
Nz = cube_image.shape[0]

logger.info("Loaded data cube: %dx%d (cells) * %d (channels)" %
                    (Nx, Ny, Nz))


frequencies = np.asarray(get_frequencies(header_data)) # in MHz
nfreq = len(frequencies)
dfreq = frequencies[1] - frequencies[0]  # [MHz]
delta_nu = dfreq*1e6 # Hz

if nfreq != Nz:
    raise RuntimeError("data cube and frequencies do not match!")

logger.info("Frequency band: %.3f-%.3f [MHz]" %
                    (frequencies.min(), frequencies.max()))

logger.info("Frequency channel width: %.3f [MHz], %d channels" %
                    (dfreq, nfreq))

# Central frequency and redshift
freqc = frequencies.mean()
zc = freq2z(freqc)
logger.info("Central frequency %.3f [MHz] <-> redshift %.4f" %
                    (freqc, zc))

### calculate delays corresponding to freq axis ##
Ndelays = np.int(nfreq)
delays = np.fft.fftfreq(Ndelays, d= dfreq*1e6)
delays = np.fft.fftshift(delays) / units.Hz
delay_array = delays.to('ns').reshape(1,Ndelays)
k_parallel = eta2kparr(delay_array,zc.reshape(1, 1),cosmo=cosmo)

# Transverse comoving distance at zc; unit: [Mpc]
DMz = cosmo.comoving_transverse_distance(zc).value

window_name = 'blackmanharris'
window_func = getattr(signal.windows, window_name)
window = window_func(nfreq, sym=False)
window_single= window
window_cube = window_single[:, np.newaxis, np.newaxis]
##### 2D FFT to uvf space ###

cube_uvf = np.fft.fftshift(fft2(np.fft.ifftshift(cube_image, axes=(1,2)), axes=(1,2)), axes=(1,2))
cube_uv_delay = np.fft.fftshift(fft(np.fft.ifftshift(cube_uvf*window_cube,axes=0),axis=0),axes=0)


### generate periodic flag ###
bw = nfreq*dfreq
ngap = (1.28/dfreq).astype(int)
gap_chan = (0.2/dfreq).astype(int)

flag_mwa = np.ones(nfreq)
for ii in range(1,7):
    flag_mwa[ii*ngap:(ii*ngap)+gap_chan] = 0.

## generate random flag ### 

flag_cube = np.ones(shape=cube_uvf.shape,dtype=np.float)
for k in range(Nx):
    for j in range(Ny):

        ind = random.sample(range(nfreq),12)
        flag_cube[ind,k,j] = np.zeros(12,dtype = np.float)
        flag_cube[:,k,j] = flag_cube[:,k,j]*flag_mwa

flag_cube_bool = flag_cube.astype(bool)
####################################

pl.plot(frequencies,np.abs(flag_mwa*flag_cube_bool[:,100,100]),color='blue')
pl.plot(frequencies,np.abs(flag_mwa),color='magenta')
pl.xlabel('Freq [MHz]')
pl.ylabel('Flag')
#pl.yscale('log')
#pl.xscale('log')
#pl.ylim([-10,15])
pl.tight_layout()
pl.show()

## multiplying flags ##  

## Here I am multiplying random flag. If you want multiply this with periodic flag :- flag_mwa ##

flagged_vis = cube_uvf*flag_cube_bool
cube_uvf_flagged_windowed = flagged_vis * window_cube
cube_uv_delay_with_flag = np.fft.fftshift(fft(np.fft.ifftshift(cube_uvf_flagged_windowed,axes=0),axis=0),axes=0)



#### AIPY CLEAN ####
x = frequencies*1e6 
filter_centers = [0.]
bl_delay_max =delay_array.max().value/1e6
filter_half_widths = [bl_delay_max]
edgecut_low = 4
edgecut_hi = 4
window_clean='tukey'
clean_model_inpaint = np.zeros(shape=(Nz,Nx,Ny),dtype=np.complex)

for i in range(Nx):
    for j in range(Ny):
        data = cube_uvf[:,i,j]
        wgts = flag_cube[:,i,j]
        model_clean, resid_clean, info_clean = clean_filter(x=x, data=data, wgts=wgts, filter_centers=filter_centers, 
                filter_half_widths=filter_half_widths,
                  clean2d=False, tol=1e-9, window=window_clean, skip_wgt=0.15,
                  maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                  edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, add_clean_residual=True,
                  zero_residual_flags=True)
        clean_model_inpaint[:,i,j] = model_clean

clean_uv_eta = np.fft.fftshift(fft(np.fft.ifftshift(clean_model_inpaint*window_cube,axes=0),axis=0),axes=0)



#### for whole data LSSA #### 

nmax = 200 # Max. order of Fourier modes to fit. Take it large to get better result
sigma = 0.1 #np.sqrt(np.std(abs(cube_uvf[:,nx_ind,ny_ind])))  #0.1
data_LSSA_inpainted = np.zeros(shape=(Nz,Nx,Ny),dtype=np.complex)

for i in range(Nx):
    for j in range(Ny):
        model, model_coeff,data_uvf_in_painted = delay_filter_leastsq_1d(cube_uvf[:,i,j], flag_cube_bool[:,i,j], 
               sigma, nmax, add_noise=False,cn_guess=None, use_linear=True, operator=None, fundamental_period=None)
        data_LSSA_inpainted[:,i,j] = data_uvf_in_painted

## fft to delay space of the in-painted data ###

cube_uv_eta_LSSA = np.fft.fftshift(fft(np.fft.ifftshift(data_LSSA_inpainted*window_cube,axes=0),axis=0),axes=0)
 
 
### Trial with single baseline, Lots of trial is required to understand the correct choice of the parameters ### 
nx_ind = 11
ny_ind = 11


## check plot in delay space ##
k_baseline = 0.09646550696311297
u_wave = k_baseline * cosmo.comoving_transverse_distance(zc) / (2 * np.pi)
wl = const.c.value/(freqc*1e6)
baseline_length = u_wave.value
delay_baseline = baseline_length/const.c.value

x1 = np.ones(100)*delay_baseline*1e9 # in ns
y1 = np.linspace(1e-4,2e3,100)


pl.plot(delay_array[0,:].value, abs(cube_uv_delay_with_flag[:,nx_ind,ny_ind]),color='blue',label='RFI')
pl.plot(delay_array[0,:].value, abs(cube_uv_eta_LSSA[:,nx_ind,ny_ind]),color='red',linestyle='--',label='LSSA')
pl.plot(delay_array[0,:].value,abs(clean_uv_eta[:,nx_ind,ny_ind]),color='green',label='CLEAN')
pl.plot(delay_array[0,:].value, abs(cube_uv_delay[:,nx_ind,ny_ind]),color='gray',lw=1.0,label='No RFI')
pl.plot(x1,y1,linestyle='--',color='black',label='delay horizon')
pl.plot(-x1,y1,linestyle='--',color='black')

pl.yscale('log')
pl.xlabel('delay [ns]')
pl.ylabel(r' Vis$(\eta)$  [Jy Hz]')
pl.ylim([1e-4,5e4])
#pl.xlim([0,8000])
pl.legend(ncol=2,loc='upper center',shadow=True)
pl.title('20% RFI')
pl.tight_layout()
pl.show()




"""
Calculate the 3D power spectrum of the image cube.

The power spectrum is properly normalized to have dimension
of [K^2 Mpc^3].
"""


logger.info("Calculating 3D power spectrum ...")
ps3d_un_normed_clean = np.abs(clean_uv_eta) ** 2  # [K^2]
ps3d_un_normed_LSSA = np.abs(cube_uv_eta_LSSA) ** 2
ps3d_un_normed_no_flag = np.abs(cube_uv_delay) ** 2 
ps3d_un_normed_BH = np.abs(cube_uv_delay_with_flag) ** 2 

# spatial and los conversion ####

pixelsize_deg = pixelsize / 3600  # [arcsec] -> [deg]
d_xy = DMz * np.deg2rad(pixelsize_deg) #[Mpc] The sampling interval along the (X, Y) spatial dimensions Reference: Ref.[liu2014].Eq.(A7)

##########################################################

c = ac.c.to("km/s").value  # [km/s]
Hz = cosmo.H(zc).value  # [km/s/Mpc]
d_z = c * (1+zc)**2 * dfreq / Hz / freq21cm  #[Mpc]  The sampling interval along the Z line-of-sight dimension Reference: Ref.[liu2014].Eq.(A9)

############################################

fs_xy = 1/d_xy # [Mpc^-1] The sampling frequency along the (X, Y) spatial dimensions
fs_z = 1/d_z # [Mpc^-1] The sampling frequency  along the Z line-of-sight dimension. 

df_xy =  fs_xy / Nx  # The spatial frequency bin size (i.e., resolution) along the (X, Y) dimensions

df_z = fs_z / Nz  # The spatial frequency bin size (i.e., resolution) along the line-of-sight (Z) direction.

dk_xy = 2*np.pi * df_xy   # The k-space (spatial) frequency bin size (i.e., resolution).
dk_z = 2*np.pi * df_z  # The k-space (los) frequency bin size (i.e., resolution).

#######################################################
#bins = 20
#bins_kperp = np.logspace(np.log10(k_perpendicular.min().value), np.log10(k_perpendicular.max().value), bins + 1)
#k_perp_bin = 10**bins_kperp

k_xy = 2*np.pi * (fftpack.fftshift(fftpack.fftfreq(Nx, d=d_xy))) # [Mpc^-1] The k-space coordinates along the (X, Y) spatial dimensions, k = 2*pi * f, where "f" is the spatial frequencies, and the Fourier dual to spatial transverse distances x/y. 

k_z = 2*np.pi * (fftpack.fftshift(fftpack.fftfreq(Nz, d=d_z)))

k_perp = k_xy[k_xy >= 0]  # Comoving wavenumbers perpendicular to the LoS
k_los =  k_z[k_z >= 0]  # Comoving wavenumbers along the LoS
       

##### 3D PS normalization ### 
fov = np.deg2rad(3.12/(freqc/200.0))
wl = const.c.value/(freqc*1e6) 
PB = 1.02*(wl/38.0) ## diameter = 38m 
area = np.pi*PB**2/(4*np.log(2)) 
#area_sr = np.deg2rad(area
BH_area = (np.sum(window/window.max())/len(window))**2.0

norm1 = 1./(Nx * Ny * Nz)
norm2 = 1. / (fs_xy**2 * fs_z)  # [Mpc^3]  ## This is volume dx * dy * dz in k-space
norm3 = 1. / (2*np.pi)**3  # turn it on if your FT does not use 1/2pi
norm4 = 1./(area*BH_area)

ps3d_clean = ps3d_un_normed_clean  * norm1 * norm2 * norm3 * norm4# [K^2 Mpc^3]
ps3d_lssa = ps3d_un_normed_LSSA  * norm1 * norm2 * norm3 * norm4
ps3d_no_rfi = ps3d_un_normed_no_flag  * norm1 * norm2 * norm3 * norm4  
ps3d_BH = ps3d_un_normed_BH* norm1 * norm2 * norm3 * norm4  

#################################################################

"""
Calculate the 2D power spectrum by cylindrically binning
the above 3D power spectrum.

Returns
 -------
ps2d : 3D `~numpy.ndarray`
       3D array of shape (3, n_k_los, n_k_perp) including:
       + average (median / mean)
       + error (1.4826*MAD / standard deviation)
       + number of averaging cells

Attributes
----------
ps2d
"""


 

logger.info("Calculating 2D power spectrum ...")
n_k_perp = len(k_perp)
n_k_los = len(k_los)

# PS2D's 3 layers: value, error, number of averaging cells
ps2d_clean = np.zeros(shape=(3, n_k_los, n_k_perp))
ps2d_LSSA = np.zeros(shape=(3, n_k_los, n_k_perp))
ps2d_no_flag = np.zeros(shape=(3, n_k_los, n_k_perp))
ps2d_BH = np.zeros(shape=(3, n_k_los, n_k_perp))

eps = 1e-8

ic_xy = (np.abs(k_xy) < eps).nonzero()[0][0]
ic_z = (np.abs(k_z) < eps).nonzero()[0][0]

p_xy = np.arange(Nx) - ic_xy
p_z = np.abs(np.arange(Nz) - ic_z)

mx, my = np.meshgrid(p_xy, p_xy)

rho, phi = cart2pol(mx, my)
rho = np.around(rho).astype(int)
meanstd = False

logger.info("Cylindrically averaging 3D power spectrum ...")
for r in range(n_k_perp):
    ix, iy = (rho == r).nonzero()
    for s in range(n_k_los):
        iz = (p_z == s).nonzero()[0]
        cells_clean = np.concatenate([ps3d_clean[z, iy, ix] for z in iz])
        ps2d_clean[2, s, r] = len(cells_clean)

        cells_lssa = np.concatenate([ps3d_lssa[z, iy, ix] for z in iz])
        ps2d_LSSA[2, s, r] = len(cells_lssa)

        cells_no_flag = np.concatenate([ps3d_no_rfi[z, iy, ix] for z in iz])
        ps2d_no_flag[2, s, r] = len(cells_no_flag)


        cells_BH = np.concatenate([ps3d_BH[z, iy, ix] for z in iz])
        ps2d_BH[2, s, r] = len(cells_BH)


        if meanstd:
           ps2d_clean[0, s, r] = cells_clean.mean()
           ps2d_clean[1, s, r] = cells_clean.std()

           ps2d_LSSA[0, s, r] = cells_lssa.mean()
           ps2d_LSSA[1, s, r] = cells_lssa.std()

           ps2d_no_flag[0, s, r] = cells_no_flag.mean()
           ps2d_no_flag[1, s, r] = cells_no_flag.std()

           ps2d_BH[0, s, r] = cells_BH.mean()
           ps2d_BH[1, s, r] = cells_BH.std()

        else:
             median_clean = np.median(cells_clean)
             mad_clean = np.median(np.abs(cells_clean - median_clean))
             ps2d_clean[0, s, r] = median_clean
             ps2d_clean[1, s, r] = mad_clean * 1.4826

             median_lssa = np.median(cells_lssa)
             mad_lssa = np.median(np.abs(cells_lssa - median_lssa))
             ps2d_LSSA[0, s, r] = median_lssa
             ps2d_LSSA[1, s, r] = mad_lssa * 1.4826

             median_no_flag = np.median(cells_no_flag)
             mad_no_flag = np.median(np.abs(cells_no_flag - median_no_flag))
             ps2d_no_flag[0, s, r] = median_no_flag
             ps2d_no_flag[1, s, r] = mad_no_flag * 1.4826

             median_BH = np.median(cells_BH)
             mad_BH = np.median(np.abs(cells_BH - median_BH))
             ps2d_BH[0, s, r] = median_BH
             ps2d_BH[1, s, r] = mad_BH * 1.4826


##### Wedge line #### 
fov = 3.12/(freqc/200.0)    # SKA1-Low has FoV ~ 3.12 / (nu/200MHz) [deg]
#fov = 3.819
e = 4 # conv-width
Hz = cosmo.H(zc).value  # [km/s/Mpc]
Dc = cosmo.comoving_distance(zc).value  # [Mpc]
bandwidth = dfreq*nfreq
c = ac.c.to("km/s").value  # [km/s]
coef = Hz * Dc / (c * (1+zc))
term1 = np.sin(np.deg2rad(fov)) * k_perp  # [Mpc^-1]
term2 = ((2*np.pi * e * freq21cm / bandwidth) /
                 ((1 + zc) * Dc))  # [Mpc^-1]
k_los_wedge = coef * (term1 + term2)


## check plot ##

x1 = np.ones(100)*k_los_wedge[11] # in 1/Mpc
y1 = np.linspace(5e-5,2e4,100)

#pl.plot(delay_array[0,:].value, abs(clean_vis_net_delay_space),color='magenta',label='Clean')
pl.plot(k_los, abs(ps2d_BH[0,:,11]),color='blue',label='RFI')
pl.plot(k_los,  abs(ps2d_LSSA[0,:,11]),color='red',linestyle='--',label='LSSA')
pl.plot(k_los,  abs(ps2d_clean[0,:,11]),color='green',linestyle='-.',label='CLEAN')
pl.plot(k_los,  abs(ps2d_no_flag[0,:,11]),color='gray',lw=2,alpha=0.8,label='No RFI')
pl.plot(x1,y1,linestyle='--',color='black',label='delay horizon')

pl.yscale('log')
pl.xlabel(r'$k_{\parallel}$ [Mpc$^{-1}$]')
pl.ylabel(r' P(k) [K$^{2}$ Mpc$^{3}$]')
#pl.ylim([5e-5,2e4])
#pl.xlim([0,8000])
pl.legend(ncol=2,loc='center',shadow=True)
pl.title('20% RFI')
pl.tight_layout()
pl.show()

## estimate the ratio ###
#t = fits.open('eor_diffuse_signal_2D_save.fits')
#signal = t[0].data[0,:,:]

ratio_clean = ps2d_clean[0,:,:]/ps2d_no_flag[0,:,:]
ratio_lssa = ps2d_LSSA[0,:,:]/ps2d_no_flag[0,:,:]
ratio_BH = ps2d_BH[0,:,:]/ps2d_no_flag[0,:,:]

from matplotlib.colors import LogNorm, SymLogNorm
fig1, ax1 = subplots(1, figsize=(9, 7))
norm = LogNorm(vmin=1e6, vmax= 1e10)
#norm = LogNorm(vmin=4.e-1, vmax= 6e1)

im = ax1.pcolormesh(k_perp[1:], k_los[1:],  np.log10(ps2d_no_flag[0,1:,1:]*1e6),vmin=-2,vmax=15,cmap='jet')

#im = ax1.pcolormesh(k_perp[1:], k_los[1:],  np.log10(ratio_clean[1:,1:]),vmin=0,vmax=1,cmap='jet')
#im = ax1.pcolormesh(k_perp[1:], k_los[1:],  np.log10(ratio_lssa[1:,1:]),vmin=0,vmax=1,cmap='jet')
#im = ax1.pcolormesh(k_perp[1:], k_los[1:],  np.log10(ratio_BH[1:,1:]),vmin=0,cmap='jet')

ax1.plot(k_perp, k_los_wedge, color="black", linewidth=2, linestyle="--")


#cbar_ticks = [1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]
cbar = fig1.colorbar(im, ax=ax1, extend='both',pad=0.08)
cbar.ax.minorticks_off()
#cbar.set_ticks(cbar_ticks)

cbar.set_label(r' log10(ratio)')
ax1.set_yscale('log')
ax1.set_xscale('log')


ax1.set_xlabel(r'$k_{\perp} [Mpc^{-1}]$')
ax1.set_ylabel(r'$k_{\parallel} [Mpc^{-1}]$')
#ax1.set_xlim([1.e-2,6e-2])
#ax1.set_xticks([1e-2,4e-2])
pl.title('10% RFI')
pl.tight_layout()
#pl.savefig('trail_eor_diffuse.png')
pl.show()

## Saving ###

#k_xy = dk_xy
#dk_z = dk_z

hdr = fits.Header()
hdr["HDUNAME"] = ("PS2D", "block name")
hdr["CONTENT"] = ("2D cylindrically averaged power spectrum",
                          "data product")
hdr["BUNIT"] = ("%s^2 Mpc^3" % unit, "data unit")
if meanstd:
            hdr["AvgType"] = ("mean + standard deviation", "average type")
else:
     hdr["AvgType"] = ("median + 1.4826*MAD", "average type")

#hdr["WINDOW"] = (window_name,
#                         "window applied along frequency axis")


hdr["LTV1"] = 0.0
hdr["LTM1_1"] = 1.0 / dk_xy
hdr["LTV2"] = 0.0
hdr["LTM2_2"] = 1.0 / dk_z

# WCS physical coordinates
hdr["WCSTY1P"] = "PHYSICAL"
hdr["CTYPE1P"] = ("k_perp", "wavenumbers perpendicular to LoS")
hdr["CRPIX1P"] = (0.5, "reference pixel")
hdr["CRVAL1P"] = (0.0, "coordinate of the reference pixel")
hdr["CDELT1P"] = (dk_xy, "coordinate delta/step")
hdr["CUNIT1P"] = ("Mpc^-1", "coordinate unit")
hdr["WCSTY2P"] = "PHYSICAL"
hdr["CTYPE2P"] = ("k_los", "wavenumbers along LoS")
hdr["CRPIX2P"] = (0.5, "reference pixel")
hdr["CRVAL2P"] = (0.0, "coordinate of the reference pixel")
hdr["CDELT2P"] = (dk_z, "coordinate delta/step")
hdr["CUNIT2P"] = ("Mpc^-1", "coordinate unit")

# Data information
hdr["PixSize"] = (pixelsize, "[arcsec] data cube pixel size")
hdr["Z_C"] = (zc, "data cube central redshift")
hdr["Freq_C"] = (freqc, "[MHz] data cube central frequency")
hdr["Freq_Min"] = (frequencies.min(),
                           "[MHz] data cube minimum frequency")
hdr["Freq_Max"] = (frequencies.max(),
                           "[MHz] data cube maximum frequency")
# Command history
hdr.add_history(" ".join(sys.argv))

# Save the calculated 2D power spectrum as a FITS image.

hdu = fits.PrimaryHDU(data=ps2d_BH, header=hdr)

outfile = 'eor_BH_2D_save_12chans_MWA.fits'
hdu.writeto(outfile, overwrite=True)

logger.info("Wrote 2D power spectrum to file: %s" % outfile)


