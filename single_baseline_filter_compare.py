## introduce flag, FFT, then use CLEAN and estimate PS ##
from functions import *

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
window_single= window/window.mean()
window_cube = window_single[:, np.newaxis, np.newaxis]
##### 2D FFT to uvf space ###

cube_uvf = np.fft.fftshift(fft2(np.fft.ifftshift(cube_image, axes=(1,2)), axes=(1,2)), axes=(1,2))
cube_uv_delay = np.fft.fftshift(fft(np.fft.ifftshift(cube_uvf*window_cube,axes=0),axis=0),axes=0)

### generate flag  ## 

flag_cube = np.ones(shape=cube_uvf.shape,dtype=np.float)
for k in range(Nx):
    for j in range(Ny):

        ind = random.sample(range(nfreq),12)
        flag_cube[ind,k,j] = np.zeros(12,dtype = np.float)

flag_cube_bool = flag_cube.astype(bool)
####################################
  
flagged_vis = cube_uvf*flag_cube_bool
cube_uvf_flagged_windowed = flagged_vis * window_cube
cube_uv_delay_with_flag = np.fft.fftshift(fft(np.fft.ifftshift(cube_uvf_flagged_windowed,axes=0),axis=0),axes=0)

### Trial with single baseline, Lots of trial is required to understand the correct choice of the parameters ### 

nx_ind = 11
ny_ind = 11

#### AIPY CLEAN ####

x = frequencies*1e6 
data =  cube_uvf[:,nx_ind,nx_ind]
filter_centers = [0.]
bl_delay_max =delay_array.max().value/1e6
filter_half_widths = [bl_delay_max]
edgecut_low = 4
edgecut_hi = 4
wgts = flag_cube[:,nx_ind,ny_ind]
window='tukey'

data_array_padded = np.pad(data, ( (int(np.ceil(edgecut_low)), int(np.floor(edgecut_hi)))), mode='constant', constant_values=(0,0))
wgts_padded = np.pad(wgts, ( (int(np.ceil(edgecut_low)), int(np.floor(edgecut_hi)))), mode='constant', constant_values=(0,0))

model_clean, resid_clean, info_clean = clean_filter(x=x, data=data, wgts=wgts, filter_centers=filter_centers, 
                filter_half_widths=filter_half_widths,
                  clean2d=False, tol=1e-9, window=window, skip_wgt=0.15,
                  maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.2,
                  edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, add_clean_residual=True,
                  zero_residual_flags=True)

clean_vis_delay_space = np.fft.fftshift(fft(np.fft.ifftshift(model_clean*window_single,axes=0),axis=0),axes=0)

### LSSA  ### 
nmax = 200 # Max. order of Fourier modes to fit. Take it large to get better result
sigma = 0.1 
model, model_coeff,data_uvf_in_painted = delay_filter_leastsq_1d(cube_uvf[:,nx_ind,ny_ind], flag_cube_bool[:,nx_ind,ny_ind], 
               sigma, nmax, add_noise=False,cn_guess=None, use_linear=True, operator=None, fundamental_period=None)  # in-painted data will store here

## fft to delay space of the in-painted data ###
cube_uv_delay_after_LSSA = np.fft.fftshift(fft(np.fft.ifftshift(data_uvf_in_painted*window_single,axes=0),axis=0),axes=0)




## plot in-painted data ##

#pl.plot(frequencies,abs(clean_vis_freq_space))
pl.plot(frequencies,abs(data_uvf_in_painted),color='magenta')
pl.plot(frequencies,abs(flagged_vis[:,nx_ind,ny_ind]),color='red')
pl.plot(frequencies,abs(model_clean),color='black')
pl.yscale('log')
pl.show()

## check plot in delay space ##
k_baseline = 0.09646550696311297
u_wave = k_baseline * cosmo.comoving_transverse_distance(zc) / (2 * np.pi)
wl = const.c.value/(freqc*1e6)
baseline_length = u_wave.value
delay_baseline = baseline_length/const.c.value

x1 = np.ones(100)*delay_baseline*1e9 # in ns
y1 = np.linspace(1e-4,2e3,100)


pl.plot(delay_array[0,:].value, abs(cube_uv_delay_with_flag[:,nx_ind,ny_ind]),color='blue',label='RFI')
pl.plot(delay_array[0,:].value, abs(cube_uv_delay_after_LSSA),color='red',linestyle='--',label='LSSA')
pl.plot(delay_array[0,:].value,abs(clean_vis_delay_space),color='green',linestyle='--',label='CLEAN')
pl.plot(delay_array[0,:].value, abs(cube_uv_delay[:,nx_ind,ny_ind]),color='gray',lw=2.0,alpha=0.8,label='No RFI')
pl.plot(x1,y1,linestyle='--',color='black',label='delay horizon')
pl.plot(-x1,y1,linestyle='--',color='black')

pl.yscale('log')
pl.xlabel('delay [ns]')
pl.ylabel(r' Vis$(\eta)$  [Jy Hz]')
pl.ylim([1e-4,6e4])
#pl.xlim([0,8000])
pl.legend(ncol=2,loc='upper center',shadow=True)
pl.title('10% RFI')
pl.tight_layout()
pl.show()



