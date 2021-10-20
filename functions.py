
from __future__ import print_function, division, absolute_import
import os
import sys
import argparse
import logging
#from functools import lru_cache

import numpy as np
from scipy import fftpack
from scipy import signal
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as ac

from astropy import constants as const
from astropy import units
from astropy.units import Quantity
from astropy.cosmology import Planck15, default_cosmology
import random
from scipy.optimize import leastsq, lsq_linear

from astropy import units



import aipy
from six.moves import range
from scipy.signal import windows
from warnings import warn
import copy

# the emission frequency of 21m photons in the Hydrogen's rest frame
f21 = 1420405751.7667 * units.Hz

# HI line frequency
freq21cm = 1420.405751  # [MHz]


def complex1dClean_arg_splitter(args, **kwargs):
    return complex1dClean(*args, **kwargs)

def complex1dClean(inp, kernel, cbox=None, gain=0.1, maxiter=10000,
                   threshold=5e-3, threshold_type='relative', verbose=False,
                   progressbar=False, pid=None, progressbar_yloc=0):

    """
    ----------------------------------------------------------------------------
    Hogbom CLEAN algorithm applicable to 1D complex array

    Inputs:

    inp      [numpy vector] input 1D array to be cleaned. Can be complex.

    kernel   [numpy vector] 1D array that acts as the deconvolving kernel. Can 
             be complex. Must be of same size as inp

    cbox     [boolean array] 1D boolean array that acts as a mask for pixels 
             which should be cleaned. Same size as inp. Only pixels with values 
             True are to be searched for maxima in residuals for cleaning and 
             the rest are not searched for. Default=None (means all pixels are 
             to be searched for maxima while cleaning)

    gain     [scalar] gain factor to be applied while subtracting clean 
             component from residuals. This is the fraction of the maximum in 
             the residuals that will be subtracted. Must lie between 0 and 1.
             A lower value will have a smoother convergence but take a longer 
             time to converge. Default=0.1

    maxiter  [scalar] maximum number of iterations for cleaning process. Will 
             terminate if the number of iterations exceed maxiter. Default=10000

    threshold 
             [scalar] represents the cleaning depth either as a fraction of the
             maximum in the input (when thershold_type is set to 'relative') or
             the absolute value (when threshold_type is set to 'absolute') in 
             same units of input down to which inp should be cleaned. Value must 
             always be positive. When threshold_type is set to 'relative', 
             threshold must lie between 0 and 1. Default=5e-3 (found to work 
             well and converge fast) assuming threshold_type is set to 'relative'

    threshold_type
             [string] represents the type of threshold specified by value in 
             input threshold. Accepted values are 'relative' and 'absolute'. If
             set to 'relative' the threshold value is the fraction (between 0
             and 1) of maximum in input down to which it should be cleaned. If 
             set to 'asbolute' it is the actual value down to which inp should 
             be cleaned. Default='relative'

    verbose  [boolean] If set to True (default), print diagnostic and progress
             messages. If set to False, no such messages are printed.

    progressbar 
             [boolean] If set to False (default), no progress bar is displayed

    pid      [string or integer] process identifier (optional) relevant only in
             case of parallel processing and if progressbar is set to True. If
             pid is not specified, it defaults to the Pool process id

    progressbar_yloc
             [integer] row number where the progressbar is displayed on the
             terminal. Default=0

    Output:

    outdict  [dictionary] It consists of the following keys and values at
             termination:
             'termination' [dictionary] consists of information on the 
                           conditions for termination with the following keys 
                           and values:
                           'threshold' [boolean] If True, the cleaning process
                                       terminated because the threshold was 
                                       reached
                           'maxiter'   [boolean] If True, the cleaning process
                                       terminated because the number of 
                                       iterations reached maxiter
                           'inrms<outrms'
                                       [boolean] If True, the cleaning process
                                       terminated because the rms inside the 
                                       clean box is below the rms outside of it
             'iter'        [scalar] number of iterations performed before 
                           termination
             'rms'         [numpy vector] rms of the residuals as a function of
                           iteration
             'inrms'       [numpy vector] rms of the residuals inside the clean 
                           box as a function of iteration
             'outrms'      [numpy vector] rms of the residuals outside the clean 
                           box as a function of iteration
             'res'         [numpy array] uncleaned residuals at the end of the
                           cleaning process. Complex valued and same size as 
                           inp
             'cc'          [numpy array] clean components at the end of the
                           cleaning process. Complex valued and same size as 
                           inp
    ----------------------------------------------------------------------------
    """

    try:
        inp, kernel
    except NameError:
        raise NameError('Inputs inp and kernel not specified')

    if not isinstance(inp, np.ndarray):
        raise TypeError('inp must be a numpy array')
    if not isinstance(kernel, np.ndarray):
        raise TypeError('kernel must be a numpy array')

    if threshold_type not in ['relative', 'absolute']:
        raise ValueError('invalid specification for threshold_type')

    if not isinstance(threshold, (int,float)):
        raise TypeError('input threshold must be a scalar')
    else:
        threshold = float(threshold)
        if threshold <= 0.0:
            raise ValueError('input threshold must be positive')

    inp = inp.flatten()
    kernel = kernel.flatten()
    kernel /= np.abs(kernel).max()
    kmaxind = np.argmax(np.abs(kernel))  ### this gives the index of the max value of the kernel/dirty beam ##

    if inp.size != kernel.size:
        raise ValueError('inp and kernel must have same size')

    if cbox is None:
        cbox = np.ones(inp.size, dtype=np.bool)
    elif isinstance(cbox, np.ndarray):
        cbox = cbox.flatten()
        if cbox.size != inp.size:
            raise ValueError('Clean box must be of same size as input')
        cbox = np.where(cbox > 0.0, True, False)
        # cbox = cbox.astype(np.int)
    else:
        raise TypeError('cbox must be a numpy array')
    cbox = cbox.astype(np.bool)

    if threshold_type == 'relative':
        lolim = threshold
    else:
        lolim = threshold / np.abs(inp).max()

    if lolim >= 1.0:
        raise ValueError('incompatible value specified for threshold')

    # inrms = [np.std(inp[cbox])]
    inrms = [np.median(np.abs(inp[cbox] - np.median(inp[cbox])))]
    if inp.size - np.sum(cbox) <= 2:
        outrms = None
    else:
        # outrms = [np.std(inp[np.invert(cbox)])]
        outrms = [np.median(np.abs(inp[np.invert(cbox)] - np.median(inp[np.invert(cbox)])))]

    if not isinstance(gain, float):
        raise TypeError('gain must be a floating point number')
    else:
        if (gain <= 0.0) or (gain >= 1.0):
            raise TypeError('gain must lie between 0 and 1')

    if not isinstance(maxiter, int):
        raise TypeError('maxiter must be an integer')
    else:
        if maxiter <= 0:
            raise ValueError('maxiter must be positive')

    cc = np.zeros_like(inp)   ## this stores the 'CLEAN' components 
    res = np.copy(inp)        ### This stores the residual 
    cond3 = False
   # prevrms = np.std(res)
   # currentrms = [np.std(res)]
    prevrms = np.median(np.abs(res - np.median(res)))
    currentrms = [np.median(np.abs(res - np.median(res)))]
    itr = 0
    terminate = False

    if progressbar:
        if pid is None:
            pid = MP.current_process().name
        else:
            pid = '{0:0d}'.format(pid)
        progressbar_loc = (0, progressbar_yloc)
        writer = WM.Writer(progressbar_loc)
        progress = PGB.ProgressBar(widgets=[pid+' ', PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Iterations '.format(maxiter), PGB.ETA()], maxval=maxiter, fd=writer).start()
    while not terminate:
        itr += 1
        indmaxres = np.argmax(np.abs(res*cbox))  ## This gives the position of 'peak' value of the data in first loop, cause I have copied the data in res. In 2nd loop it gives the 'peak' in residual. 
        maxres = res[indmaxres]                ## Value of the 'peak' in the data. 
        
        ccval = gain * maxres     ## now I multiply the gain (0.01 in my case, you may give anything in the start) with the peak. 
        cc[indmaxres] += ccval     ## I put the gain*peak value to the Component list at the position of the peak. 
        res = res - ccval * np.roll(kernel, indmaxres-kmaxind) ## rolling shift the kernel/dirty beam to the 'peak' position 
        
        prevrms = np.copy(currentrms[-1])
       # currentrms += [np.std(res)]
        currentrms += [np.median(np.abs(res - np.median(res)))]

       # inrms += [np.std(res[cbox])]
        inrms += [np.median(np.abs(res[cbox] - np.median(res[cbox])))]
            
        # cond1 = np.abs(maxres) <= inrms[-1]
        cond1 = np.abs(maxres) <= lolim * np.abs(inp).max()
        cond2 = itr >= maxiter
        terminate = cond1 or cond2
        if outrms is not None:
            # outrms += [np.std(res[np.invert(cbox)])]
            outrms += [np.median(np.abs(res[np.invert(cbox)] - np.median(res[np.invert(cbox)])))]
            cond3 = inrms[-1] <= outrms[-1]
            terminate = terminate or cond3

        if progressbar:
            progress.update(itr)
    if progressbar:
        progress.finish()

    inrms = np.asarray(inrms)
    currentrms = np.asarray(currentrms)
    if outrms is not None:
        outrms = np.asarray(outrms)
        
    outdict = {'termination':{'threshold': cond1, 'maxiter': cond2, 'inrms<outrms': cond3}, 'iter': itr, 'rms': currentrms, 'inrms': inrms, 'outrms': outrms, 'cc': cc, 'res': res}

    return outdict


def eta2kparr(eta, z, cosmo=None):
    """Conver delay eta to k_parallel (comoving 1./Mpc along line of sight).

    Parameters
    ----------
    eta : Astropy Quantity object with units equivalent to time.
        The inteferometric delay observed in units compatible with time.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    kparr : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight probed by the input delay (eta).

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (eta * (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))
            / (const.c * (1 + z)**2)).to('1/Mpc')


@units.quantity_input(kparr='wavenumber')
def kparr2eta(kparr, z, cosmo=None):
    """Convert k_parallel (comoving 1/Mpc along line of sight) to delay eta.

    Parameters
    ----------
    kparr : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    eta : Astropy Quantity units equivalent to time
        The inteferometric delay which probes the spatial scale given by kparr.

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (kparr * const.c * (1 + z)**2
            / (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))).to('s')


def calc_z(freq):
    """Calculate the redshift from a given frequency or frequncies.

    Parameters
    ----------
    freq : Astropy Quantity Object units equivalent to frequency
        The frequency to calculate the redshift of 21cm emission

    Returns
    -------
    redshift : float
        The redshift consistent with 21cm observations of the input frequency.

    """
    return (f21 / freq).si.value - 1


def calc_freq(redshift):
    """Calculate the frequency or frequencies of a given 21cm redshift.

    Parameters
    ----------
    redshift : float
        The redshift of the expected 21cm emission

    Returns
    -------
    freq : Astropy Quantity Object units equivalent to frequency
        Frequency of the emission in the rest frame of emission

    """
    return f21 / (1 + redshift)

def freq2z(freq):
    z = freq21cm / freq - 1.0
    return z


def get_frequencies(header):
    """
    Get the frequencies for each cube slice.
    Unit: [MHz]
    """
    nfreq = header["NAXIS3"]
    freq0 = header["CRVAL3"]  # [Hz]
    freqstep = header["CDELT3"]  # [Hz]
    frequencies = freq0 + freqstep*np.arange(nfreq)
    return frequencies / 1e6  # [MHz]


def get_pixelsize(header):
    """
    Get the pixel size of cube image.
    Unit: [arcsec]
    """
    try:
        pixelsize = header["PixSize"]  # [arcsec]
    except KeyError:
        try:
            pixelsize = header["CDELT1"]  # [deg]
            if abs(pixelsize-1.0) < 1e-8:
                # Place-holder value set by ``fitscube.py``
                pixelsize = None
            else:
                pixelsize = abs(pixelsize)*3600  # [arcsec]
        except KeyError:
            pixelsize = None
    return pixelsize

def gen_window(nfreq, name=None):
        if (name is None) or (name.upper() == "NONE"):
            return None

        window_func = getattr(signal.windows, name)
        nfreq = nfreq
        window = window_func(nfreq, sym=False)
        width_pix = nfreq
        logger.info("Generated window: %s (%d pixels)" % (name, width_pix))
        return window



def save(self, outfile, clobber=False):
        """
        Save the calculated 2D power spectrum as a FITS image.
        """
        hdu = fits.PrimaryHDU(data=self.ps2d, header=self.header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)
        logger.info("Wrote 2D power spectrum to file: %s" % outfile)

def plot(self, ax, ax_err, colormap="jet"):
        """
        Plot the calculated 2D power spectrum.
        """
        x = self.k_perp
        y = self.k_los

        if self.meanstd:
            title = "2D Power Spectrum (mean)"
            title_err = "Error (standard deviation)"
        else:
            title = "2D Power Spectrum (median)"
            title_err = "Error (1.4826*MAD)"

        # median/mean
        mappable = ax.pcolormesh(x[1:], y[1:],
                                 np.log10(self.ps2d[0, 1:, 1:]),
                                 cmap=colormap)
        vmin, vmax = mappable.get_clim()
        ax.set(xscale="log", yscale="log",
               xlim=(x[1], x[-1]), ylim=(y[1], y[-1]),
               xlabel=r"$k_{\perp}$ [Mpc$^{-1}$]",
               ylabel=r"$k_{||}$ [Mpc$^{-1}$]",
               title=title)
        cb = ax.figure.colorbar(mappable, ax=ax, pad=0.01, aspect=30)
        cb.ax.set_xlabel(r"[%s$^2$ Mpc$^3$]" % self.unit)

        # error
        mappable = ax_err.pcolormesh(x[1:], y[1:],
                                     np.log10(self.ps2d[1, 1:, 1:]),
                                     cmap=colormap)
        mappable.set_clim(vmin, vmax)
        ax_err.set(xscale="log", yscale="log",
                   xlim=(x[1], x[-1]), ylim=(y[1], y[-1]),
                   xlabel=r"$k_{\perp}$ [Mpc$^{-1}$]",
                   ylabel=r"$k_{||}$ [Mpc$^{-1}$]",
                   title=title_err)
        cb = ax_err.figure.colorbar(mappable, ax=ax_err, pad=0.01, aspect=30)
        cb.ax.set_xlabel(r"[%s$^2$ Mpc$^3$]" % self.unit)

        return (ax, ax_err)

## Reading cube properties #

def nx(cube):
        """
        Number of cells/pixels along the X axis.
        Cube shape/dimensions: [Z, Y, X]
        """
        return cube.shape[2]


def ny(cube):
        return cube.shape[1]


def nz(cube):
        return scube.shape[0]

### cosmological conversion ###

def d_xy(pixelsize,DMz):
        """
        The sampling interval along the (X, Y) spatial dimensions,
        translated from the pixel size and given the comoving distance for the central redshift
        Unit: [Mpc]

        Reference: Ref.[liu2014].Eq.(A7)
        """
        pixelsize = pixelsize / 3600  # [arcsec] -> [deg]
        d_xy = DMz * np.deg2rad(pixelsize)
        return d_xy


def d_z(dfreq,zc):
        """
        The sampling interval along the Z line-of-sight dimension,
        translated from the frequency channel width.
        Unit: [Mpc]

        Reference: Ref.[liu2014].Eq.(A9)
        """
        dfreq = dfreq  # [MHz]
        c = ac.c.to("km/s").value  # [km/s]
        Hz = cosmo.H(zc).value  # [km/s/Mpc]
        d_z = c * (1+zc)**2 * dfreq / Hz / freq21cm
        return d_z


def fs_xy(d_xy):
        """
        The sampling frequency along the (X, Y) spatial dimensions:
            Fs = 1/T (inverse of interval)
        Unit: [Mpc^-1]
        """
        return 1/d_xy

def fs_z(d_z):
        """
        The sampling frequency along the Z line-of-sight dimension.
        Unit: [Mpc^-1]
        """
        return 1/d_z


def df_xy(fs_xy,Nx):
        """
        The spatial frequency bin size (i.e., resolution) along the
        (X, Y) dimensions.
        Unit: [Mpc^-1]
        """
        return fs_xy / Nx


def df_z(fs_z,Nz):
        """
        The spatial frequency bin size (i.e., resolution) along the
        line-of-sight (Z) direction.
        Unit: [Mpc^-1]
        """
        return fs_z / Nz

def dk_xy(df_xy):
        """
        The k-space (spatial) frequency bin size (i.e., resolution).
        """
        return 2*np.pi * df_xy


def dk_z(df_z):
        return 2*np.pi * df_z


def k_xy(Nx,d_xy):
        """
        The k-space coordinates along the (X, Y) spatial dimensions,
        which describe the spatial frequencies.

        NOTE:
        k = 2*pi * f, where "f" is the spatial frequencies, and the
        Fourier dual to spatial transverse distances x/y.

        Unit: [Mpc^-1]
        """
        f_xy = fftpack.fftshift(fftpack.fftfreq(Nx, d=d_xy))
        k_xy = 2*np.pi * f_xy
        return k_xy


def k_z(Nz,d_z):
        f_z = fftpack.fftshift(fftpack.fftfreq(Nz, d=d_z))
        k_z = 2*np.pi * f_z
        return k_z


def k_perp(k_xy):
        """
        Comoving wavenumbers perpendicular to the LoS

        NOTE: The Nyquist frequency just located at the first element
              after fftshift when the length is even, and it is negative.
        """
        k_x = k_xy
        return k_x[k_x >= 0]


def k_los(k_z):
        """
        Comoving wavenumbers along the LoS
        """
        k_z = k_z
        return k_z[k_z >= 0]


def cart2pol(x, y):
        """
        Convert Cartesian coordinates to polar coordinates.
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)


def header(self):
        dk_xy = self.dk_xy
        dk_z = self.dk_z
        hdr = fits.Header()
        hdr["HDUNAME"] = ("PS2D", "block name")
        hdr["CONTENT"] = ("2D cylindrically averaged power spectrum",
                          "data product")
        hdr["BUNIT"] = ("%s^2 Mpc^3" % self.unit, "data unit")
        if self.meanstd:
            hdr["AvgType"] = ("mean + standard deviation", "average type")
        else:
            hdr["AvgType"] = ("median + 1.4826*MAD", "average type")

        hdr["WINDOW"] = (self.window_name,
                         "window applied along frequency axis")

        # Physical coordinates: IRAF LTM/LTV
        # Li{Image} = LTMi_i * Pi{Physical} + LTVi
        # Reference: ftp://iraf.noao.edu/iraf/web/projects/fitswcs/specwcs.html
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
        hdr["PixSize"] = (self.pixelsize, "[arcsec] data cube pixel size")
        hdr["Z_C"] = (self.zc, "data cube central redshift")
        hdr["Freq_C"] = (self.freqc, "[MHz] data cube central frequency")
        hdr["Freq_Min"] = (self.frequencies.min(),
                           "[MHz] data cube minimum frequency")
        hdr["Freq_Max"] = (self.frequencies.max(),
                           "[MHz] data cube maximum frequency")
        # Command history
        hdr.add_history(" ".join(sys.argv))
        return hdr


def plot(ps2d,k_perp,k_los,meanstd,unit, ax, ax_err, colormap="jet"):
        """
        Plot the calculated 2D power spectrum.
        """
        x = k_perp
        y = k_los

        if meanstd:
            title = "2D Power Spectrum (mean)"
            title_err = "Error (standard deviation)"
        else:
            title = "2D Power Spectrum (median)"
            title_err = "Error (1.4826*MAD)"

        # median/mean
        mappable = ax.pcolormesh(x[1:], y[1:],
                                 np.log10(ps2d[0, 1:, 1:]),
                                 cmap=colormap)
        vmin, vmax = mappable.get_clim()
        ax.set(xscale="log", yscale="log",
               xlim=(x[1], x[-1]), ylim=(y[1], y[-1]),
               xlabel=r"$k_{\perp}$ [Mpc$^{-1}$]",
               ylabel=r"$k_{||}$ [Mpc$^{-1}$]",
               title=title)
        cb = ax.figure.colorbar(mappable, ax=ax, pad=0.01, aspect=30)
        cb.ax.set_xlabel(r"[%s$^2$ Mpc$^3$]" % unit)

        # error
        mappable = ax_err.pcolormesh(x[1:], y[1:],
                                     np.log10(ps2d[1, 1:, 1:]),
                                     cmap=colormap)
        mappable.set_clim(vmin, vmax)
        ax_err.set(xscale="log", yscale="log",
                   xlim=(x[1], x[-1]), ylim=(y[1], y[-1]),
                   xlabel=r"$k_{\perp}$ [Mpc$^{-1}$]",
                   ylabel=r"$k_{||}$ [Mpc$^{-1}$]",
                   title=title_err)
        cb = ax_err.figure.colorbar(mappable, ax=ax_err, pad=0.01, aspect=30)
        cb.ax.set_xlabel(r"[%s$^2$ Mpc$^3$]" % unit)

        return (ax, ax_err)


def rmean(data, axis=0):
    '''Remove the mean in direction axis'''
    return data - np.mean(data, axis=axis)

def robust_freq_width(freqs):
    '''Return frequency width, robust to gaps in freqs'''
    dfs = np.diff(freqs)
    m, idx, c = np.unique(np.round(dfs * 1e-3) * 1e3, return_counts=True, return_inverse=True)
    return dfs[np.where(idx == np.argmax(c))].mean()


def nudft(x, y, M=None, w=None, dx=None):
    """Non uniform discrete Fourier transform

    Args:
        x (array): x axis
        y (array): y axis
        M (int, optional): Number of Fourier components that will be computed, default to len(x)
        w (array, optional): Tapper

    Returns:
        (array, array): k modes, Fourier transform of y
    """
    if M is None:
        M = len(x)

    if dx is None:
        dx = robust_freq_width(x)

    if w is not None:
        y = y * w

    df = 1 / (dx * M)
    k = df * np.arange(-(M // 2), M - (M // 2))

    X = np.exp(2 * np.pi * 1j * k * x[:, np.newaxis])

    return k, np.tensordot(y, X.conj().T, axes=[0, 1]).conj().T


def lssa_cov(x, C, M, dx=None):
    if dx is None:
        dx = robust_freq_width(x)

    W = np.linalg.pinv(C)

    k = np.fft.fftshift(np.fft.fftfreq(M, float(dx)))

    A = np.exp(-2. * np.pi * 1j * k * x[:, np.newaxis]) / len(x)

    return np.linalg.pinv(np.dot(np.dot(A.real.T, W), A.real))


def do_weighted_lssa(A, y, w):
    n_x, n_k = A.shape
    n_modes = y.shape[1]

    w = w.astype(np.complex128)
    w.imag = w.real

    d = np.zeros((n_k, n_modes), dtype=np.complex128)

    for i in xrange(n_modes):
        C = diags(w[:, i])
        A_C = C.T.dot(A).T
        Y = np.dot(np.linalg.pinv(np.dot(A_C, A)), A_C)
        d[:, i] = np.dot(y[:, i].T, Y.T).T

    return d


def lssa(x, y, M, w=None, weights=None, dx=None):
    """Least-squares spectral analysis

    Args:
        x (array): x axis
        y (array): y axis
        M (int, optional): Number of Fourier components that will be computed, default to len(x)
        w (array, optional): Tapper
        weights (array, optional): Weights

    Returns:
        (array, array): k modes, Fourier transform of y
    """
    if dx is None:
        dx = robust_freq_width(x)

    k = np.fft.fftshift(np.fft.fftfreq(M, float(dx)))

    if w is not None:
        y = y * w  # [:, np.newaxis]

    A = np.exp(-2. * np.pi * 1j * k * x[:, np.newaxis]) / len(x)

    if weights is None:
        Y = np.dot(np.linalg.pinv(np.dot(A.conj().T, A)), A.conj().T)
        d = np.tensordot(y.conj().T, Y.conj().T, axes=[0, 0]).conj().T
    else:
        d = do_weighted_lssa(A, y, weights)

    return k, d


def ufft(x, y, M=None, w=None, weights=None, dx=None):
    """SImple FFT

    Args:
        x (array): x axis
        y (array): y axis
        M (int, optional): Number of Fourier components that will be computed, default to len(x)
        w (array, optional): Tapper
        weights (array, optional): Weights

    Returns:
        (array, array): k modes, Fourier transform of y
    """
    if M is None:
        M = len(x)

    if dx is None:
        dx = robust_freq_width(x)

    k = np.fft.fftshift(np.fft.fftfreq(M, float(dx)))

    if w is not None:
        y = y * w  # [:, np.newaxis]

    d = np.fft.fftshift(fft(np.fft.fftshift(y), axis=0, n=M))

    return k, d


def get_delay(freqs, M=None, dx=None, half=True):
    ''' Convert frequencies to delay '''
    if dx is None:
        dx = robust_freq_width(freqs)
    if M is None:
        M = len(freqs)

    df = 1 / (dx * M)
    delay = df * np.arange(-(M // 2), M - (M // 2))

    if half:
        M = len(delay)
        delay = delay[M // 2 + 1:]

    return delay


def delay_transform_cube(freqs, ft_cube, M=None, window=None, dx=None,
                         method='nudft', weights=None, rmean_axis=None):
    '''Frequency -> delay transform ft_cube'''
    if method == 'nudft':
        delay, dft_cube = nudft(freqs, ft_cube, M=M, w=window, dx=dx)
    elif method == 'lssa':
        delay, dft_cube = lssa(freqs, rmean(ft_cube, axis=rmean_axis), M=M, w=window, dx=dx)
    elif method == 'wlssa':
        delay, dft_cube = lssa(freqs, rmean(ft_cube, axis=rmean_axis), M=M, w=window, dx=dx, weights=weights)
    elif method == 'ufft':
        delay, dft_cube = ufft(freqs, ft_cube , M=M, w=window, dx=dx, weights=weights)
    else:
        print("'method' should be one of: nudft, lssa, wlssa, ufft")
    #dft_cube[delay == 0] = np.mean(ft_cube, axis=0)

    return delay, dft_cube







def delay_filter_leastsq_1d(data, flags, sigma, nmax, add_noise=False,
                            cn_guess=None, use_linear=True, operator=None, fundamental_period=None):
    """
    Fit a smooth model to 1D complex-valued data with flags, using a linear
    least-squares solver. The model is a Fourier series up to a specified
    order. As well as calculating a best-fit model, this will also return a
    copy of the data with flagged regions filled in ('in-painted') with the
    smooth solution.
    Optionally, you can also add an uncorrelated noise realization on top of
    the smooth model in the flagged region.
    Parameters
    ----------
    data : array_like, complex
        Complex visibility array as a function of frequency, with shape
        (Nfreqs,).
    flags : array_like, bool
        Boolean flags with the same shape as data.
    sigma : float or array_like
        Noise standard deviation, in the same units as the data. If float,
        assumed to be homogeneous in frequency. If array_like, must have
        the same shape as the data.
        Note that the choice of sigma will have some bearing on how sensitive
        the fits are to small-scale variations.
    nmax: int or 2-tuple of ints
        Max. order of Fourier modes to fit. A model with complex Fourier modes
        between [-n, n] will be fitted to the data, where the Fourier basis
        functions are ~ exp(-i 2 pi n nu / (Delta nu). If 2-tuple fit [-n0, n1].
    add_noise : bool, optional
        Whether to add an unconstrained noise realization to the in-painted areas.
        This uses sigma to set the noise standard deviation. Default: False.
    cn_guess : array_like, optional
        Initial guess for the series coefficients. If None, zeros will be used.
        A sensible choice of cn_guess can speed up the solver significantly.
        Default: None.
    use_linear : bool, optional
        Whether to use a fast linear least-squares solver to fit the Fourier
        coefficients, or a slower generalized least-squares solver.
        Default: True.
    operator : array_like, optional
        Fourier basis operator matrix. This is used to pass in a pre-computed
        matrix operator when calling from other functions, e.g. from
        delay_filter_leastsq. Operator must have shape (Nmodes, Nfreq), where
        Nmodes = 2*nmax + 1. A complex Fourier basis will be automatically
        calculated if no operator is specified.
    fundamental_period : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.
    Returns
    -------
    model : array_like
        Best-fit model, composed of a sum of Fourier modes.
    model_coeffs : array_like
        Coefficients of Fourier modes, ordered from modes [-nmax, +nmax].
    data_out : array_like
        In-painted data.
    """
    # Construct Fourier basis operator if not specified
    if isinstance(nmax, tuple) or isinstance(nmax, list):
        nmin = nmax[0]
        nmax = nmax[1]
        assert isinstance(nmin, int) and isinstance(nmax, int), "Provide integers for nmax and nmin"
    elif isinstance(nmax, int):
        nmin = -nmax
    if operator is None:
        F = fourier_operator(dsize=data.size, nmin = nmin, nmax=nmax, L=fundamental_period)
    else:
        F = operator
        cshape = nmax - nmin + 1
        if F.shape[0] != cshape:
            raise ValueError("Fourier basis operator has the wrong shape. "
                             "Must have shape (Nmodes, Nfreq).")
    # Turn flags into a mask
    #w = np.logical_not(flags)
    w = flags
    #print(w*data)
    # Define model and likelihood function
    def model(cn, F):
        return np.dot(cn, F)

    nmodes = nmax - nmin + 1

    # Initial guess for Fourier coefficients (real + imaginary blocks)
    cn_in = np.zeros(2 * nmodes)
    if cn_guess is not None:
        if cn_in.size != 2 * cn_guess.size:
            raise ValueError("cn_guess must be of size %s" % (cn_in.size / 2))
        cn_in[:cn_guess.shape[0]] = cn_guess.real
        cn_in[cn_guess.shape[0]:] = cn_guess.imag

    # Make sure sigma is the right size for matrix broadcasting
    if isinstance(sigma, np.ndarray):
        mat_sigma = np.tile(sigma, (nmodes, 1)).T
    else:
        mat_sigma = sigma

    # Run least-squares fit
    if use_linear:
        # Solve as linear system
        A = np.atleast_2d(w).T * F.T
        res = lsq_linear(A / mat_sigma ** 2., w * data / sigma ** 2.)
        cn_out = res.x
    else:
        # Use full non-linear leastsq fit
        def loglike(cn):
            """
            Simple log-likelihood, assuming Gaussian data. Calculates:
                logL = -0.5 [w*(data - model)]^2 / sigma^2.
            """
            # Need to do real and imaginary parts separately, otherwise
            # leastsq() fails
            _delta = w * (data - model(cn[:nmodes] + 1.j * cn[nmodes:], F))
            delta = np.concatenate((_delta.real / sigma, _delta.imag / sigma))
            return -0.5 * delta**2.

        # Do non-linear least-squares calculation
        cn, stat = leastsq(loglike, cn_in)
        cn_out = cn[:nmodes] + 1.j * cn[nmodes:]

    # Inject smooth best-fit model into masked areas
    bf_model = model(cn_out, F)
    data_out = data.copy()
    data_out[flags] = bf_model[flags]

    # Add noise to in-painted regions if requested
    if add_noise:
        noise = np.random.randn(np.sum(flags)) \
            + 1.j * np.random.randn(np.sum(flags))
        if isinstance(sigma, np.ndarray):
            data_out[flags] += sigma[flags] * noise
        else:
            data_out[flags] += sigma * noise

    # Return coefficients and best-fit model
    return bf_model, cn_out, data_out

def fourier_operator(dsize, nmax, nmin=None, L=None):
    """
    Return a complex Fourier analysis operator for a given data dimension and number of Fourier modes.
    Parameters
    ----------
    dsize : int
        Size of data array.
    nmax : int
        Maximum Fourier mode number. Modes will be constructed between
        [nmin, nmax], for a total of (nmax - min) + 1 modes.
    nmin : int, optional, default nmin = nmax
        minimum integer of fourier mode numbers. Modes will be constructed between
        [nmin, nmax] for total of (nmax - nmin) + 1 modes.
    L : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.
    Returns
    -------
    F : array_like
        Fourier matrix operator, of shape (Nmodes, Ndata)
    """
    nu = np.arange(dsize)
    if L is None:
        L = nu[-1] - nu[0]
    if nmin is None:
        nmin = -nmax
    # Construct frequency array (*not* in physical frequency units)
    # Build matrix operator for complex Fourier basis
    n = np.arange(nmin, nmax + 1)
    F = np.array([np.exp(-1.j * _n * nu / L) for _n in n])
    return F

def delay_filter_leastsq(data, flags, sigma, nmax, add_noise=False,
                         cn_guess=None, use_linear=True, operator=None, fundamental_period=None):
    """
    Fit a smooth model to each 1D slice of 2D complex-valued data with flags,
    using a linear least-squares solver. The model is a Fourier series up to a
    specified order. As well as calculating a best-fit model, this will also
    return a copy of the data with flagged regions filled in ('in-painted')
    with the smooth solution.
    Optionally, you can also add an uncorrelated noise realization on top of
    the smooth model in the flagged region.
    N.B. This is just a wrapper around delay_filter_leastsq_1d() but with some
    time-saving precomputations. It fits to each 1D slice of the data
    individually, and does not perform a global fit to the 2D data.
    Parameters
    ----------
    data : array_like, complex
        Complex visibility array as a function of frequency, with shape
        (Ntimes, Nfreqs).
    flags : array_like, bool
        Boolean flags with the same shape as data.
    sigma : float or array_like
        Noise standard deviation, in the same units as the data. If float,
        assumed to be homogeneous in frequency. If array_like, must have
        the same shape as the data.
        Note that the choice of sigma will have some bearing on how sensitive
        the fits are to small-scale variations.
    nmax: int
        Max. order of Fourier modes to fit. A model with complex Fourier modes
        between [-n, n] will be fitted to the data, where the Fourier basis
        functions are ~ exp(-i 2 pi n nu / (Delta nu).
    add_noise : bool, optional
        Whether to add an unconstrained noise realization to the in-painted areas.
        This uses sigma to set the noise standard deviation. Default: False.
    cn_guess : array_like, optional
        Initial guess for the series coefficients of the first row of the
        input data. If None, zeros will be used. Default: None.
    use_linear : bool, optional
        Whether to use a fast linear least-squares solver to fit the Fourier
        coefficients, or a slower generalized least-squares solver.
        Default: True.
    operator : array_like, optional
        Fourier basis operator matrix. Must have shape (Nmodes, Nfreq), where
        Nmodes = 2*nmax + 1. A complex Fourier basis will be used by default.
    fundamental_period : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.
    Returns
    -------
    model : array_like
        Best-fit model, composed of a sum of Fourier modes. Same shape as the
        data.
    model_coeffs : array_like
        Coefficients of Fourier modes, ordered from modes [-n, +n].
    data_out : array_like
        In-painted data.
    """
    if isinstance(nmax, tuple) or isinstance(nmax, list):
        nmin = nmax[0]
        nmax = nmax[1]
        assert isinstance(nmin, int) and isinstance(nmax, int), "Provide integers for nmax and nmin"
    elif isinstance(nmax, int):
        nmin = -nmax
    # Construct and cache Fourier basis operator (for speed)
    if operator is None:
        F = fourier_operator(dsize=data.shape[1], nmax=nmax, nmin=nmin, L=fundamental_period)
    else:
        # delay_filter_leastsq_1d will check for correct dimensions
        F = operator

    nmodes = nmax - nmin + 1
    # Array to store in-painted data
    inp_data = np.zeros(data.shape, dtype=np.complex)
    cn_array = np.zeros((data.shape[0], nmodes), dtype=np.complex)
    mdl_array = np.zeros(data.shape, dtype=np.complex)

    # Loop over array
    cn_out = None
    for i in range(data.shape[0]):
        bf_model, cn_out, data_out = delay_filter_leastsq_1d(
            data[i], flags[i], sigma=sigma, nmax=(nmin, nmax), add_noise=add_noise,
            use_linear=use_linear, cn_guess=cn_out, operator=F, fundamental_period=fundamental_period)
        inp_data[i, :] = data_out
        cn_array[i, :] = cn_out
        mdl_array[i, :] = bf_model

    return mdl_array, cn_array, inp_data





def get_filter_area(x, filter_center, filter_width):
    """
    Return an 'area' vector demarking where cleaning should be allowed
    to take place.
    Arguments:
        x : array-like real space vector listing where data is sampled.
        filter_center : center of the area to be cleaned. Units of 1/(x-units)
        filter_width : width of the region of area to be cleaned. Units of 1/(x-units)
    """
    nx = len(x)
    dx = np.mean(np.diff(x))
    if not np.isinf(filter_width):
        av = np.ones(len(x))
        filter_size = ((-filter_center + filter_width), (filter_center + filter_width))
        ut, lt = calc_width(filter_size, dx, nx)
        av[ut:lt] = 0.
    else:
        av = np.ones(nx)
    return av



def fourier_filter_hash(filter_centers, filter_half_widths,
                         filter_factors, x, w=None, hash_decimal=10, **kwargs):
    '''
    Generate a hash key for a fourier filter
    Parameters
    ----------
        filter_centers: list,
                        list of floats for filter centers
        filter_half_widths: list
                        list of float filter half widths (in fourier space)
        filter_factors: list
                        list of float filter factors
        x: the x-axis of the data to be subjected to the hashed filter.
        w: optional vector of float weights to hash to. default, none
        hash_decimal: number of decimals to use for floats in key.
        kwargs: additional hashable elements the user would like to
                include in their filter key.
    Returns
    -------
    A key for fourier_filter arrays hasing the information provided in the args.
    '''
    filter_key = ('x:',) + tuple(np.round(x,hash_decimal))\
    + ('filter_centers x N x DF:',) + tuple(np.round(np.asarray(filter_centers) * np.mean(np.diff(x)) * len(x), hash_decimal))\
    + ('filter_half_widths x N x DF:',) + tuple(np.round(np.asarray(filter_half_widths) * np.mean(np.diff(x)) * len(x), hash_decimal))\
    + ('filter_factors x 1e9:',) + tuple(np.round(np.asarray(filter_factors) * 1e9, hash_decimal))
    if w is not None:
        filter_key = filter_key + ('weights', ) +  tuple(np.round(w.tolist(), hash_decimal))
    filter_key = filter_key + tuple([kwargs[k] for k in kwargs])
    return filter_key

def calc_width(filter_size, real_delta, nsamples):
    '''Calculate the upper and lower bin indices of a fourier filter
    Arguments:
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
            Alternatively, can be fed as len-2 tuple specifying the absolute value of the negative
            and positive bound of the filter in fourier space respectively.
            Example: (20, 40) --> (-20 < tau < 40)
        real_delta: the bin width in real space
        nsamples: the number of samples in the array to be filtered
    Returns:
        uthresh, lthresh: bin indices for filtered bins started at uthresh (which is filtered)
            and ending at lthresh (which is a negative integer and also not filtered).
            Designed for area = np.ones(nsamples, dtype=np.int); area[uthresh:lthresh] = 0
    '''
    if isinstance(filter_size, (list, tuple, np.ndarray)):
        _, l = calc_width(np.abs(filter_size[0]), real_delta, nsamples)
        u, _ = calc_width(np.abs(filter_size[1]), real_delta, nsamples)
        return (u, l)
    bin_width = 1.0 / (real_delta * nsamples)
    w = int(np.around(filter_size / bin_width))
    uthresh, lthresh = w + 1, -w
    if lthresh == 0:
        lthresh = nsamples
    return (uthresh, lthresh)

def get_bl_dly(bl_len, horizon=1., standoff=0., min_dly=0.):
    # construct baseline delay
    bl_dly = horizon * bl_len + standoff

    # check minimum delay
    bl_dly = np.max([bl_dly, min_dly])

    return bl_dly


def gen_window(window, N, alpha=0.5, edgecut_low=0, edgecut_hi=0, normalization=None, **kwargs):
    """
    Generate a 1D window function of length N.
    Args:
        window : str, window function
        N : int, number of channels for windowing function.
        edgecut_low : int, number of bins to consider as zero-padded at the low-side
            of the array, such that the window smoothly connects to zero.
        edgecut_hi : int, number of bins to consider as zero-padded at the high-side
            of the array, such that the window smoothly connects to zero.
        alpha : if window is 'tukey', this is its alpha parameter.
        normalization : str, optional
            set to 'rms' to divide by rms and 'mean' to divide by mean.
    """
    if normalization is not None:
        if normalization not in ["mean", "rms"]:
            raise ValueError("normalization must be one of ['rms', 'mean']")
    # parse multiple input window or special windows
    w = np.zeros(N, dtype=np.float)
    Ncut = edgecut_low + edgecut_hi
    if Ncut >= N:
        raise ValueError("Ncut >= N for edgecut_low {} and edgecut_hi {}".format(edgecut_low, edgecut_hi))
    if edgecut_hi > 0:
        edgecut_hi = -edgecut_hi
    else:
        edgecut_hi = None
    if window in ['none', None, 'None', 'boxcar', 'tophat']:
        w[edgecut_low:edgecut_hi] = windows.boxcar(N - Ncut)
    elif window in ['blackmanharris', 'blackman-harris', 'bh', 'bh4']:
        w[edgecut_low:edgecut_hi] =  windows.blackmanharris(N - Ncut)
    elif window in ['hanning', 'hann']:
        w[edgecut_low:edgecut_hi] =  windows.hann(N - Ncut)
    elif window == 'tukey':
        w[edgecut_low:edgecut_hi] =  windows.tukey(N - Ncut, alpha)
    elif window in ['blackmanharris-7term', 'blackman-harris-7term', 'bh7']:
        # https://ieeexplore.ieee.org/document/293419
        a_k = [0.27105140069342, 0.43329793923448, 0.21812299954311, 0.06592544638803, 0.01081174209837,
              0.00077658482522, 0.00001388721735]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    elif window in ['cosinesum-9term', 'cosinesum9term', 'cs9']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.384331152777942e-1, 4.00554534864382e-1, 2.358242530472107e-1, 9.527918858383112e-2,
               2.537395516617152e-2, 4.152432907505835e-3, 3.68560416329818e-4, 1.38435559391703e-5,
               1.161808358932861e-7]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    elif window in ['cosinesum-11term', 'cosinesum11term', 'cs11']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.151527506679809e-1, 3.731348357785249e-1, 2.424243358446660e-1, 1.166907592689211e-1,
               4.077422105878731e-2, 1.000904500852923e-2, 1.639806917362033e-3, 1.651660820997142e-4,
               8.884663168541479e-6, 1.938617116029048e-7, 8.482485599330470e-10]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    else:
        try:
            # return any single-arg window from windows
            w[edgecut_low:edgecut_hi] = getattr(windows, window)(N - Ncut)
        except AttributeError:
            raise ValueError("Didn't recognize window {}".format(window))
    if normalization == 'rms':
        w /= np.sqrt(np.mean(np.abs(w)**2.))
    if normalization == 'mean':
        w /= w.mean()
    return w


def clean_filter(x, data, wgts, filter_centers, filter_half_widths,
                  clean2d=False, tol=1e-9, window='none', skip_wgt=0.1,
                  maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                  edgecut_low=0, edgecut_hi=0, add_clean_residual=False,
                  zero_residual_flags=True):
    '''
    core cleaning functionality
    Input sanitation not implemented. Should be called through
    fourier_filter and the higher level functions that call fourier_filter.
    Parameters
    ----------
    x : array-like (or 2-tuple/list of arrays for filter2d)
        x-values of data to be cleaned. Each x-axis must be equally spaced.
    data : array-like, complex, 1d or 2d numpy array of data to be filtered.
    wgts : array-like, float, 1d or 2d numpy array of wgts for data.
    filter_centers : list of floats (1d clean) 2-list of lists of floats (2d clean)
                     centers of filtering regions in units of 1 / x-units
    filter_half_widths : list of floats (1d clean) 2-list of lists of floats (2d clean)
                     half-widths of filtering regions in units of 1 / x-units
    clean2d : bool, optional, specify if 2dclean is to be performed.
              if False, just clean axis -1.
    tol : float, tolerance parameter for clean.
    window : str, apodization to perform on data before cleaning.
    skip_wgt : float, If less then skip_wgt fraction of data is flagged, skip the clean.
    maxiter : int, maximum number of clean iterations.
    gain : float, fraction of detected peak to subtract on each clean iteration.
    filt2d_mode : str, only applies if clean2d == True. options = ['rect', 'plus']
        If 'rect', a 2D rectangular filter is constructed in fourier space (default).
        If 'plus', the 'rect' filter is first constructed, but only the plus-shaped
        slice along 0 delay and fringe-rate is kept.
    edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
        such that the windowing function smoothly approaches zero. For 2D cleaning, can
        be fed as a tuple specifying edgecut_low for first and second FFT axis.
    edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
        such that the windowing function smoothly approaches zero. For 2D cleaning, can
        be fed as a tuple specifying edgecut_hi for first and second FFT axis.
    add_clean_residual : bool, if True, adds the CLEAN residual within the CLEAN bounds
        in fourier space to the CLEAN model. Note that the residual actually returned is
        not the CLEAN residual, but the residual in input data space.
    zero_residual_flags : bool, optional.
        If true, set flagged channels in the residual equal to zero.
        Default is True.
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    if not clean2d:
        #pad = [0, pad]
        _x = [np.fft.fftfreq(len(x), x[1]-x[0])]
        x = [x]
        edgecut_hi = [edgecut_hi]
        edgecut_low = [edgecut_low]
        filter_centers = [copy.deepcopy(filter_centers)]
        filter_half_widths = [copy.deepcopy(filter_half_widths)]
        window = [window]

    window = [gen_window(window[m], data.shape[m], alpha=alpha,
                       edgecut_low=edgecut_low[m], edgecut_hi=edgecut_hi[m]) for m in range(1)]
   # window[0] = np.atleast_2d(window[0]).T
    area_vecs = [ np.zeros(len(_x[m])) for m in range(1) ]
    #set area equal to one inside of filtering regions
    info = {}
    info['filter_params'] = {'axis_0':{}, 'axis_1':{}}
    info['clean_status'] = {'axis_0':{}, 'axis_1':{}}
    info['status'] = {'axis_0':{}, 'axis_1':{}}
    if filt2d_mode == 'rect' or not clean2d:
        for m in range(1):
            for fc, fw in zip(filter_centers[m], filter_half_widths[m]):
                area_vecs[m] = get_filter_area(x[m], fc, fw)
        area = area_vecs[0]



    _wgts = np.fft.ifft( window[0] * wgts , axis=-1)
    _data = np.fft.ifft( window[0] * wgts * data , axis=-1)
    _d_cl = np.zeros_like(_data)
    _d_res = np.zeros_like(_data)
 
    _d_cl, _info = aipy.deconv.clean(_data, _wgts, area=area, tol=tol, stop_if_div=False,
                                                maxiter=maxiter, gain=gain)
    _d_res = _info['res']
    _info['skipped'] = False
 #   del(_info['res'])
    info['clean_status']['axis_1'] = _info
    info['status']['axis_1'] = 'success'

    if add_clean_residual:
        _d_cl = _d_cl + _d_res * area
    if clean2d:
        model = np.fft.fft2(_d_cl)
    else:
        model = np.fft.fft(_d_cl, axis=-1)
    #transpose back if filtering the 0th dimension.
    residual = (data - model)
 
    return model, residual, info



def _clean_filter_hera(x, data, wgts, filter_centers, filter_half_widths,
                  clean2d=False, tol=1e-9, window='none', skip_wgt=0.1,
                  maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                  edgecut_low=0, edgecut_hi=0, add_clean_residual=False,
                  zero_residual_flags=True):
    '''
    core cleaning functionality
    Input sanitation not implemented. Should be called through
    fourier_filter and the higher level functions that call fourier_filter.
    Parameters
    ----------
    x : array-like (or 2-tuple/list of arrays for filter2d)
        x-values of data to be cleaned. Each x-axis must be equally spaced.
    data : array-like, complex, 1d or 2d numpy array of data to be filtered.
    wgts : array-like, float, 1d or 2d numpy array of wgts for data.
    filter_centers : list of floats (1d clean) 2-list of lists of floats (2d clean)
                     centers of filtering regions in units of 1 / x-units
    filter_half_widths : list of floats (1d clean) 2-list of lists of floats (2d clean)
                     half-widths of filtering regions in units of 1 / x-units
    clean2d : bool, optional, specify if 2dclean is to be performed.
              if False, just clean axis -1.
    tol : float, tolerance parameter for clean.
    window : str, apodization to perform on data before cleaning.
    skip_wgt : float, If less then skip_wgt fraction of data is flagged, skip the clean.
    maxiter : int, maximum number of clean iterations.
    gain : float, fraction of detected peak to subtract on each clean iteration.
    filt2d_mode : str, only applies if clean2d == True. options = ['rect', 'plus']
        If 'rect', a 2D rectangular filter is constructed in fourier space (default).
        If 'plus', the 'rect' filter is first constructed, but only the plus-shaped
        slice along 0 delay and fringe-rate is kept.
    edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
        such that the windowing function smoothly approaches zero. For 2D cleaning, can
        be fed as a tuple specifying edgecut_low for first and second FFT axis.
    edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
        such that the windowing function smoothly approaches zero. For 2D cleaning, can
        be fed as a tuple specifying edgecut_hi for first and second FFT axis.
    add_clean_residual : bool, if True, adds the CLEAN residual within the CLEAN bounds
        in fourier space to the CLEAN model. Note that the residual actually returned is
        not the CLEAN residual, but the residual in input data space.
    zero_residual_flags : bool, optional.
        If true, set flagged channels in the residual equal to zero.
        Default is True.
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    if not clean2d:
        #pad = [0, pad]
        _x = [np.zeros(data.shape[0]), np.fft.fftfreq(len(x), x[1]-x[0])]
        x = [np.zeros(data.shape[0]), x]
        edgecut_hi = [ 0, edgecut_hi ]
        edgecut_low = [ 0, edgecut_low ]
        filter_centers = [[0.], copy.deepcopy(filter_centers)]
        filter_half_widths = [[np.inf], copy.deepcopy(filter_half_widths)]
        window = ['none', window]
    else:
        if not np.all(np.isclose(np.diff(x[1]), np.mean(np.diff(x[1])))):
            raise ValueError("Data must be equally spaced for CLEAN mode!")
        _x = [np.fft.fftfreq(len(x[m]), x[m][1]-x[m][0]) for m in range(2)]
        #window_opt = window
    for m in range(2):
        if not np.all(np.isclose(np.diff(x[m]), np.mean(np.diff(x[m])))):
            raise ValueError("Data must be equally spaced for CLEAN mode!")
    window = [gen_window(window[m], data.shape[m], alpha=alpha,
                       edgecut_low=edgecut_low[m], edgecut_hi=edgecut_hi[m]) for m in range(1)]
    window[0] = np.atleast_2d(window[0]).T
    area_vecs = [ np.zeros(len(_x[m])) for m in range(2) ]
    #set area equal to one inside of filtering regions
    info = {}
    info['filter_params'] = {'axis_0':{}, 'axis_1':{}}
    info['clean_status'] = {'axis_0':{}, 'axis_1':{}}
    info['status'] = {'axis_0':{}, 'axis_1':{}}
    if filt2d_mode == 'rect' or not clean2d:
        for m in range(2):
            for fc, fw in zip(filter_centers[m], filter_half_widths[m]):
                area_vecs[m] = _get_filter_area(x[m], fc, fw)
        area = np.outer(area_vecs[0], area_vecs[1])
    elif filt2d_mode == 'plus' and clean2d:
        area = np.zeros(data.shape)
        #construct and add a 'plus' for each filtering window pair in each dimension.
        for fc0, fw0 in zip(filter_centers[0], filter_half_widths[0]):
            for fc1, fw1 in zip(filter_centers[1], filter_half_widths[1]):
                area_temp = np.zeros(area.shape)
                if fc0 >= _x[0].min() and fc0 <= _x[0].max():
                    #generate area vector centered at zero
                    av = _get_filter_area(x[1], fc1, fw1)
                    area_temp[np.argmin(np.abs(_x[0]-fc0)), :] = av
                if fc1 >= _x[1].min() and fc1 <= _x[1].max():
                    #generate area vector centered at zero
                    av = _get_filter_area(x[0], fc0, fw0)
                    area_temp[:, np.argmin(np.abs(_x[1]-fc1))] = av
                area += area_temp
        area = (area>0.).astype(int)
    else:
        raise ValueError("%s is not a valid filt2d_mode! choose from ['rect', 'plus']"%(filt2d_mode))
    if clean2d:
        _wgts = np.fft.ifft2(window[0] * wgts * window[1])
        _data = np.fft.ifft2(window[0] * data * wgts * window[1])
    else:
        _wgts = np.fft.ifft(window[0] * wgts * window[1], axis=1)
        _data = np.fft.ifft(window[0] * wgts * data * window[1], axis=1)
    _d_cl = np.zeros_like(_data)
    _d_res = np.zeros_like(_data)
    if not clean2d:
        for i, _d, _w, _a in zip(np.arange(_data.shape[0]).astype(int), _data, _wgts, area):
            # we skip steps that might trigger infinite CLEAN loops or divergent behavior.
            # if the weights sum up to a value close to zero (most of the data is flagged)
            # or if the data itself is close to zero.
            if _w[0] < skip_wgt or np.all(np.isclose(_d, 0.)):
                _d_cl[i] = 0.
                _d_res[i] = _d
                info['status']['axis_1'][i] = 'skipped'
            else:
                _d_cl[i], _info = aipy.deconv.clean(_d, _w, area=_a, tol=tol, stop_if_div=False,
                                                maxiter=maxiter, gain=gain)
                _d_res[i] = _info['res']
                _info['skipped'] = False
                del(_info['res'])
                info['clean_status']['axis_1'][i] = _info
                info['status']['axis_1'][i] = 'success'
    elif clean2d:
            # we skip 2d cleans if all the data is close to zero (which can cause an infinite clean loop)
            # or the weights are all equal to zero which can also lead to a clean loop.
            # the maximum of _wgts should be the average value of all cells in 2d wgts.
            # since it is the 2d fft of wgts.
            if not np.all(np.isclose(_data, 0.)) and np.abs(_wgts).max() > skip_wgt:
                _d_cl, _info = aipy.deconv.clean(_data, _wgts, area=area, tol=tol, stop_if_div=False,
                                                maxiter=maxiter, gain=gain)
                _d_res = _info['res']
                del(_info['res'])
                info['clean_status']['axis_1'] = _info
                info['clean_status']['axis_0'] = info['clean_status']['axis_1']
                info['status']['axis_1'] = {i:'success' for i in range(_data.shape[0])}
                info['status']['axis_0'] = {i:'success' for i in range(_data.shape[1])}
            else:
                info['clean_status']['axis_0'] = {'skipped':True}
                info['clean_status']['axis_1'] = {'skipped':True}
                info['status']['axis_1'] = {i:'skipped' for i in range(_data.shape[0])}
                info['status']['axis_0'] = {i:'skipped' for i in range(_data.shape[1])}
                _d_cl = np.zeros_like(_data)
                _d_res = np.zeros_like(_d_cl)
    if add_clean_residual:
        _d_cl = _d_cl + _d_res * area
    if clean2d:
        model = np.fft.fft2(_d_cl)
    else:
        model = np.fft.fft(_d_cl, axis=1)
    #transpose back if filtering the 0th dimension.
    residual = (data - model)
    if zero_residual_flags:
        windmat = np.outer(window[0], window[1])
        residual *= (~np.isclose(wgts * windmat, 0.0, atol=1e-10)).astype(float)
    return model, residual, info

def _get_filter_area(x, filter_center, filter_width):
    """
    Return an 'area' vector demarking where cleaning should be allowed
    to take place.
    Arguments:
        x : array-like real space vector listing where data is sampled.
        filter_center : center of the area to be cleaned. Units of 1/(x-units)
        filter_width : width of the region of area to be cleaned. Units of 1/(x-units)
    """
    nx = len(x)
    dx = np.mean(np.diff(x))
    if not np.isinf(filter_width):
        av = np.ones(len(x))
        filter_size = ((-filter_center + filter_width), (filter_center + filter_width))
        ut, lt = calc_width(filter_size, dx, nx)
        av[ut:lt] = 0.
    else:
        av = np.ones(nx)
    return av

def wedge_filter(data, wgts, bl_len, sdf, standoff=0., horizon=1., min_dly=0.0, skip_wgt=0.5,
                 mode='clean', **kwargs):
    '''Apply a wideband delay filter to data. Variable names preserved for
        backward compatability with capo/PAPER analysis.
    Arguments:
        data: 1D or 2D (real or complex) numpy array where last dimension is frequency.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data.
        bl_len: length of baseline (in 1/[sdf], typically ns)
        sdf: frequency channel width (typically in GHz)
        standoff: fixed additional delay beyond the horizon (same units as bl_len)
        horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
        min_dly: a minimum delay used for cleaning: if bl_dly < min_dly, use min_dly. same units as bl_len
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        mode: filtering mode (see supported modes in fourier_filter docstring)
        kwargs: see fourier_filter documentation
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    # get bl delay
    bl_dly = _get_bl_dly(bl_len, horizon=horizon, standoff=standoff, min_dly=min_dly)
    return delay_filter(sdf=sdf, data=data, wgts=wgts, max_dly=bl_dly,
                          skip_wgt=skip_wgt, **kwargs)

def delay_filter(data, wgts, max_dly, sdf, skip_wgt=0.5,
                 mode='clean', **kwargs):
    '''Apply a wideband delay filter to data. Variable names preserved for
        backward compatability with capo/PAPER analysis.
    Arguments:
        data: 1D or 2D (real or complex) numpy array where last dimension is frequency.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data.
            max_dly: maximum abs of delay to filter to (around delay = 0.)
        sdf: frequency channel width (typically in GHz)
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        mode: filtering mode (see supported modes in fourier_filter docstring)
        kwargs: see fourier_filter documentation
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    freqs = np.arange(data.shape[-1]) * sdf
    return fourier_filter(x=freqs, data=data, wgts=wgts, filter_centers=[0.], filter_half_widths=[max_dly],
                          skip_wgt=skip_wgt, filter_dims=1, mode=mode, **kwargs)

## Import these for plotting ## 

###### For plotting ######

import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as pl 
import matplotlib 
###################################
pl.rcParams['figure.figsize'] = 8, 7
pl.rcParams['ytick.minor.visible'] =True
pl.rcParams['xtick.minor.visible'] = True
pl.rcParams['xtick.top'] = True
pl.rcParams['ytick.right'] = True
pl.rcParams['font.size'] = '20'
pl.rcParams['legend.fontsize'] = '15'
pl.rcParams['legend.borderaxespad'] = '1.9'
#pl.rcParams['legend.numpoints'] = '1'

pl.rcParams['figure.titlesize'] = 'medium'
pl.rcParams['figure.titlesize'] = 'medium'
pl.rcParams['xtick.major.size'] = '10'
pl.rcParams['xtick.minor.size'] = '6'
pl.rcParams['xtick.major.width'] = '2'
pl.rcParams['xtick.minor.width'] = '1'
pl.rcParams['ytick.major.size'] = '10'
pl.rcParams['ytick.minor.size'] = '6'
pl.rcParams['ytick.major.width'] = '2'
pl.rcParams['ytick.minor.width'] = '1'
pl.rcParams['xtick.direction'] = 'in'
pl.rcParams['ytick.direction'] = 'in'
pl.rcParams['axes.labelpad'] = '10.0'
pl.rcParams['lines.dashed_pattern']=3.0, 1.4
#pl.rcParams['axes.formatter.limits']=-10,10
pl.rcParams['lines.dotted_pattern']= 1.0, 0.7

pl.rcParams['xtick.labelsize'] = '16'
pl.rcParams['ytick.labelsize'] = '16'
pl.rcParams['axes.labelsize'] = '16'
pl.rcParams['axes.labelsize'] = '16'
pl.rcParams['axes.labelweight'] = 'bold'

pl.rcParams['xtick.major.pad']='10'
pl.rcParams['xtick.minor.pad']='10'
#pl.rcParams['hatch.color'] = 'black'
pl.rc('axes', linewidth=2)
