import numpy as np
from scipy import fft
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from .helper import chi2Reduced, findIndex

class Subscan:
    """ A subscan is composed of two fits objects : signal and weight """
    def __init__(self, signal, weight):
        self.signal = signal
        self.weight = weight
        self.corrected = self._computeCorrected(signal.data, weight.data)
        self.fft2 = self._computeFFT2()
        self.powerSpectrum, self.frequency, self.powerSpectrumErr = self._computePowerSpectrum()

    """
    Dunder methods
    """
    def __repr__(self):
        return f"{self.signal}"

    def __eq__(self, value):
        return self.signal.filePath == value

    """
    Private methods
    """
    def _computeCorrected(self, signal_data, weight_data):
        """ Computes corrected maps """
        # avoid NaN/Inf propagation: only multiply where both are finite
        finite_mask = np.isfinite(signal_data) & np.isfinite(weight_data)
        corrected = np.zeros_like(signal_data, dtype=float)
        np.multiply(signal_data, weight_data, out=corrected, where=finite_mask)
        return corrected

    def _computeFFT2(self):
        """ Compute FFT2 """
        # ensure no NaN/Inf remain (replace with 0) before FFT
        clean = np.nan_to_num(self.corrected, nan=0.0, posinf=0.0, neginf=0.0)
        return fft.fft2(clean)

    def _computePowerSpectrum(self):
        """ Compute the 1D power spectrum along with errors """
        F = np.fft.fftshift(self.fft2)
        P2D = np.abs(F)**2  # 2D power spectrum

        ny, nx = self.corrected.shape

        # Wavenumbers
        kx = np.fft.fftshift(np.fft.fftfreq(nx))
        ky = np.fft.fftshift(np.fft.fftfreq(ny))
        KX, KY = np.meshgrid(kx, ky)

        # Radial wavenumber
        K = np.sqrt(KX**2 + KY**2).flatten()
        P = P2D.flatten()

        # Hist
        nBins=int(np.sqrt(len(K)))
        hist_n, bin_edges = np.histogram(K, bins=nBins)
        hist_p, _ = np.histogram(K, bins=nBins, weights=P)
        hist_p2, _ = np.histogram(K, bins=nBins, weights=P**2)
        mean_P = hist_p/hist_n
        mean_Psquared = hist_p2/hist_n
        powerSpectrumErr = np.sqrt(mean_Psquared - mean_P**2) / np.sqrt(hist_n)
        # compute midpoints of each radial bin
        frequency = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        return mean_P, frequency, powerSpectrumErr

    """ Fitting models """
    def _powerSpectrumFit(self, f, A, alpha, f_t):
        return A**2 * (1+(f_t/f)**alpha)


    """
    Public methods
    """
    def plotFFT2(self):
        """ Plots the real part of the fft2 """
        im = plt.imshow(self.fft2.real, vmin=-0.5, vmax=0.5, cmap='jet', origin='lower')
        plt.colorbar(im, label="Jy/beam")
        plt.tight_layout()
        plt.show()
        return

    def plotPowerSpectrum(self):
        """ Plot the 1D power spectrum """
        # Plot the binned mean power spectrum (mean_P) vs radial wavenumber midpoints
        # bin edges are stored in `self.frequency`, mean values in `self.powerSpectrum`
        frequency = self.frequency[1:99]
        powerSpectrum = self.powerSpectrum[1:99]
        powerSpectrumErr = self.powerSpectrumErr[1:99]
        popt, pcov = curve_fit(self._powerSpectrumFit, frequency, powerSpectrum, sigma=powerSpectrumErr, p0=[1e2, 2, 1], bounds=([0,0,0], [1e10, 50, 10]))
        chi2 = chi2Reduced(powerSpectrum, self._powerSpectrumFit(frequency, *popt), powerSpectrumErr, len(popt))

        # Display data and model
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        ax1.plot(frequency, self._powerSpectrumFit(frequency, *popt), color='red', label=rf"""$A$={popt[0]:.2e}$\pm${pcov[0][0]:.2e},
        $\alpha$={popt[1]:.2e}$\pm${pcov[1][1]:.2e},
        $f_t$={popt[2]:.2e}$\pm${pcov[2][2]:.2e},
        $\chi^2_{{red}}$={chi2:.2f}""")
        ax1.errorbar(frequency, powerSpectrum, yerr=powerSpectrumErr, fmt='-o', capsize=3, label='binned mean P(k)')
        ax1.set_yscale('log')
        ax1.set_ylabel("P(k) [$(Jy/beam)^2$]")
        ax1.grid()
        ax1.legend()

        # Display residuals
        residuals = self._powerSpectrumFit(frequency, *popt) - powerSpectrum
        ax2.axhline(0, color='red', linestyle='--')
        ax2.scatter(frequency, np.abs(residuals), marker='x')
        ax2.set_yscale('log')
        ax2.set_xlabel("k [Hz]")
        ax2.set_ylabel("Residuals")
        ax2.grid()
        fig.tight_layout()
        plt.show()
        return