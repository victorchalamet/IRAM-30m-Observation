import numpy as np
from scipy import fft
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from .helper import chi2Reduced

class Subscan:
    """ A subscan is composed of two fits objects : signal and weight """
    def __init__(self, signal, weight):
        self.signal = signal
        self.weight = weight
        self.corrected = self._computeCorrected(signal.data, weight.data)
        self.fft2 = self._computeFFT2()
        self.frequency, self.powerSpectrum = self._computePowerSpectrum()

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
        """ Compute the 1D power spectrum """
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

        # Radial binning
        k_bins = np.linspace(0, np.max(K), num=min(nx, ny)//2)
        k_vals = 0.5*(k_bins[1:] + k_bins[:-1])  # midpoint of bins

        # Bin average
        P1D = np.zeros(len(k_vals))
        digitized = np.digitize(K, k_bins)
        for i in range(1, len(k_bins)):
            mask = digitized == i
            if np.any(mask):
                P1D[i-1] = P[mask].mean()
        return k_vals, P1D
    
    # def _ComputeErrors(self, frequency, powerSpectrum):
    #     hist_n, _ = np.histogram(frequency, len(frequency))
    #     hist_p, _ = np.histogram(frequency, len(frequency), weights=powerSpectrum)
    #     hist_p2, _ = np.histogram(frequency, len(frequency), weights=powerSpectrum**2)
    #     mean_Psquared = hist_p2/hist_n
    #     mean_P = hist_p/hist_n
    #     return np.sqrt(mean_Psquared - mean_P**2)

    def _powerSpectrumFit(self, f, alpha, f_t):
        return 1e4 + (1+f_t/f)**alpha

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
        # Cut the high frequencies because of PIIC gaussian smoothing and filter out the null value
        frequency = self.frequency[:49][self.powerSpectrum[:49]>0]
        powerSpectrum = self.powerSpectrum[:49][self.powerSpectrum[:49]>0]
        plt.plot(frequency, powerSpectrum)
        # errors = self._ComputeErrors(frequency, powerSpectrum)
        # popt, pcov = curve_fit(self._powerSpectrumFit, frequency, powerSpectrum, p0=[1, 1])
        # chi2 = chi2Reduced(powerSpectrum, self._powerSpectrumFit(frequency, *popt), errors, 2)
        # print(chi2)
        # print(popt)
        # print(pcov)
        # plt.plot(frequency, self._powerSpectrumFit(frequency, *popt), label=rf"$f_t$={round(popt[1],2)}=$\pm${round(pcov[1][1],2)}$\alpha$={round(popt[0],2)}$\pm${round(pcov[0][0],2)}")
        plt.yscale('log')
        plt.xlabel("k [Hz]")
        plt.ylabel("P(k) [$(Jy/beam)^2$]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        return