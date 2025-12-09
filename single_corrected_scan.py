import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from MapsViewer.fits import Fits
from MapsViewer.subscan import Subscan
from MapsViewer.env import DATA_PATH

plt.style.use('/home/vchalamet/Documents/Code/IRAM-30m-Observation/presentation.mplstyle')

def main(args):
    subscans = []
    signalFiles = []
    weightFiles = []
    dataPath = DATA_PATH + "PSZ2G107/"
    for file in sorted(os.listdir(dataPath)):
        if file.endswith("n.fits"):
            signalFiles.append(Fits(dataPath + file, False))
        else:
            weightFiles.append(Fits(dataPath + file, False))

    for i in range(len(signalFiles)): subscans.append(Subscan(signalFiles[i], weightFiles[i]))
    for subscan in subscans:
        subscan.plotPowerSpectrum()
 
    return
    im = plt.imshow(final_corrected, cmap='jet', origin='lower')
    plt.title('Final scan')
    plt.colorbar(im, label="Jy/beam")
    plt.tight_layout()
    plt.show()



if __name__=='__main__':
    main(sys.argv)
