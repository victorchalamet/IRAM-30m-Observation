import sys

import matplotlib.pyplot as plt

from MapsViewer.helper import find_scans
from MapsViewer.fits import Fits

plt.style.use('/home/vchalamet/Documents/Code/IRAM-30m-Observation/presentation.mplstyle')


def main(args):
    print("Generating 4 scans figure...")
    filePath = args[1]

    # Find the 4 rotations
    fitsList = [Fits(filePath)]
    find_scans(fitsList, filePath)
    print(f"Found scans : {fitsList}")

    fig = plt.figure()
    for i, fit in enumerate(fitsList):
        # Use WCS projection for each subplot
        ax = fig.add_subplot(2, 2, i+1, projection=fit.wcs)
        if(fit.filePath.endswith("n.fits")):
            im = ax.imshow(fit.data, origin='lower', vmin=-10, vmax=10, cmap='jet')
        else:
            im = ax.imshow(fit.data, origin='lower', cmap='jet')
        ax.set_title(f'Angle : {i*45}')
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
    cbar_ax = fig.add_axes([0.50, 0.50-(0.95/2), 0.02, 0.95])
    fig.colorbar(im, cax=cbar_ax, label="Jy/beam")
    fig.tight_layout()
    fig.savefig(fitsList[0].pngPath[:-4]+"_full.png", bbox_inches="tight", pad_inches=0.1)
    print(f"Saved figure to {fitsList[0].pngPath}")
    return

if __name__=='__main__':
    main(sys.argv)
