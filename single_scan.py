import sys

import matplotlib.pyplot as plt

from MapsViewer.fits import Fits

plt.style.use('/home/vchalamet/Documents/Code/IRAM-30m-Observation/presentation.mplstyle')

def main(args):
    print("Generating single scan figure...")
    filePath = args[1]

    file = Fits(filePath)
    # Create figure with WCS projection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=file.wcs)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    if(filePath.endswith("n.fits")):
        im = ax.imshow(file.data, origin='lower', vmin=-10, vmax=10, cmap='jet')
    else:
        im = ax.imshow(file.data, origin='lower', cmap='jet')
    plt.colorbar(im, ax=ax, label="Jy/beam")
    plt.tight_layout()
    plt.savefig(file.pngPath, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved figure to {file.pngPath}")
    return

if __name__=='__main__':
    main(sys.argv)
