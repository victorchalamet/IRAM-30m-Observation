import os
import json

from astropy.io import fits
from astropy.wcs import WCS

from .env import TMP_PATH

class Fits:
    def __init__(self, filePath, save_header=True):
        self.filePath = filePath
        self.pngPath = TMP_PATH + os.path.basename(filePath).replace('.fits', '.png')
        self.jsonPath = TMP_PATH + os.path.basename(filePath).replace('.fits', '.json')
        self.data, self.header = fits.getdata(filePath, header=True)
        if save_header: self.save_header()
        self.wcs = WCS(self.header)
    
    def __repr__(self):
        return os.path.basename(self.filePath)

    def __eq__(self, value):
        return self.filePath == value

    def save_header(self):
        """Function that saves header to a temporary json file"""
        header = dict(self.header)
        header.pop('COMMENT')
        with open(f"{self.jsonPath}", "w") as f:
            json.dump(header, f)
        print(f"Saved header to {self.jsonPath}")
        return
