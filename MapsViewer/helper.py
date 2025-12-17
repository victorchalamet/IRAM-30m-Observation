import os

import numpy as np

from .fits import Fits

def findScans(scan_list, filePath):
    """
    # Function that finds the 4 consecutive scans
    Give the first of the four as `filePath`
    """
    fileName = os.path.basename(filePath)
    scan_number = fileName[30:33]
    if scan_number.endswith("_"):
        scan_number = scan_number[:-1]
    scan_number = int(scan_number)
    for i in range(1, 4):
        if int(scan_number)>100:
            newFileName = os.path.dirname(filePath) + "/" + fileName[:30] + f'{scan_number+i}' + fileName[33:]
        else:
            newFileName = os.path.dirname(filePath) + "/" + fileName[:30] + f'{scan_number+i}' + fileName[32:]
        
        print("Looking for :", os.path.basename(newFileName))
        
        if os.path.exists(newFileName):
            print("Found :", os.path.basename(newFileName))
            scan_list.append(Fits(newFileName))
        else:
            break
    if len(scan_list)<4: print(f"Couldn't find 4 consecutive scans. Only found {len(scan_list)}")
    elif len(scan_list)>4: print(f"Found too much scans : {len(scan_list)}")
    return scan_list

def chi2(y_measure, y_predict, errors):
    """Calculate the chi squared value given a measurement with errors and prediction"""
    return np.sum((y_measure-y_predict)**2 / errors**2)

def chi2Reduced(y_measure, y_predict, errors, dof):
    """Calculate the reduced chi squared value given a measurement with errors and prediction,
    and knowing the number of parameters in the model."""
    return chi2(y_measure, y_predict, errors)/(y_measure.size - dof)

def findIndex(array, value, option='top'):
    """ Find the index of the first element above or below a given value in a sorted array"""
    if option == 'top':
        for i in range(len(array)):
            if array[i]>=value:
                return i
    elif option == 'bottom':
        for i in range(len(array)):
            if array[i]<=value:
                return i