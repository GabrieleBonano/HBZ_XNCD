"""
Python script to access a SPEC file and retrieve the columns I am interest in: Energy, Detector, Monitor, RingCurr.
Additionally, already compute the normalisation Detector/Monitor.
Save these information in a .txt file in the same folder with (,) separation.

NOTES:
Before running the code remember to check:
 1) spec: the file name or path
 2) lista: list  of scans you want to convert into .txt

The code will not overwrite any scan for which a .txt file already exists

Created by Gabriele Bonano on 01/13/2025 
Last edited by Gabriele Bonano on 01/13/2025
"""

import os
from liquidspec_XAS_Claude import *   

spec = SpecFile("UE52Dec2024_EckertHartlieb")

lista = arange(700, 801, 1)

for scan_number in lista:
    file_name = f'Scan{scan_number}.txt'

    if os.path.exists(file_name):
        print(f'File {file_name} already exists. Skipping scan {scan_number}.')
        continue

    Energy_name = 'Energy'
    Detector_name = 'Detector'
    Monitor_name = 'Monitor'
    DetectorNorm_name = 'DetectorNorm'
    RingCurr_name = 'RingCurr'
    
    Energy, Detector = spec.scans(scans=str(scan_number), value="Detector", minX=None, maxX=None, stepX=None, norm=1, show=False, key='Energy', save_individual=False, filtering=False)
    Energy, Monitor = spec.scans(scans=str(scan_number), value="Monitor", minX=None, maxX=None, stepX=None, norm=1, show=False, key='Energy', save_individual=False, filtering=False)
    Energy, RingCurr = spec.scans(scans=str(scan_number), value="Ringcurr", minX=None, maxX=None, stepX=None, norm=1, show=False, key='Energy', save_individual=False, filtering=False)

    DetectorNorm = Detector / Monitor
    
    combined_array = np.column_stack((Energy, Detector, Monitor, DetectorNorm, RingCurr))
    
    headers = f'{Energy_name},{Detector_name},{Monitor_name},{DetectorNorm_name},{RingCurr_name}'
    
    np.savetxt(file_name, combined_array, fmt='%.6e', delimiter=',', header=headers, comments='')

    print(f'Saved {file_name}')