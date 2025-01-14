"""
liquidspec_XAS_Claude - X-ray Absorption Spectroscopy Analysis Library - Heavily modified version of liquidspec.py

NOTE (O.G.): 
liquidspec is a python library which aims to facilitate the analysis of experimental data of the field resonant soft X-ray spectroscopy. 
The library was mainly designed to work in jupyter notebooks. The installation of the Python 3 version of anaconda (https://www.anaconda.com) 
is higly recommended. 
The latest version of the library can be downloaded from https://it-ed-git.basisit.de/lte/liquidspec_XAS/-/raw/master/liquidspec_XAS.py?inline=false.

A Python 3 library for analyzing experimental data in resonant soft X-ray spectroscopy.
This is a shortened version without RIXS extensions.

Main Features:
- Facilitates analysis of X-ray absorption spectroscopy data
- Designed for use in Jupyter notebooks
- Built on pandas for efficient data handling
- Provides tools for data processing, visualization, and analysis

Dependencies:
- pandas
- matplotlib
- numpy
- scipy
- PIL

Last modified by Gabriele Bonano - 01/14/25
"""

from pandas import *
from matplotlib.pyplot import *
from numpy import *
rcParams['figure.figsize'] = (9.6, 5.4)
import os
from fnmatch import filter
from scipy import optimize
from scipy.signal import medfilt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import correlate
from scipy.special import wofz, erfc, erf
from PIL import Image
from copy import deepcopy
import pylab as pyLLLL
from matplotlib.legend_handler import HandlerBase

####################################################################################################    
#                           General Utility Functions                                              #
####################################################################################################

def scan_numbers(string):
    """
    Transform a string into a list of scan numbers.
    
    Format rules:
    - Individual numbers separated by spaces
    - Ranges defined as 'first-last' (inclusive)
    - Skip numbers in range with '#' (e.g., "1-5#3" excludes 3)
    - Step values with '%' (e.g., "2-10%2" for every 2nd number)
    
    Examples:
    >>> scan_numbers("1-5#3 22-28%2 42")
    [1, 2, 4, 5, 22, 24, 26, 28, 42]
    
    Args:
        string (str): Formatted string of scan numbers
        
    Returns:
        list: List of integer scan numbers
    """
    out = []
    for entry in string.split():
        if '-' not in entry:
            out.append(int(entry))
        else:
            if '#' not in entry and '%' not in entry:
                first, last = entry.split('-')
                out += range(int(first), int(last) + 1)
            else:
                if '#' in entry and '%' in entry:
                    print(f"WARNING: Entry {entry} cannot be evaluated because it contains '%' and '#'.")
                else:
                    if '#' in entry:
                        parts = entry.split('#')
                        first, last = parts[0].split('-')
                        temp = list(range(int(first), int(last) + 1))
                        for value in parts[1:]:
                            temp.remove(int(value))
                        out += temp    
                    else:  # '%' is in entry
                        limits, step = entry.split('%')
                        first, last = limits.split('-')
                        out += range(int(first), int(last) + 1, int(step))                                    
    return out

def nplot(x, y, fmt="", **kwargs):
    """
    Plot data normalized by area with smallest values set to zero.
    
    Args:
        x (array-like): X-axis values
        y (array-like): Y-axis values to be normalized
        fmt (str): Format string for plot
        **kwargs: Additional arguments passed to plot()
    """
    y = y - percentile(y, 1)  # Subtract offset (lowest 1% of datapoints)
    y = y / trapz(y)  # Normalize by area
    plot(x, y, fmt, **kwargs)

def get_shift(a, b):
    """
    Calculate shift between two arrays using cross-correlation.
    
    Args:
        a (array): First array (1D or 2D)
        b (array): Second array of same shape as a
        
    Returns:
        int: Shift value
        
    Raises:
        ValueError: If input arrays have different dimensions
    """
    shape_a = shape(a)
    if not shape_a == shape(b):
        raise ValueError("Input arrays must have equal dimensions!")
        
    if len(shape_a) == 1:
        return argmax(correlate(a, b)) - (len(a) - 1)
    else:
        if not shape_a[0] > shape_a[1]:
            print(f"WARNING: Shift is determined along the shorter axis ({shape_a[0]} entries)!")
        shifts = zeros(shape_a[1])
        for i in range(shape_a[1]):
            shifts[i] = argmax(correlate(a[:, i], b[:, i]))
        return int(round(mean(shifts) - (shape_a[0] - 1)))

def binData(x, y, binning_step=2):
    """
    Bin the data into groups based on the binning step.
    
    Args:
        x (array-like): The x values of the data.
        y (array-like): The y values corresponding to x values.
        binning_step (int): The number of points to include in each bin.
        
    Returns:
        binned_x (array): The center of each bin.
        binned_y (array): The average y value for each bin.
    """
    # Convert inputs to numpy arrays
    x = array(x)
    y = array(y)

    # Check if arrays are of equal length
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length.")

    # Number of bins
    n_bins = len(x) // binning_step

    # Initialize lists to store binned results
    binned_x = []
    binned_y = []

    # Loop over the bins
    for i in range(n_bins):
        # Define the start and end indices of the bin
        start_idx = i * binning_step
        end_idx = (i + 1) * binning_step

        # Slice the x and y values for the current bin
        x_bin = x[start_idx:end_idx]
        y_bin = y[start_idx:end_idx]

        # Compute the average x and y values for this bin
        binned_x.append(mean(x_bin))
        binned_y.append(mean(y_bin))

    # Convert the binned results to numpy arrays
    binned_x = array(binned_x)
    binned_y = array(binned_y)
    return binned_x, binned_y

def spectra_diff(E1, A1, E2, A2, stepX, stretch=False, shows=True):
    """
    General function to obtain the difference A1-A2 between two spectra, may it be absorbance or raw signal.

    Args:
        E1 (array-like): The energy domain of the 1st spectrum.
        A1 (array-like): The 1st spectrum signal.
        E2 (array-like): The energy domain of the 2nd spectrum.
        A2 (array-like): The 2nd spectrum signal.
        stepX (float): Step size in case common domain has to be retrieved.
        stretch (bool): If (rough) stretching is required.
        
    Returns:
        E_xx (array): Energy range.
        A (array): Difference spectrum A1-A2.
    """
    if not np.array_equal(E1, E2):
        # Find the higher minimum value
        min_value = max(min(E1), min(E2))
        # Find the lower maximum value
        max_value = min(max(E1), max(E2))
        # Create a third array with step size stepX between min_value and max_value
        E_xx = arange(min_value, max_value, stepX)
        # Interpolate the two absorbances
        f_1 = interp1d(E1, A1)
        A_1 = f_1(E_xx)
        f_2 = interp1d(E2, A2)
        A_2 = f_2(E_xx)
    else:
        A_1 = A1
        A_2 = A2
        E_xx = E1

    if stretch:
        # Stretch and subtract
        s1 = A_1[0] - A_1[-1]
        s2 = A_2[0] - A_2[-1]
        A = A_1 - A_2 * s1 / s2
    else:
        # No stretching, just subtract
        A = A_1 - A_2
    if shows:
        # Plot the original spectra on the primary Y-axis
        fig, ax1 = subplots()
        ax1.plot(E_xx, A_1, label='Spectrum 1', color='b')
        ax1.plot(E_xx, A_2, label='Spectrum 2', color='g')
        ax1.set_xlabel('Energy')
        ax1.set_ylabel('Signal (Spectra)', color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        
        # Create a secondary Y-axis for the difference spectrum
        ax2 = ax1.twinx()
        ax2.plot(E_xx, A, label='Difference Spectrum', color='r', linestyle='--')
        ax2.set_ylabel('Difference Signal', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
        # Show the plot
        title('Spectra and Difference Spectrum')
        show()


    return E_xx, A


def solvent_subtraction(E_sample, A_sample, E_sol, A_sol, stepX, shows=True):
    """
    Function to perform solvent subtraction

    Args:
        E_sample (array-like): The energy domain of the sample data.
        A_sample (array-like): The sample absorbance.
        E_sol (array-like): The energy domain of the solvent data.
        A_sol (array-like): The solvent absorbance.
        stepX (float): step size in case common domain has to be retrieved.
        
    Returns:
        E_xx (array): Energy range.
        A (array): Solvent subtracted absorbance.
    """
    if not np.array_equal(E_sample, E_sol):
        # Find the higher minimum value
        min_value = max(min(E_sample), min(E_sol))
        # Find the lower maximum value
        max_value = min(max(E_sample), max(E_sol))    
        # Create a third array with step size 0.1 between min_value and max_value
        E_xx = arange(min_value, max_value, step)
        # Interpolate the two absorbances
        f_sample = interp1d(E_sample, A_sample)
        A_SAMPLE = f_sample(E_xx)
        f_sol = interp1d(E_sol, A_sol)
        A_SOL = f_sol(E_xx)
    else:
        A_SAMPLE=A_sample
        A_SOL=A_sol
        E_xx=E_sample
    # Stretch and subtract
    s1=A_SAMPLE[0]-A_SAMPLE[-1]
    s2=A_SOL[0]-A_SOL[-1]
    A = A_SAMPLE - A_SOL*s1/s2
    if shows:
        # Plot the original spectra on the primary Y-axis
        fig, ax1 = subplots()
        ax1.plot(E_xx, A_sample, label='Sample Spectrum', color='r')
        ax1.plot(E_xx, A_sol, label='Solvent Spectrum', color='b')
        ax1.set_xlabel('Energy')
        ax1.set_ylabel('Absorbance (arb. units)', color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        
        # Create a secondary Y-axis for the difference spectrum
        ax2 = ax1.twinx()
        ax2.plot(E_xx, A, label='Difference Spectrum', color='g', linestyle='--')
        ax2.set_ylabel('Difference Signal', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
        # Show the plot
        title('Spectra and Solvent subtracted Spectrum')
        show()
    return E_xx, A

def mov_avg(X, Y, window_size):
    """
    Perform moving average smoothing on the Y values and plot the results.
    
    Args:
        X (array-like): The x values of the data.
        Y (array-like): The y values corresponding to X values.
        window_size (int): The size of the moving window.
        
    Returns:
        smoothed_Y (array): The smoothed y values after applying the moving average.
    """
    X = array(X)
    Y = array(Y)
    smoothed_Y = []
    for i in range(len(Y)):
        # Define the range of the window 
        start_idx = max([0, i - window_size // 2])
        end_idx = min([len(Y), i + window_size // 2 + 1])
        window_avg = mean(Y[start_idx:end_idx])
        smoothed_Y.append(window_avg)
    smoothed_Y = array(smoothed_Y)
    # Plot the original data and the smoothed curve
    figure(figsize=(10, 6))
    plot(X, Y, label='Original Data', color='k', linewidth = 1)  
    plot(X, smoothed_Y, color='r', linestyle='-', linewidth=4, alpha=0.75)
    title('Moving Average of Data')
    xlabel('X')
    ylabel('Y')
    legend()
    show()

    return smoothed_Y  

    
####################################################################################################    
#                           SPEC File Processing Class                                             #
####################################################################################################

def _spec_to_pandas(filename):
    """
    Convert SPEC file to pandas DataFrame.
    
    Internal helper function for SpecFile class.
    
    Args:
        filename (str): Path to SPEC file
        
    Returns:
        tuple: (DataFrame, last_complete_scan_number)
    """
    rows = []
    last_complete = -1
    
    with open(filename) as spec_file:
        scan_number = 0
        label = []
        
        for line in spec_file:
            if line == "\n":
                continue
                
            if line[:2] == "#S":
                scan_number = int(line.split()[1])
                continue
                
            if line[:2] == "#L":
                label = (line.replace("SB current", "SingleBunch_Current")
                            .replace("SB Current", "SingleBunch_Current")
                            .replace("ShortBunch curr", "ShortBunch_Current")
                            .replace("Exit Slit", "ExitSlit")
                            .split())
                label[0] = "ScanNumber"
                label[1] = "Keys"
                continue
                
            if line[:2] == "#C":
                last_complete = scan_number
                continue
                
            if line[:1] == "#":
                continue
                
            row = [scan_number] + line.split()
            rows.append(dict(zip(label, row)))
            
    return DataFrame(rows), last_complete

class SpecFile:
    """
    Class for analyzing SPEC file data with efficient data loading and processing.
    
    Main features:
    - Load and parse SPEC files
    - Access column data easily
    - Process multiple scans
    - Handle live updates
    
    Args:
        filename (str): Path and name of the SPEC file
        update (str): Optional command for live analysis updates (default: "ls")
    """
    
    def __init__(self, filename, update="ls"):
        self._filename = filename
        self._update = update
        
        try:
            self.df, self._last_complete = _spec_to_pandas(self._filename)
        except:
            os.system(self._update)
            self.df, self._last_complete = _spec_to_pandas(self._filename)

    def value_list(self):
        """
        Get available column names in the SPEC file.
        
        Returns:
            array: Column names that can be used as value or norm parameters
        """
        return self.df.columns.values

    def _scan_numbers_and_potentially_update(self, scans, post_update_func=None):
        """
        Update dataframe if required scans are not present.
        
        Internal helper function.
        
        Args:
            scans (str): Scan numbers string
            post_update_func (callable, optional): Function to call after update
            
        Returns:
            list: Processed scan numbers
        """
        scan_nums = scan_numbers(scans)
        if len(scan_nums) > 0:
            if max(scan_nums) > self._last_complete:
                os.system(self._update)
                self.df, self._last_complete = _spec_to_pandas(self._filename)
        return scan_nums

    def filter_outliers(self, x, y, window_size=5, filtering=10):
        """
        Remove spikes from signal data.
        
        Args:
            x (array-like): Domain values
            y (array-like): Signal values
            window_size (int): Window for median filter
            filtering (float): Spike detection threshold
            
        Returns:
            tuple: (cleaned_x, cleaned_y)
        """
        x, y = array(x), array(y)
        baseline = medfilt(y, kernel_size=window_size)
        deviation = abs(y - baseline)
        spike_mask = deviation > filtering * std(deviation)
        
        return x[~spike_mask], y[~spike_mask]

    def get(self, scans, value, print_unit=False):
        """
        Get average and standard deviation of a value over multiple scans.
        
        Args:
            scans (str): Scan numbers string
            value (str): Column name in spec file
            print_unit (str/bool): Unit string for printing results
            
        Returns:
            tuple: (average, standard_deviation)
        """
        if not api.types.is_numeric_dtype(self.df[value]):
            self.df[value] = to_numeric(self.df[value], errors='coerce')
            
        scan_nums = self._scan_numbers_and_potentially_update(scans)
        mask = self.df['ScanNumber'].isin(scan_nums)
        
        ret_mean = self.df.loc[mask][value].mean()
        ret_std = self.df.loc[mask][value].std()
        
        if print_unit:
            if ret_std > 0:
                digits = max(int(-log10(ret_std)) + 1, 0)
                print(f"{value} = ({ret_mean:.{digits}f} ± {ret_std:.{digits}f}){print_unit}")
            else:
                print(f"{value} = {ret_mean:.5g}{print_unit}")
                
        return ret_mean, ret_std

    def scans(self, scans, value, minX=None, maxX=None, stepX=None, norm=1, 
             show='absolute', key='Energy', save_individual=False, filtering=False):
        """
        Load and process multiple scans with optional normalization and visualization.
        
        Args:
            scans (str): Scan numbers string
            value (str): Column name for values
            minX (float, optional): Lower limit for keys
            maxX (float, optional): Upper limit for keys
            stepX (float, optional): Step size for keys
            norm (str/int): Normalization column or 1 for no normalization
            show (str): Plot mode ('absolute', 'relative', or False)
            key (str): Column name for keys (default: 'Energy')
            save_individual (bool): Save individual scan data
            filtering (float/bool): Spike filtering threshold
            
        Returns:
            tuple: (x_values, intensities or individual_intensities)
        """
        scan_nums = self._scan_numbers_and_potentially_update(scans)
        
        # Process first scan
        keys = self.df.loc[self.df['ScanNumber'] == scan_nums[0], key].astype(float).values
        values = self.df.loc[self.df['ScanNumber'] == scan_nums[0], value].astype(float).values
        
        if norm != 1:
            values = values / self.df.loc[self.df['ScanNumber'] == scan_nums[0], norm].astype(float).values
            
        if filtering:
            keys, values = self.filter_outliers(keys, values, filtering=filtering)
            
        if show:
            if show == "absolute":
                plot(keys, values, ".", markersize="2", picker=5, label=str(scan_nums[0]))
            else:
                nplot(keys, values, ".", markersize="2", picker=5, label=str(scan_nums[0]))
                
        # Interpolation setup
        if minX and maxX and stepX:
            x = arange(minX, maxX, stepX)
            f = interp1d(keys, values)
            I = f(x)
        else:
            x = keys
            I = values
            
        individual_I_arrays = {scan_nums[0]: I} if save_individual else None
        c = 1.0
        
        # Process remaining scans
        for n in scan_nums[1:]:
            try:
                keys = self.df.loc[self.df['ScanNumber'] == n, key].astype(float).values
                values = self.df.loc[self.df['ScanNumber'] == n, value].astype(float).values
                
                if norm != 1:
                    values = values / self.df.loc[self.df['ScanNumber'] == n, norm].astype(float).values
                    
                if filtering:
                    keys, values = self.filter_outliers(keys, values, filtering=filtering)
                    
                if show:
                    if show == "absolute":
                        plot(keys, values, ".", markersize="2", picker=5, label=str(n))
                    else:
                        nplot(keys, values, ".", markersize="2", picker=5, label=str(n))
                        
                f = interp1d(keys, values)
                I_scan = f(x)
                
                if save_individual:
                    individual_I_arrays[n] = I_scan
                else:
                    I += I_scan
                    c += 1.0
                    
            except Exception as e:
                print(f"ERROR while reading scan number {n}: {e}")
                
        if show:
            legend()
            
        return (x, individual_I_arrays) if save_individual else (x, I/c)


    
    def absorbance_easy(self, sample_scans, beamline_scans, minX=None, maxX=None, stepX=None, norm=1, show='absolute', 
               bgCorrectPoints=0, splineFitBeamline=-1, value="Detector", filter_bl=False, filter_sample=False):
        """
        Returns the absorbance from a static T-NEXAFS measurement. Simple version without any thickness estimation or 
        any kind of normalisation.
    
        Absorbance / OD is A(E) = -log_10(I(E)/I0(E)) [Fondell et al., Struct. Dyn. 4 (2017), 054902]
        
        Parameters:
            sample_scans (str): Sample scan (through the jet) numbers as described for scanNumbers()
            beamline_scans (str): Beamline scan (without jet) numbers as described for scanNumbers()    
            minX (float): Low limit of the keys (e.g. energy, scanned motor)
            maxX (float): High limit of the keys
            stepX (float): Stepsize of the keys
            norm (str): Column name of the monitor in the spec file (e.g. "Monitor")
            show (str): If True, the raw data is plotted in separate figures 
                       (if show="absolute" the data is not scaled to facilitate the comparison).
                       The figures are hidden if show=False.
            bgCorrectPoints (int): Number of points used to fit a linear background, which is then subtracted
            splineFitBeamline (float): If > 0: beamline scan(s) will be smoothed by spline fit which is 
                                      rougher the larger s is (e.g. 1e-4 is good for N K-edge scans)
            value (str): Column name of values in the spec file (default "Detector")
            filter_bl: filtering for spec.scans() for beamline spectra
            filter_sample: filtering for spec.scans() for sample spectra
    
        Returns:
            tuple: (energies, absorbance/OD) or (energies, extinction coefficient M⁻¹µm⁻¹)
        """
        # PROCESS BEAMLINE SCANS --> E, I0 (, Ifit) 
        # Open figure if plots are demanded 
        if show:
            figure()
            title("Beamline scans")
            xlabel("Energy (eV)")
            ylabel("Initial intensity " + ("(arb. u.)" if show == "absolute" else "(scaled)"))
        
        # Load all beamline scans, average them and plot them if demanded 
        E, I0 = self.scans(beamline_scans, value, minX, maxX, stepX, norm=norm, show=show, save_individual=False, filtering=filter_bl)
        
        # Fit beamline (Ifit) - optional
        if splineFitBeamline > 0: 
            fit = UnivariateSpline(E, I0, s=splineFitBeamline)
            Ifit = fit(E)
            if show:
                if show == "absolute":
                    plot(E, Ifit, "r", label="spline fit")
                    title('Absolute')
                else:
                    nplot(E, Ifit, "r", label="spline fit")
                    title('Relative')

        # Add averaged beamline to figure if show=True (scaled) or show='absolute' (raw data)
        if show:
            if show == "absolute":
                plot(E, I0, "k", label="raw average")
                title('Absolute')
            else:
                nplot(E, I0, "k", label="raw average")
                title('Relative')
            legend()

        # PROCESS SAMPLE SCANS INDIVIDUALLY --> E, [I1,I2...] 
        # Open figure if plots are demanded 
        if show:
            figure()
            title("Sample scans")
            xlabel("Energy (eV)")
            ylabel("Initial intensity " + ("(arb. u.)" if show == "absolute" else "(scaled)"))
        
        # Load all beamline scans, average them and plot them if demanded 
        Es, I = self.scans(sample_scans, value, minX, maxX, stepX, norm=norm, show=show, save_individual=True, filtering=filter_sample)
        # Convert I.values() to an array
        I_array = array([I_array for I_array in I.values()])
        # Calculate the mean intensity
        mean_I = mean(I_array, axis=0)
        # Plot the mean intensity
        if show:
            if show == "absolute":
                plot(Es, mean_I, label="raw average", color='k', linestyle='-', linewidth=2)
            else:
                nplot(Es, mean_I, label="raw average", color='k', linestyle='-', linewidth=2)
            legend()
        
        # COMPUTE THE ABSORBANCE FOR EACH SCAN
        # Calculate the absorbance for each of the scans as A = -log10(I/I0)
        A = -log10(I_array/I0)                     # absorbance from raw data
        baseline=empty_like(A)
        A_all=empty_like(A)
        # Background subtraction
        if bgCorrectPoints > 0:
            # Perform linear fitting for each scan in A
            for i in range(A.shape[0]):  # Loop over each scan 
                scan = A[i,:]  
                # Perform linear fit 
                fit_params = polyfit(Es[:bgCorrectPoints], scan[:bgCorrectPoints], 1)  
                baseline[i, :] = polyval(fit_params, Es)
                A_all[i,:]=scan-baseline[i,:]
                # Optionally, plot the individual fit for each scan
                figure()
                if show:
                    plot(Es, scan, label=f'Scan {i+1} - data', alpha=0.7)
                    plot(Es, polyval(fit_params, Es), label=f'Scan {i+1} - fit', linestyle='--')
                    plot(Es[:bgCorrectPoints],scan[:bgCorrectPoints],"r", label="baseline of background substraction", linewidth=3)
                    legend()
        else:
            baseline = zeros(len(E))
            A_all=A

        # Calculate the mean absorbance of baseline subtracted data
        mean_A_all = mean(A_all, axis=0)

        # Calculate the mean absorbance and std for thickness estimation (+ mean baseline because why not?)
        mean_A = mean(A, axis=0)
        std_A = std(A, axis=0)
        mean_baseline = mean(baseline, axis=0)

        # Open figure if plots are demanded
        if show:
            figure()
            title("Absorbance (lin. bkg subtracted)")
            xlabel("Energy (eV)")
            ylabel("A " + ("(arb. u.)" if show == "absolute" else "(scaled)"))
            
        # Plot 
        if show:
            if show == "absolute":
                for scan_number, scan in zip(I.keys(), A_all): plot(Es, scan, label=f'{scan_number}', alpha=0.5)
                plot(Es, mean_A_all, label="raw average", color='k', linestyle='-', linewidth=2)
                plot(Es[:bgCorrectPoints],mean_A_all[:bgCorrectPoints],"r", label="baseline of background substraction", linewidth=4)
            else:
                for scan_number, scan in zip(I.keys(), A_all): nplot(Es, scan, label=f'{scan_number}', alpha=0.5)
                nplot(Es, mean_A_all, label="raw average", color='k', linestyle='-', linewidth=2)
            legend()     

        return Es, mean_A 

    def absorbance_full(self, sample_scans, beamline_scans, minX=None, maxX=None, stepX=None, norm=1, show='absolute', 
               bgCorrectPoints=0, splineFitBeamline=-1, value="Detector", energyT=False, absorbance_solvent_1um=False, concentration=False, filter_bl=False, filter_sample=False):
        """
        Returns the absorbance from a static T-NEXAFS measurement. Full version with thickness estimation.
    
        Absorbance / OD is A(E) = -log_10(I(E)/I0(E)) [Fondell et al., Struct. Dyn. 4 (2017), 054902]
        
        Parameters:
            sample_scans (str): Sample scan (through the jet) numbers as described for scanNumbers()
            beamline_scans (str): Beamline scan (without jet) numbers as described for scanNumbers()    
            minX (float): Low limit of the keys (e.g. energy, scanned motor)
            maxX (float): High limit of the keys
            stepX (float): Stepsize of the keys
            norm (str): Column name of the monitor in the spec file (e.g. "Monitor")
            show (str): If True, the raw data is plotted in separate figures 
                       (if show="absolute" the data is not scaled to facilitate the comparison).
                       The figures are hidden if show=False.
            bgCorrectPoints (int): Number of points used to fit a linear background, which is then subtracted
            splineFitBeamline (float): If > 0: beamline scan(s) will be smoothed by spline fit which is 
                                      rougher the larger s is (e.g. 1e-4 is good for N K-edge scans)
            value (str): Column name of values in the spec file (default "Detector")
            energyT (float): Energy at which the thickness should be estimated (pre edge region) -> it is a bit rough but ok for analysis on the fly
            absorbance_solvent_1um (float): Only needed if the extinction coefficient should be returned: Get the transmission (T) of 1 um solvent 
                                            e.g. from http://henke.lbl.gov/optical_constants/filter2.html. (absorbance_solvent_1um = -log10(T) )
            concentration (float): Only needed if the extinction coefficient should be returned: Concentration of the sample (recommended unit: M)
            filter_bl: filtering for spec.scans() for beamline spectra
            filter_sample: filtering for spec.scans() for sample spectra
    
        Returns:
            tuple: (energies, absorbance/OD) or (energies, extinction coefficient M⁻¹µm⁻¹)
        """
        sampleNums = self._scan_numbers_and_potentially_update(sample_scans)
        total_sample_scan_number = len(sampleNums)
        # PROCESS BEAMLINE SCANS --> E, I0 (, Ifit) 
        # Open figure if plots are demanded 
        if show:
            figure()
            title("Beamline scans")
            xlabel("Energy (eV)")
            ylabel("Initial intensity " + ("(arb. u.)" if show == "absolute" else "(scaled)"))
        
        # Load all beamline scans, average them and plot them if demanded 
        E, I0 = self.scans(beamline_scans, value, minX, maxX, stepX, norm=norm, show=show, save_individual=False, filtering=filter_bl)
        
        # Fit beamline (Ifit) - optional
        if splineFitBeamline > 0: 
            fit = UnivariateSpline(E, I0, s=splineFitBeamline)
            Ifit = fit(E)
            if show:
                if show == "absolute":
                    plot(E, Ifit, "r", label="spline fit")
                    title('Absolute')
                else:
                    nplot(E, Ifit, "r", label="spline fit")
                    title('Relative')

        # Add averaged beamline to figure if show=True (scaled) or show='absolute' (raw data)
        if show:
            if show == "absolute":
                plot(E, I0, "k", label="raw average")
                title('Absolute')
            else:
                nplot(E, I0, "k", label="raw average")
                title('Relative')
            legend()

        # PROCESS SAMPLE SCANS INDIVIDUALLY --> E, [I1,I2...] 
        # Open figure if plots are demanded 
        if show:
            figure()
            title("Sample scans")
            xlabel("Energy (eV)")
            ylabel("Initial intensity " + ("(arb. u.)" if show == "absolute" else "(scaled)"))
        
        # Load all beamline scans, average them and plot them if demanded 
        Es, I = self.scans(sample_scans, value, minX, maxX, stepX, norm=norm, show=show, save_individual=True, filtering=filter_sample)
        # Convert I.values() to an array
        I_array = array([I_array for I_array in I.values()])
        # Calculate the mean intensity
        mean_I = mean(I_array, axis=0)
        # Plot the mean intensity
        if show:
            if show == "absolute":
                plot(Es, mean_I, label="raw average", color='k', linestyle='-', linewidth=2)
            else:
                nplot(Es, mean_I, label="raw average", color='k', linestyle='-', linewidth=2)
            legend()
        
        # COMPUTE THE ABSORBANCE FOR EACH SCAN
        # Calculate the absorbance for each of the scans as A = -log10(I/I0)
        A = -log10(I_array/I0)                     # absorbance from raw data
        baseline=empty_like(A)
        A_all=empty_like(A)
        # Background subtraction
        if bgCorrectPoints > 0:
            # Perform linear fitting for each scan in A
            for i in range(A.shape[0]):  # Loop over each scan 
                scan = A[i,:]  
                # Perform linear fit 
                fit_params = polyfit(Es[:bgCorrectPoints], scan[:bgCorrectPoints], 1)  
                baseline[i, :] = polyval(fit_params, Es)
                A_all[i,:]=scan-baseline[i,:]
                # Optionally, plot the individual fit for each scan
                figure()
                if show:
                    plot(Es, scan, label=f'Scan {i+1} - data', alpha=0.7)
                    plot(Es, polyval(fit_params, Es), label=f'Scan {i+1} - fit', linestyle='--')
                    plot(Es[:bgCorrectPoints],scan[:bgCorrectPoints],"r", label="baseline of background substraction", linewidth=3)
                    legend()
        else:
            baseline = zeros(len(E))
            A_all=A

        # Calculate the mean absorbance of baseline subtracted data
        mean_A_all = mean(A_all, axis=0)

        # Calculate the mean absorbance and std for thickness estimation (+ mean baseline because why not?)
        mean_A = mean(A, axis=0)
        std_A = std(A, axis=0)
        mean_baseline = mean(baseline, axis=0)

        # Open figure if plots are demanded
        if show:
            figure()
            title("Absorbance")
            xlabel("Energy (eV)")
            ylabel("A " + ("(arb. u.)" if show == "absolute" else "(scaled)"))
            if show == "absolute":
                for scan_number, scan in zip(I.keys(), A_all): plot(Es, scan, label=f'{scan_number}', alpha=0.5)
                plot(Es, mean_A_all, label="raw average", color='k', linestyle='-', linewidth=2)
                plot(Es[:bgCorrectPoints],mean_A_all[:bgCorrectPoints],"r", label="baseline of background substraction", linewidth=4)
            else:
                for scan_number, scan in zip(I.keys(), A_all): nplot(Es, scan, label=f'{scan_number}', alpha=0.5)
                nplot(Es, mean_A_all, label="raw average", color='k', linestyle='-', linewidth=2)
            legend()    

        # Thickness estimation
        if energyT:
            idx_T = argmin(np.abs(Es - 395))
            bg_A_mean = mean_A[idx_T]
            bg_A = A[:, idx_T]
            bg_A_std = std_A[idx_T]
            
        if absorbance_solvent_1um>0 and concentration>0:
            d = bg_A_mean / absorbance_solvent_1um
            derr = bg_A_std  / absorbance_solvent_1um
            A =  A/d/concentration
            print("%d sample scans have been analyzed. The estimated sample thickness during these scans is (%g \u00B1 %g)\u03BCm."%(total_sample_scan_number, d, derr))
            d_array = bg_A / absorbance_solvent_1um
            A_norm = A/d_array[:, newaxis]/concentration
            
            if show:
                figure()
                title("extinction coefficient (OD/(um*M))")
                xlabel("Energy (eV)")
                ylabel("OD/(um*M)")
                if show == "absolute":
                    for scan_number, scan in zip(I.keys(), A_norm): plot(Es, scan, label=f'{scan_number}', alpha=0.5)
                    #plot(Es, mean_A_all, label="raw average", color='k', linestyle='-', linewidth=2)
                else:
                    for scan_number, scan in zip(I.keys(), A_norm): nplot(Es, scan, label=f'{scan_number}', alpha=0.5)
                    #nplot(Es, mean_A_all, label="raw average", color='k', linestyle='-', linewidth=2)
                legend()    
                
        else:
            print("%d sample scans have been analyzed. The background absorbance at %.2feV is (%g \u00B1 %g)OD."%(total_sample_scan_number, Es[idx_T], bg_A_mean, bg_A_std))

        return Es, mean_A_all, A_norm




   