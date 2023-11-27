
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, argrelextrema
from scipy.ndimage import gaussian_filter1d



chunk_size = 10 ** 7
reader = pd.read_csv('2Gb signal.csv', iterator=True, chunksize=chunk_size)
save_data = []
tissue_peak = []
water_peak=[]
baseline_data=[]
tissue_peaks_baselineremoved=[]
water_peaks_baselineremoved=[]



def main():
    cumulative_length = 0  # Variable to keep track of the cumulative length of chunks
    for i, chunk in enumerate(reader):
        print("datas",i)
        x = chunk['adc2']
        # Tissue and water peaks before baseline removal
        tissue_peaks, properties = find_peaks(x, height=(1200, 4000), distance=25000)
        water_peaks, properties = find_peaks(x, height=(635, 755), distance=25000)
        
    
        
        #Processed Tissue and WaterPeak After Baseline Removal 
        smoothed_y = gaussian_filter1d(x, sigma=5)
        peaks, _ = find_peaks(smoothed_y, height=(1200,4000))
        valleys, _ = find_peaks(-smoothed_y, height=(-1200,-4000))
        minima = argrelextrema(smoothed_y, np.less)[0]
        baseline = np.interp(np.arange(len(smoothed_y)), minima, smoothed_y[minima])
        baseline_removed_signal = x - baseline
        tissue_peaks_baseline, _ = find_peaks(baseline_removed_signal, height=(500, 1500), distance=25000)
        water_peaks_baseline, _ = find_peaks(baseline_removed_signal, height=(-50, 80), distance=25000)
    
    
        # Adjust the indices of peaks based on the cumulative length on normal peaks
        adjusted_tissue_peaks = [tissue_peak + cumulative_length for tissue_peak in tissue_peaks]
        adjusted_water_peaks = [water_peak + cumulative_length for water_peak in water_peaks]

        # Adjust the indices of peaks based on the cumulative length on  baseline removed peaks
        adjusted_tissue_peaks_br = [tissue_peak_br + cumulative_length for tissue_peak_br in tissue_peaks_baseline]
        adjusted_water_peaks_br = [water_peak_br + cumulative_length for water_peak_br in water_peaks_baseline]

        # append the saved data and baselineremoved data
        save_data.append(x)
        baseline_data.append(baseline_removed_signal)

        # Append on the normal data (without baseline removal)
        tissue_peak.append(adjusted_tissue_peaks)
        water_peak.append(adjusted_water_peaks)

        # Append on the baseline reomoved data
        tissue_peaks_baselineremoved.append(adjusted_tissue_peaks_br)
        water_peaks_baselineremoved.append(adjusted_water_peaks_br)

        cumulative_length += len(chunk)  # Update the cumulative length

def visualize():
    path = "images"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    # Perform visualization on baseline removed data
    for i in range(len(save_data)):
        plt.ylim(-1000, 4500)
        plt.plot(baseline_data[i])
        plt.plot(tissue_peaks_baselineremoved[i], baseline_data[i][tissue_peaks_baselineremoved[i]], "o", color='red', label='Tissue Peaks')
        plt.plot(water_peaks_baselineremoved[i], baseline_data[i][water_peaks_baselineremoved[i]], "o", color='green', label='water Peaks')

        plt.savefig('images/final_baselineremoved_{}.png'.format(i))
        plt.clf()
        plt.close('all')
        print("saving",i)
        
if __name__ == '__main__':
    main()
    visualize()