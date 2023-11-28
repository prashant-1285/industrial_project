
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, argrelextrema,peak_widths
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


chunk_size = 10 ** 7
reader = pd.read_csv('2Gb signal.csv', iterator=True, chunksize=chunk_size)
save_data = []
tissue_peak = []
water_peak=[]
baseline_data=[]
tissue_peaks_baselineremoved=[]
water_peaks_baselineremoved=[]
peak_width_lst=[]


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
        tissue_peaks_baseline, _ = find_peaks(baseline_removed_signal, height=(500, 5000), distance=25000)
        water_peaks_baseline, _ = find_peaks(baseline_removed_signal, height=(-50, 80), distance=25000)
        widths_baseline, widths_heights, widths_interval_left, widths_interval_right = peak_widths(baseline_removed_signal, tissue_peaks_baseline, rel_height=0.5)
        peak_width_lst.append(widths_baseline)
    
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
        
def kmeans_cluster():
    all_peak_features = []

    # Append columns from the loop to all_peak_features
    for i in range(len(save_data)):  # Assuming the range is up to 3 based on your example
        peak_features = np.column_stack((tissue_peaks_baselineremoved[i], baseline_data[i][tissue_peaks_baselineremoved[i]],peak_width_lst[i]))
        all_peak_features.append(peak_features)

    # Concatenate all the appended columns into a single array
    result = np.concatenate(all_peak_features, axis=0)
    # Normalize the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(result)
    print("fitted data for k means")
    # Choose the number of clusters
    k = 3

    # Apply KMeans clustering to the entire dataset
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_features)
    labels = kmeans.labels_


    path = "images"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    # # Loop through each chunk
    for chunk_num in range(len(save_data)):
        # Calculate the total length of data from previous chunks
        previous_chunk_length = sum(len(tissue_peaks_baselineremoved[i]) for i in range(chunk_num))

        # Filter data for the current chunk
        chunk_data = result[previous_chunk_length:previous_chunk_length + len(tissue_peaks_baselineremoved[chunk_num])]

        # Apply the labels from the global clustering to the chunk
        chunk_labels = labels[previous_chunk_length:previous_chunk_length + len(tissue_peaks_baselineremoved[chunk_num])]

        
        plt.ylim(-700, 4500)
        plt.plot(baseline_data[chunk_num], label='Signal')
        plt.plot(tissue_peaks_baselineremoved[chunk_num], baseline_data[chunk_num][tissue_peaks_baselineremoved[chunk_num]], 'rx', label='Detected Peaks', markersize=10)
        plt.title(f'Peak Detection and K-Means Clustering - Chunk {chunk_num + 1}')
        plt.legend()
        # Display different colors for different clusters
        for cluster_num in range(k):
            cluster_points = chunk_data[chunk_labels == cluster_num]
            plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'o', label=f'Cluster {cluster_num + 1}')
        
        plt.legend()
        plt.savefig('images/final_baselineremoved_clustered{}.png'.format(chunk_num))
       
        plt.clf()
        plt.close('all')
        print("saving clustered",chunk_num)






if __name__ == '__main__':
    main()
    visualize()
    kmeans_cluster()