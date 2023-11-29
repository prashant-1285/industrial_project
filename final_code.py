
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
reader = pd.read_csv('/home/prashant/Documents/legion/UEF/idp/2Gb signal.csv', iterator=True, chunksize=chunk_size)
save_data = []
tissue_peak = []
water_peak=[]
baseline_data=[]
tissue_peaks_baselineremoved=[]
water_peaks_baselineremoved=[]
peak_width_lst=[]
tissue_peak_info_csv=[]
water_peak_info_csv=[]

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
        
        # Get the peak widts, its starting left and ending right coordinates
        widths_baseline, widths_heights, widths_interval_left, widths_interval_right = peak_widths(baseline_removed_signal, tissue_peaks_baseline, rel_height=0.5)
        water_widths_baseline, water_widths_heights, water_widths_interval_left, water_widths_interval_right = peak_widths(baseline_removed_signal, water_peaks_baseline, rel_height=0.5)
        
        peak_width_lst.append(widths_baseline)
        tissue_peak_info_csv.append((widths_interval_left,widths_interval_right))
        water_peak_info_csv.append((water_widths_interval_left,water_widths_interval_right))
    
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

def get_csv_data():
    """
    Take the all the peaks of tissue and water and convert their starting and ending point of the peaks into the following format.
    For example: combined_peaks:[(0.00100,0.00200,'water'),(0.00400,0.00600,'tissue')]
    """
    tissue_csv_list=[]
    water_csv_list=[]
    for item in tissue_peak_info_csv:
        for i in range(len(item[0])):
            tissue_csv_list.append((item[0][i], item[1][i],'tissue'))
    
    for item in water_peak_info_csv:
        for i in range(len(item[0])):
            water_csv_list.append((item[0][i], item[1][i],'water'))
    #print("tissue peak info ",tissue_peak_info_csv)
    #print("tissue peak info correct format",tissue_csv_list)
    #print("water peak info ",water_peak_info_csv)
    #print("water peak info correct format ",water_csv_list)
    print("length of water peaks",len(water_csv_list))
    print("length of tissue peaks",len(tissue_csv_list))

    # Combine the peaks serially
    combined_peaks = []

    # Merge peaks based on their order
    i, j = 0, 0
    while i < len(water_csv_list) and j < len(tissue_csv_list):
        water_start, water_end, water_label = water_csv_list[i]
        tissue_start, tissue_end, tissue_label = tissue_csv_list[j]

        # Compare the start points of water and tissue peaks
        if water_start < tissue_start:
            combined_peaks.append((water_start, water_end, water_label))
            i += 1
        else:
            combined_peaks.append((tissue_start, tissue_end, tissue_label))
            j += 1
    # Append any remaining peaks from water or tissue
    combined_peaks.extend(water_csv_list[i:])
    combined_peaks.extend(tissue_csv_list[j:])

    
    print("length of combined peaks",len(combined_peaks))
    print("first few elemetns",combined_peaks[:20])

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
    get_csv_data()
    visualize()
    kmeans_cluster()