import click
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6), dpi=80)
import os
from scipy.signal import find_peaks, argrelextrema,peak_widths
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


sampling_rate = 5 * 10**4

def run_analisys(datapath, chunk_size = 10**7):
    reader = pd.read_csv(datapath, iterator=True, chunksize=chunk_size) #'../2Gb signal.csv'

    tissue_data = []
    water_data = []

    for i, chunk in enumerate(reader):
        print("Processing data, chunk number", i)
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
        _, _, tissue_intervals_left, tissue_intervals_right = peak_widths(baseline_removed_signal, tissue_peaks_baseline, rel_height=0.5)
        _, _, water_intervals_left, water_intervals_right = peak_widths(baseline_removed_signal, water_peaks_baseline, rel_height=0.5)
        
        # Convert chunk coordinates into time coordinates
        transform = lambda x: (x + i * chunk_size) / sampling_rate

        for left, right in zip(tissue_intervals_left, tissue_intervals_right):
            tissue_data.append((
                transform(left), 
                transform(right),
                'tissue'
            ))

        # for left, right in zip(water_intervals_left, water_intervals_right):
        #     water_data.append((
        #         transform(left), 
        #         transform(right),
        #         'water'
        #     ))
        break
    
    if len(tissue_data) == 0:
        peaks_combined = sorted(water_data)
    elif len(water_data) == 0:
        peaks_combined = sorted(tissue_data)
    else:
        peaks_combined = sorted(tissue_data + water_data)
    
    return peaks_combined


# Saving analysis data to a file
def save_analysis_data(peaks_combined, path):
    # format to 6 decimal digits
    peaks_combined_format = map(lambda p: (format(p[0], '.6f'), format(p[1], '.6f'), p[2]), peaks_combined)

    df = pd.DataFrame(peaks_combined_format)
    df.to_csv(path, header=['startTime', 'endTime', 'label'], index=False)


# Reading analysis data from a file
def read_analysis_data(path):
    peaks_combined = []

    with open(path, 'w') as f:
        data = f.read()

        for line in data.split('\n'):
            if len(line) == 0:
                break
            
            # extract start, end and material of the peak from each line
            start, end, material = line.split(',')
            peak = (float(start), float(end), material)
            peaks_combined.append(peak)
    
    return peaks_combined


def visualize_analysis(datapath, peaks_combined, path, chunk_size = 10**5):
    reader = pd.read_csv(datapath, iterator=True, chunksize=chunk_size) #'../2Gb signal.csv'

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

    # Perform visualization on baseline removed data
    print("Saving visualization data:")
    iterator = 0
    for i, chunk in enumerate(reader):
        print("Visualizing data, chunk number", i)

        signal = chunk['adc2']

        plt.ylim(0, 4095)
        index = [s / sampling_rate for s in list(signal.index)]
        plt.plot(index, signal)

        has_tissue = False
        while True:
            # print(iterator, peaks_combined[iterator])
            # print(index[0], index[-1])
            if iterator >= len(peaks_combined):
                break

            start, end, material = peaks_combined[iterator]

            if start > index[-1]:
                break

            if end < index[0]:
                iterator += 1
                continue

            if end > index[-1]:
                end = index[-1]

            if start < index[0]:
                start = index[0]

            if material == 'tissue':
                plt.axvspan(start, end, 0, 4095, facecolor="g", alpha=0.1)
                has_tissue = True

            iterator += 1


        # plt.plot(tissue_peaks_baselineremoved[i], baseline_data[i][tissue_peaks_baselineremoved[i]], "o", color='red', label='Tissue Peaks')
        # plt.scatter(water_peaks_baselineremoved[i], baseline_data[i][water_peaks_baselineremoved[i]], color='green', label='Water Peaks', s=25)
        # plt.plot(water_peaks_baselineremoved[i], baseline_data[i][water_peaks_baselineremoved[i]], "o", color='green', label='Water Peaks')

        if has_tissue:
            plt.savefig(os.path.join(path, 'final_baselineremoved_{}.png'.format(i)))
        plt.clf()
        plt.close('all')


# Let's put it aside for some time, will finish if we have time 
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


@click.command()
@click.option("--datapath", required=True, help="Path to the signal file", type=str)
@click.option(
    "--savepath", required=True, help="Path to the save/read the analysis data", type=str
)
@click.option(
    "--analyze",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to perform the analysis of the signal",
)
@click.option(
    "--visualize",
    is_flag=True,
    show_default=True,
    default=False,
    help="The person to greet.",
)
@click.option(
    "--save",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to save/overwrite anaylysis results into a file",
)
@click.option(
    "--vis_path", default="images", show_default=True, help="If provided, visualization will be saved to this folder", type=str
)
def run(datapath, savepath, analyze, visualize, save, vis_path):
    print(datapath, savepath, analyze, visualize, save, vis_path)

    if analyze:
        peaks_combined = run_analisys(datapath)

        if save:
            save_analysis_data(peaks_combined, savepath)
    else:
        peaks_combined = read_analysis_data(savepath)

    if visualize:
        visualize_analysis(datapath, peaks_combined, vis_path)


if __name__ == '__main__':
    # python final_code.py --datapath="../2Gb signal.csv" --savepath="../tmp1.csv" --analyze --save --visualize --vis_path="./images"
    run()