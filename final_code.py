import click
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, argrelextrema, peak_widths
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


sampling_rate = 5 * 10 ** 4


def run_analisys(datapath, chunk_size=10 ** 7):
    reader = pd.read_csv(datapath, iterator=True, chunksize=chunk_size)

    tissue_data = []
    water_data = []

    for i, chunk in enumerate(reader):
        print("Processing data, chunk number", i)
        x = chunk["adc2"]
        # Tissue and water peaks before baseline removal
        tissue_peaks, properties = find_peaks(x, height=(1200, 4000), distance=25000)
        water_peaks, properties = find_peaks(x, height=(635, 755), distance=25000)

        # Processed Tissue and WaterPeak After Baseline Removal
        smoothed_y = gaussian_filter1d(x, sigma=5)
        peaks, _ = find_peaks(smoothed_y, height=(1200, 4000))
        valleys, _ = find_peaks(-smoothed_y, height=(-1200, -4000))
        minima = argrelextrema(smoothed_y, np.less)[0]
        baseline = np.interp(np.arange(len(smoothed_y)), minima, smoothed_y[minima])
        baseline_removed_signal = x - baseline
        tissue_peaks_baseline, _ = find_peaks(
            baseline_removed_signal, height=(500, 5000), distance=25000
        )
        water_peaks_baseline, _ = find_peaks(
            baseline_removed_signal, height=(-50, 80), distance=25000
        )

        # Get the peak widts, its starting left and ending right coordinates
        _, _, tissue_intervals_left, tissue_intervals_right = peak_widths(
            baseline_removed_signal, tissue_peaks_baseline, rel_height=0.5
        )
        _, _, water_intervals_left, water_intervals_right = peak_widths(
            baseline_removed_signal, water_peaks_baseline, rel_height=0.5
        )

        # Convert chunk coordinates into time coordinates
        transform = lambda x: (x + i * chunk_size) / sampling_rate

        # Saving tissue peaks with its left and right boarders
        for left, right in zip(tissue_intervals_left, tissue_intervals_right):
            tissue_data.append((transform(left), transform(right), "tissue"))

        # Saving water peaks with its left and right boarders
        for left, right in zip(water_intervals_left, water_intervals_right):
            water_data.append((transform(left), transform(right), "water"))

    # Merging water and tissue peaks together and sort them chronologically
    if len(tissue_data) == 0:
        peaks_combined = sorted(water_data)
    elif len(water_data) == 0:
        peaks_combined = sorted(tissue_data)
    else:
        peaks_combined = sorted(tissue_data + water_data)

    return peaks_combined


def save_analysis_data(peaks_combined, path):
    # Format to 6 decimal digits
    peaks_combined_format = map(
        lambda p: (format(p[0], ".6f"), format(p[1], ".6f"), p[2]), peaks_combined
    )

    # Save to a file
    df = pd.DataFrame(peaks_combined_format)
    df.to_csv(path, header=["startTime", "endTime", "label"], index=False)


def read_analysis_data(path):
    peaks_combined = []

    with open(path, "r") as f:
        data = f.read()

        for i, line in enumerate(data.split("\n")):
            if i == 0:
                continue
            if len(line) == 0:
                break

            # Extract start, end and material of the peak from each line and write to a list storage
            start, end, material = line.split(",")
            peak = (float(start), float(end), material)
            peaks_combined.append(peak)

    return peaks_combined


# Visualize the result of the analysis (computed or read from a file)
def visualize_analysis(datapath, peaks_combined, path, chunk_size=10 ** 6, lines=5):
    reader = pd.read_csv(datapath, iterator=True, chunksize=chunk_size)

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

        # Get the index of the chuck with respect to its visualization possition (since we are stacking several chunks on one plot)
        plot_index = i % lines

        # Initializing the plots and setting axis limit
        if plot_index == 0:
            f, axarr = plt.subplots(lines, 1, figsize=(100, 30), dpi=200)
        axarr[plot_index].set_ylim(0, 4095)

        # Plotting the signal
        signal = chunk["adc2"]
        index = [s / sampling_rate for s in list(signal.index)]
        axarr[plot_index].plot(index, signal)

        # An algorithm to align multy-chunck peaks (ivolves iterator over the peaks and looping through chuncks)
        while True:
            # If no more peaks left quit
            if iterator >= len(peaks_combined):
                break

            # Extract the peak data
            start, end, material = peaks_combined[iterator]

            # if start of the peak is further then the end of the chunk - go for a next chunk
            if start > index[-1]:
                break

            # if end of the peak has finished before the chunk has started - go to the next peak
            if end < index[0]:
                iterator += 1
                continue

            # if peak ends further then the chunk - cut it by the chunk border
            if end > index[-1]:
                end = index[-1]

            # if peak starts earlier then the chunk - cut it by the chunk border
            if start < index[0]:
                start = index[0]

            # plot the tissue peaks
            if material == "tissue":
                axarr[plot_index].axvspan(start, end, 0, 4095, facecolor="g", alpha=0.1)

            # go to the next peak
            iterator += 1

        # Saving plots
        if plot_index == lines - 1:
            f.savefig(
                os.path.join(path, "visualization_part_{}.png".format(i // lines)),
                bbox_inches="tight",
                pad_inches=0,
            )
            f.clf()
            plt.close("all")


# NOT USED IN THE FINAL SOLUTION
def kmeans_cluster():
    all_peak_features = []

    # Append columns from the loop to all_peak_features
    for i in range(
        len(save_data)
    ):  # Assuming the range is up to 3 based on your example
        peak_features = np.column_stack(
            (
                tissue_peaks_baselineremoved[i],
                baseline_data[i][tissue_peaks_baselineremoved[i]],
                peak_width_lst[i],
            )
        )
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
        previous_chunk_length = sum(
            len(tissue_peaks_baselineremoved[i]) for i in range(chunk_num)
        )

        # Filter data for the current chunk
        chunk_data = result[
            previous_chunk_length : previous_chunk_length
            + len(tissue_peaks_baselineremoved[chunk_num])
        ]

        # Apply the labels from the global clustering to the chunk
        chunk_labels = labels[
            previous_chunk_length : previous_chunk_length
            + len(tissue_peaks_baselineremoved[chunk_num])
        ]

        plt.ylim(-700, 4500)
        plt.plot(baseline_data[chunk_num], label="Signal")
        plt.plot(
            tissue_peaks_baselineremoved[chunk_num],
            baseline_data[chunk_num][tissue_peaks_baselineremoved[chunk_num]],
            "rx",
            label="Detected Peaks",
            markersize=10,
        )
        plt.title(f"Peak Detection and K-Means Clustering - Chunk {chunk_num + 1}")
        plt.legend()
        # Display different colors for different clusters
        for cluster_num in range(k):
            cluster_points = chunk_data[chunk_labels == cluster_num]
            plt.plot(
                cluster_points[:, 0],
                cluster_points[:, 1],
                "o",
                label=f"Cluster {cluster_num + 1}",
            )

        plt.legend()
        plt.savefig("images/final_baselineremoved_clustered{}.png".format(chunk_num))

        plt.clf()
        plt.close("all")
        print("saving clustered", chunk_num)


@click.command()
@click.option("--datapath", required=True, help="Path to the signal file", type=str)
@click.option(
    "--savepath",
    required=True,
    help="Path to the save/read the analysis data",
    type=str,
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
    help="Whether to visualize the result of the analysis",
)
@click.option(
    "--save",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to save/overwrite anaylysis results into a file",
)
@click.option(
    "--vis_path",
    default="./images",
    show_default=True,
    help="If provided, visualization will be saved to this folder",
    type=str,
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


if __name__ == "__main__":
    # Example of running a program
    # python final_code.py --datapath="../2Gb signal.csv" --savepath="../tmp1.csv" --analyze --save --visualize --vis_path="./images"
    run()
