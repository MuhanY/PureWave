import pandas as pd
import numpy as np
import warnings
import neurokit2 as nk
import ast
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import welch, butter, filtfilt

# -----------------------------------------
# 1. Basic preprocessing
# -----------------------------------------
# Helper function to convert time to index (500 Hz sampling rate)
def time_to_index(time_in_seconds):
    return int(time_in_seconds * 500)

def lead_label_generation(signal_data, label_data, artifact_type_map, no_artifact_label=0):
    # Initialize labels array for lead-specific artifact information
    num_samples, seq_length, num_leads = signal_data.shape
    lead_labels = np.zeros((num_samples, seq_length, num_leads), dtype=np.int64)  # Shape: (samples, time_steps, leads)

    # Process label data for artifact types and positions
    for idx, row in label_data.iterrows():
        for lead_num in range(1, 13):  # Leads are from 1 to 12
            lead_info = row[str(lead_num)]
            if lead_info != "[]":
                artifacts = ast.literal_eval(lead_info)
                for artifact in artifacts:
                    for artifact_type, (start_time, end_time) in artifact.items():
                        start_idx = time_to_index(start_time)
                        end_idx = time_to_index(end_time)
                        if artifact_type in artifact_type_map:
                            lead_labels[idx, start_idx:end_idx, lead_num - 1] = artifact_type_map[artifact_type]
                        else:
                            lead_labels[idx, start_idx:end_idx, lead_num - 1] = no_artifact_label
    return lead_labels

# Baseline deivation removal
def removd_baseline_deviation(signal):        
    # Apply baseline deviation removal using a high-pass filter
    b, a = butter(1, 0.01 / (500 / 2), btype='highpass')
    signal = filtfilt(b, a, signal, axis=0)
    return signal

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Raw signal normalization function (apply seperately to limb/precordial leads groups)
def normalize_signal_raw(signal):
    signal[:, :6] = minmax_scaler.fit_transform(signal[:, :6])
    signal[:, 6:] = minmax_scaler.fit_transform(signal[:, 6:])
    return signal

# Feature normalization function
def normalize_signal_features(signal):
    signal = standard_scaler.fit_transform(signal).astype(np.float32)
    return signal

# -----------------------------------------
# 2. Chunking
# -----------------------------------------
# Helper function to find longest sequence of each artifact
def longest_sequence(arr):
    unique_values = np.unique(arr)
    longest_sequences = {}

    for val in unique_values:
        # Create a mask where the value is equal to val
        mask = (arr == val).astype(int)
        
        # Find consecutive ones in the mask (which represent consecutive occurrences of `val`)
        changes = np.diff(np.concatenate(([0], mask, [0])))  # Add 0 at the start and end
        starts = np.where(changes == 1)[0]  # Start indices of sequences
        ends = np.where(changes == -1)[0]   # End indices of sequences
        
        # Compute lengths of consecutive sequences
        lengths = ends - starts
        
        # Store the maximum length for this value
        longest_sequences[val] = lengths.max() if lengths.size > 0 else 0
    
    return longest_sequences

# Labeling function for chunk signals (lead-wise).
# Cutoffs for the longest sequence are 1.4sec for NCR, 0.8sec for BW, 0.4sec for HF, SF, and MA.
def artifact_for_chunk(chunk):
    artifacts_by_lead = []
    
    for lead in range(chunk.shape[1]):
        # Get non-zero values in the lead
        non_zero_values = chunk[:, lead][chunk[:, lead] != 0]
        
        # If there is a non-zero value, parse the length of the sequence and add to list if its longer than the threshold
        if non_zero_values.size > 0:
            longest_sequences = longest_sequence(non_zero_values)
            artifacts = list(longest_sequences.keys())
            artifacts = [x for x in artifacts if x != 0]
            artifact_of_lead = 0
            for artifact in artifacts:
                if (artifact == 1) and (longest_sequences[artifact] >= 700):
                    artifact_of_lead = artifact
                elif (artifact == 4) and (longest_sequences[artifact] >= 400):
                    artifact_of_lead = artifact
                elif (artifact == 2 or artifact == 3) and (longest_sequences[artifact] >= 200):
                    artifact_of_lead = artifact
                elif (artifact == 5) and (longest_sequences[artifact] >= 200):
                    artifact_of_lead = artifact
            artifacts_by_lead.append(artifact_of_lead)
        else:
            artifacts_by_lead.append(0)  # If all values are zero (no artifact in the lead)
            
    return artifacts_by_lead

# Chunking the data into pieces
def chunk_signal(signal, global_label, lead_label, chunk_size, step):
    signal_chunks = []
    lead_label_chunks = []
    global_label_chunks = []
    for start in range(0, signal.shape[0] - chunk_size + 1, step):
        end = start + chunk_size
        # Signal chunk
        signal_chunks.append(signal[start:end])
        
        # Lead label chunk (1 label for each lead)
        lead_label_chunk = artifact_for_chunk(lead_label[start:end, :])
        lead_label_chunks.append(lead_label_chunk)
        
        # Global label chunk (1 label for each chunk)
        global_label_chunk = 0 if all(x == 0 for x in lead_label_chunk) else 1
        global_label_chunks.append(global_label_chunk)
    return signal_chunks, lead_label_chunks, global_label_chunks

def Chunking(signals, global_labels, lead_labels, chunk_size, step):
    # Prepare lists to hold chunked data
    signals_chunks = []
    global_labels_chunks = []
    lead_labels_chunks = []

    # Iterate over signals
    for i in range(len(signals)):
        signal = signals[i]
        global_label = global_labels[i]
        lead_label = lead_labels[i]
        
        signal_chunks, lead_label_chunks, global_label_chunks = chunk_signal(signal, global_label, lead_label, chunk_size, step)
        
        signals_chunks.extend(signal_chunks)
        global_labels_chunks.extend(global_label_chunks)
        lead_labels_chunks.extend(lead_label_chunks)

    # Convert lists to numpy arrays
    signals_chunks = np.array(signals_chunks)
    global_labels_chunks = np.array(global_labels_chunks)
    lead_labels_chunks = np.array(lead_labels_chunks)
    
    return signals_chunks, global_labels_chunks, lead_labels_chunks

# -----------------------------------------
# 3. Feature extraction
# -----------------------------------------

# Helper function to calcualte signal-to-noise ratio
def calculate_snr(signal, noise):
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Helper function to calculate the HRV, amplitude, baseline drift, and SNRs.
def _compute_signal_features(signal):
    # Cleaned signal and Peak detection
    signal_cleaned = nk.ecg_clean(signal, sampling_rate=500, method='neurokit', powerline = 60)
    peaks, info = nk.ecg_peaks(signal_cleaned, sampling_rate=500, method='neurokit', correct_artifacts=False)
    
    # 1. HR variability
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            HRV = nk.hrv_time(info, sampling_rate=500)
            SDNN = HRV["HRV_SDNN"][0]
            SDSD = HRV["HRV_SDSD"][0]
        except IndexError:
            SDNN = 0
            SDSD = 0
        
    # 2. Signal amplitude
    amplitude = np.max(signal) - np.min(signal)

    # 3. Baseline drift
    signal_detrended = nk.signal_detrend(signal)
    baseline_drift = signal - signal_detrended
    drift = np.mean(baseline_drift)

    # 4. Signal-to-Noise Ratio (SNR) for AC interference
    signal_denoised_AC = nk.signal_filter(signal, method="powerline", powerline=60)
    snr_AC = calculate_snr(signal, signal - signal_denoised_AC)
    
    # 5. Signal-to-Noise Ratio (SNR) for general
    snr_total = calculate_snr(signal, signal - signal_cleaned)

    return SDNN, SDSD, amplitude, drift, snr_AC, snr_total
    
# Calculate the time-domain features.
def compute_signal_features(signal):
    SDNNs = []
    SDSDs = []
    amplitudes = []
    drifts = []
    ACs = []
    totals = []
    
    # features for the whole signal
    SDNN, SDSD, amplitude, drift, AC, total = _compute_signal_features(signal)

    SDNNs.append(SDNN)
    SDSDs.append(SDSD)
    amplitudes.append(amplitude)
    drifts.append(drift)
    ACs.append(AC)
    totals.append(total)
    
    # features for 2-sec segments
    for i in range(int(signal.shape[0]/1000)):
        _, _, amplitude, drift, AC, total = _compute_signal_features(signal[i*1000:i*1000+1000])
        amplitudes.append(amplitude)
        drifts.append(drift)
        ACs.append(AC)
        totals.append(total)

    return SDNNs + SDSDs + amplitudes + drifts + ACs + totals 
    
# Pearson correlation among the 12 leads (min-absolute value & mean value).
def correlation_among_leads(signals):
    correlation_matrix = signals.corr()
    
    # Find the index of the value nearest to zero and extract the value
    nearest_to_zero_index = np.nanargmin(np.abs(correlation_matrix))
    row, col = np.unravel_index(nearest_to_zero_index, correlation_matrix.shape)
    value_nearest_to_zero = correlation_matrix.to_numpy()[row, col]
    
    # Calculate the mean value of corr matrix
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    mean = upper_triangle.stack().mean()
    
    return value_nearest_to_zero, mean

# Frequency-domain features
def compute_psd(data, fs):
    """
    Computes the Power Spectral Density (PSD) using Welch's method.
    """
    freqs, psd = welch(data, fs=fs, nperseg=min(data.shape[1], int(fs * 2)), axis=1)
    return freqs, psd

# Other time-domain features
def compute_statistical_features(signal):
    """
    Computes statistical features of the signal.
    """
    mean_val = np.mean(signal, axis=1)
    std_val = np.std(signal, axis=1)
    skewness = skew(signal, axis=1)
    kurt = kurtosis(signal, axis=1)
    rms = np.sqrt(np.mean(signal**2, axis=1))
    peak_to_peak = np.ptp(signal, axis=1)
    return mean_val, std_val, skewness, kurt, rms, peak_to_peak

def extract_global_features_vectorized(data, fs, num_leads):
    """
    Vectorized extraction of global features for each sample and each lead.
    """
    num_samples = data.shape[0]
    features_per_lead = 30 # 26 Time-domain features + 4 Frequency-domain features
    total_features = num_leads * features_per_lead
    features = np.zeros((num_samples, total_features), dtype=np.float32)

    def process_lead(lead):
        lead_signals = data[:, :, lead]  # Shape: (num_samples, num_timepoints)

        # Time-domain features
        signal_features = np.array([compute_signal_features(x) for x in lead_signals])

        # Frequency-domain features
        freqs, psd = compute_psd(lead_signals, fs)  # psd shape: (num_samples, num_freq_bins)
        bands = {
            'wandering_baseline': (0.4, 3),
            'motion_artifact': (0.4, 15),
            'muscle_tremor': (4, 150),
            'ac_interference': (58, fs/2)
        }

        freq_features = []
        for band in bands.values():
            idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])  # idx_band shape: (num_freq_bins,)

            if idx_band.any():
                # Compute mean power across the selected frequency bins for each sample
                mean_power = psd[:, idx_band].mean(axis=1)  # Shape: (num_samples,)
            else:
                # If no frequency bins fall within the band, set mean power to 0 for all samples
                mean_power = np.zeros(num_samples)

            freq_features.append(mean_power)  # Append mean power array of shape (num_samples,)

        # Combine features for this lead
        freq_features = np.array(freq_features).T
        combined_features = np.concatenate((signal_features, freq_features), axis=1)  # Shape: (num_samples, features_per_lead)
        
        # Set nan values to 0 (Unspecified HRV cases)
        combined_features = np.nan_to_num(combined_features, nan=0)
        
        return combined_features

    # Parallelize the loop over leads
    lead_features = Parallel(n_jobs=-1)(delayed(process_lead)(lead) for lead in range(num_leads))

    # Assign features from each lead to the final array
    for lead in range(num_leads):
        start_idx = lead * features_per_lead
        end_idx = (lead + 1) * features_per_lead
        features[:, start_idx:end_idx] = lead_features[lead]
        
    # Add correlation features (global, 2)
    correlations = np.array([correlation_among_leads(pd.DataFrame(x)) for x in data])
    features = np.concatenate((features, correlations), axis=1)
    
    # Add correlation features (2-sec segments, 6)
    for i in range(int(data.shape[1]/1000)):
        data[:, i*1000:i*1000+1000, :]
        seg_correlations = np.array([correlation_among_leads(pd.DataFrame(x)) for x in data])
        features = np.concatenate((features, seg_correlations), axis=1)
    
    return features

# Feature naming function
def feature_names(target_include=True):
    suffixes_seg = ['whole', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5']

    amplitude_strings = [f'amplitude_{suffix}' for suffix in suffixes_seg]
    drift_strings = [f'drift_{suffix}' for suffix in suffixes_seg]
    ACsnr_strings = [f'AC_{suffix}' for suffix in suffixes_seg]
    total_strings = [f'total_{suffix}' for suffix in suffixes_seg]
    freqs = ['bandWB', 'bandMA', 'bandMT', 'bandAC']
    colnames_lead = ['SDNN_whole', 'SDSD_whole'] + amplitude_strings + drift_strings + ACsnr_strings + total_strings + freqs
    colnames_leads = []
    for i in range(12):
        colnames_leads += [f'{name}_lead{i+1}' for name in colnames_lead]
        
    corr_names = ['Corr_min', 'Corr_mean', 'Corr_min_seg1', 'Corr_mean_seg1', 'Corr_min_seg2', 'Corr_mean_seg2',
                'Corr_min_seg3', 'Corr_mean_seg3', 'Corr_min_seg4', 'Corr_mean_seg4', 'Corr_min_seg5', 'Corr_mean_seg5']
    
    if target_include:
        merged_colnames = colnames_leads + corr_names + ['target']
    else:
        merged_colnames = colnames_leads + corr_names
    return merged_colnames