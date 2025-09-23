# COR分析工具 - 针对t_data_分割后样本.csv数据

import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt

def calculate_skf6205_frequencies(n):
    """
    Calculates the characteristic frequencies for an SKF6205 (DE) bearing.
    Args:
        n (float): The rotational speed of the inner race in rpm.
    Returns:
        A dictionary containing the calculated frequencies:
        BPFO, BPFI, BSF, and FTF.
    """
    # Bearing parameters for SKF6205 (DE)
    Nd = 9.0  # Number of rolling elements
    d = 0.3126  # Rolling element diameter (inches)
    D = 1.537  # Pitch diameter (inches)

    # Convert rpm to Hz
    fr = n / 60.0

    # Calculate frequencies
    BPFO = fr * (Nd / 2.0) * (1 - d / D)
    BPFI = fr * (Nd / 2.0) * (1 + d / D)
    BSF = fr * (D / d) * (1 - (d / D)**2)
    FTF = 0.5 * fr * (1 - d / D)

    return {
        "BPFO": BPFO,
        "BPFI": BPFI,
        "BSF": BSF,
        "FTF": FTF
    }

def high_pass_filter(signal_data, cutoff_freq, sampling_rate, order=4):
    """
    Apply high-pass filter to remove low-frequency interference
    """
    nyquist = sampling_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal_data)
    return filtered_signal

def hilbert_envelope(signal_data):
    """
    Apply Hilbert transform to extract envelope
    """
    analytic_signal = hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    return envelope

def order_analysis(signal_data, rpm, sampling_rate):
    """
    Perform order analysis to reduce speed influence
    """
    # Convert rpm to Hz
    rotation_freq = rpm / 60.0

    # Resample signal to angular domain
    # This is a simplified version - full order analysis would require more sophisticated resampling
    time = np.arange(len(signal_data)) / sampling_rate

    # Create angle vector based on rotation frequency
    angle = 2 * np.pi * rotation_freq * time

    # For now, return the signal normalized by rotation frequency
    return signal_data / rotation_freq

def compute_envelope_spectrum(envelope_signal, sampling_rate):
    """
    Compute FFT to get envelope spectrum
    """
    n = len(envelope_signal)
    fft_values = fft(envelope_signal)
    frequencies = fftfreq(n, 1/sampling_rate)

    # Return only positive frequencies
    positive_freq_idx = frequencies >= 0
    return frequencies[positive_freq_idx], np.abs(fft_values[positive_freq_idx])

def calculate_cor_index(spectrum_frequencies, spectrum_amplitudes, fault_frequencies, bandwidth=5):
    """
    Calculate COR (Cyclostationarity Order Ratio) index to select spectral features
    This is a simplified version - real COR would be more complex
    """
    cor_scores = {}

    for fault_name, fault_freq in fault_frequencies.items():
        # Find frequency bin closest to fault frequency
        idx = np.argmin(np.abs(spectrum_frequencies - fault_freq))

        # Calculate energy around fault frequency
        bandwidth_idx = int(bandwidth / (spectrum_frequencies[1] - spectrum_frequencies[0]))
        start_idx = max(0, idx - bandwidth_idx)
        end_idx = min(len(spectrum_amplitudes), idx + bandwidth_idx + 1)

        fault_energy = np.sum(spectrum_amplitudes[start_idx:end_idx]**2)

        # Calculate total energy
        total_energy = np.sum(spectrum_amplitudes**2)

        # COR index (ratio of fault energy to total energy)
        cor_index = fault_energy / total_energy if total_energy > 0 else 0
        cor_scores[fault_name] = cor_index

    return cor_scores

def process_signal_sample(signal_data, rpm, sampling_rate=12000, highpass_cutoff=100):
    """
    Complete signal processing pipeline for one sample
    """
    # Step 1: High-pass filter
    filtered_signal = high_pass_filter(signal_data, highpass_cutoff, sampling_rate)

    # Step 2: Hilbert transform for envelope
    envelope = hilbert_envelope(filtered_signal)

    # Step 3: Order analysis
    order_normalized = order_analysis(envelope, rpm, sampling_rate)

    # Step 4: FFT envelope spectrum
    frequencies, spectrum = compute_envelope_spectrum(order_normalized, sampling_rate)

    # Step 5: Calculate fault characteristic frequencies
    fault_frequencies = calculate_skf6205_frequencies(rpm)

    # Step 6: Calculate COR index
    cor_scores = calculate_cor_index(frequencies, spectrum, fault_frequencies)

    return {
        'filtered_signal': filtered_signal,
        'envelope': envelope,
        'order_normalized': order_normalized,
        'frequencies': frequencies,
        'spectrum': spectrum,
        'fault_frequencies': fault_frequencies,
        'cor_scores': cor_scores
    }

def load_and_process_t_data(csv_path, num_samples=1984, rpm=600):
    """
    Load new CSV data format and process samples
    """
    print("Loading data...")
    # Read data
    df = pd.read_csv(csv_path, nrows=num_samples)

    results = []

    for i in range(min(num_samples, len(df))):
        # Extract signal data (window_0 to window_X)
        signal_columns = [col for col in df.columns if col.startswith('window_')]
        signal_data = df.iloc[i][signal_columns].values.astype(float)

        # Extract metadata
        sample_index = df.iloc[i]['index']

        print(f"Processing sample {i+1}/{num_samples}: Index={sample_index}")

        # Process signal with fixed RPM=600
        result = process_signal_sample(signal_data, rpm)
        result['sample_info'] = {
            'index': sample_index,
            'rpm': rpm
        }
        # Store original signal for plotting
        result['original_signal'] = signal_data

        results.append(result)

    return results

def analyze_cor_patterns_t_data(results):
    """
    Analyze COR patterns across different samples (modified for t_data)
    """
    cor_data = []

    for result in results:
        sample_info = result['sample_info']
        cor_scores = result['cor_scores']

        row = {
            'sample_index': sample_info['index'],
            'rpm': sample_info['rpm']
        }
        row.update(cor_scores)
        cor_data.append(row)

    cor_df = pd.DataFrame(cor_data)

    print("=== COR Pattern Analysis (t_data) ===")
    print(f"Processed {len(cor_df)} samples")
    print(f"RPM: {cor_df['rpm'].iloc[0]}")  # All should be 600
    print()

    # Statistical summary of COR scores
    print("COR Score Statistics:")
    fault_types = ['BPFO', 'BPFI', 'BSF', 'FTF']
    for fault_type in fault_types:
        mean_cor = cor_df[fault_type].mean()
        std_cor = cor_df[fault_type].std()
        max_cor = cor_df[fault_type].max()
        min_cor = cor_df[fault_type].min()

        print(f"{fault_type}:")
        print(f"  Mean: {mean_cor:.4f} ± {std_cor:.4f}")
        print(f"  Range: {min_cor:.4f} - {max_cor:.4f}")

    return cor_df

def calculate_theoretical_fault_frequencies(rpm=600):
    """
    Calculate theoretical fault frequencies for RPM=600
    """
    print("=== Theoretical Fault Frequencies (RPM=600) ===")

    fault_freqs = calculate_skf6205_frequencies(rpm)
    print(f"RPM = {rpm}:")
    for fault_type, freq in fault_freqs.items():
        print(f"  {fault_type}: {freq:.2f} Hz")
    print()

    return fault_freqs

def identify_top_cor_samples(cor_df):
    """
    Identify samples with highest COR scores for each fault type
    """
    print("=== Top Samples by COR Score ===")
    fault_types = ['BPFO', 'BPFI', 'BSF', 'FTF']

    for fault_type in fault_types:
        top_sample = cor_df.loc[cor_df[fault_type].idxmax()]
        print(f"{fault_type} - Highest COR Score:")
        print(f"  Sample {top_sample['sample_index']}: {top_sample[fault_type]:.4f}")
        print(f"  RPM: {top_sample['rpm']}")
        print()

def plot_sample_results(results, sample_idx=0):
    """
    Plot processing results for visualization (simplified version)
    """
    result = results[sample_idx]
    sample_info = result['sample_info']

    print(f"Plotting sample {sample_info['index']} (RPM: {sample_info['rpm']})")

    # Create a simple plot showing COR scores
    plt.figure(figsize=(10, 6))

    cor_scores = result['cor_scores']
    fault_names = list(cor_scores.keys())
    cor_values = list(cor_scores.values())

    bars = plt.bar(fault_names, cor_values, color=['#237B9F', '#FEE066', '#EC817E', '#AD0B08'], alpha=0.85)
    plt.title(f'COR Index Scores - Sample {sample_info["index"]} (RPM: {sample_info["rpm"]})', fontweight='bold')
    plt.ylabel('COR Index')
    plt.xlabel('Fault Type')

    # Add value labels on bars
    for bar, value in zip(bars, cor_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f't_data_cor_sample_{sample_idx}_simple.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: t_data_cor_sample_{sample_idx}_simple.png")

def main():
    """
    Generate comprehensive analysis summary for t_data
    """
    csv_path = "/Users/matthew/Workspace/HuaweiCup-E/data/t_data_分割后样本.csv"

    # Process all 1984 samples
    print("Loading and processing t_data...")
    results = load_and_process_t_data(csv_path, num_samples=1984, rpm=600)

    # Calculate theoretical fault frequencies
    calculate_theoretical_fault_frequencies(rpm=600)

    # Analyze COR patterns
    cor_df = analyze_cor_patterns_t_data(results)

    # Identify top samples
    identify_top_cor_samples(cor_df)

    # Plot a few samples for visualization
    print("Generating plots for sample analysis...")
    plot_sample_results(results, sample_idx=0)
    plot_sample_results(results, sample_idx=500)
    plot_sample_results(results, sample_idx=1000)
    plot_sample_results(results, sample_idx=1500)

    # Save COR data to CSV for further analysis
    cor_df.to_csv('t_data_cor_analysis_results.csv', index=False)
    print("COR analysis results saved to 't_data_cor_analysis_results.csv'")

    return cor_df

if __name__ == "__main__":
    cor_df = main()