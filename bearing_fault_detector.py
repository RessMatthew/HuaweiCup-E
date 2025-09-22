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

def load_and_process_data(csv_path, num_samples=10):
    """
    Load CSV data and process first num_samples samples
    """
    print("Loading data...")
    # Read first num_samples + 1 rows (including header)
    df = pd.read_csv(csv_path, nrows=num_samples+1)
    
    results = []
    
    for i in range(min(num_samples, len(df))):
        # Extract signal data (value_0 to value_4095)
        signal_columns = [col for col in df.columns if col.startswith('value_')]
        signal_data = df.iloc[i][signal_columns].values.astype(float)
        
        # Extract metadata
        rpm = df.iloc[i]['rpm']
        target_label = df.iloc[i]['target label']
        sample_index = df.iloc[i]['sample_index']
        
        print(f"Processing sample {i+1}/{num_samples}: RPM={rpm}, Label={target_label}")
        
        # Process signal
        result = process_signal_sample(signal_data, rpm)
        result['sample_info'] = {
            'index': sample_index,
            'rpm': rpm,
            'target_label': target_label
        }
        # Store original signal for plotting
        result['original_signal'] = signal_data
        
        results.append(result)
    
    return results

def plot_results(results, sample_idx=0, original_signal_data=None):
    """
    Plot processing results for visualization with updated color scheme
    """
    result = results[sample_idx]
    sample_info = result['sample_info']
    
    # Specified color scheme with 85% opacity
    main_color = '#237B9F'  # For first 5 plots solid lines
    accent_colors = ['#FEE066D9', '#EC817ED9', '#AD0B08D9', '#71BFB2D9']  # For bars and dashed lines (85% opacity)
    
    # Create 2x3 subplot layout for 6 plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Bearing Fault Analysis - Sample {sample_info['index']} (RPM: {sample_info['rpm']}, Label: {sample_info['target_label']})", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Original signal (before filtering) - if available
    if original_signal_data is not None:
        axes[0,0].plot(original_signal_data, color=main_color, linewidth=1.5)
        axes[0,0].set_title('Original Signal (Before Filtering)', fontweight='bold', color='black')
    else:
        # If no original data, show a placeholder
        axes[0,0].text(0.5, 0.5, 'Original Signal\n(Data Not Available)', 
                       ha='center', va='center', transform=axes[0,0].transAxes, fontsize=12, color=main_color)
        axes[0,0].set_title('Original Signal (Before Filtering)', fontweight='bold', color='black')
    axes[0,0].set_xlabel('Sample')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].grid(True, alpha=0.85)
    
    # Plot 2: Filtered signal (after high-pass)
    axes[0,1].plot(result['filtered_signal'], color=main_color, linewidth=1.5)
    axes[0,1].set_title('Filtered Signal (High-pass)', fontweight='bold', color='black')
    axes[0,1].set_xlabel('Sample')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].grid(True, alpha=0.85)
    
    # Plot 3: Envelope (Hilbert transform)
    axes[0,2].plot(result['envelope'], color=main_color, linewidth=1.5)
    axes[0,2].set_title('Envelope (Hilbert Transform)', fontweight='bold', color='black')
    axes[0,2].set_xlabel('Sample')
    axes[0,2].set_ylabel('Amplitude')
    axes[0,2].grid(True, alpha=0.85)
    
    # Plot 4: Order normalized signal
    axes[1,0].plot(result['order_normalized'], color=main_color, linewidth=1.5)
    axes[1,0].set_title('Order Normalized Signal', fontweight='bold', color='black')
    axes[1,0].set_xlabel('Sample')
    axes[1,0].set_ylabel('Amplitude')
    axes[1,0].grid(True, alpha=0.85)
    
    # Plot 5: Envelope Spectrum (FFT)
    half_len = len(result['frequencies'])//2
    axes[1,1].plot(result['frequencies'][:half_len], result['spectrum'][:half_len], color=main_color, linewidth=1.5)
    axes[1,1].set_title('Envelope Spectrum (FFT)', fontweight='bold', color='black')
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].set_ylabel('Amplitude')
    axes[1,1].grid(True, alpha=0.85)
    
    # Mark fault frequencies on spectrum with accent colors (dashed lines)
    fault_freqs = result['fault_frequencies']
    for i, (name, freq) in enumerate(fault_freqs.items()):
        if freq <= max(result['frequencies'])/2:
            axes[1,1].axvline(x=freq, color=accent_colors[i % len(accent_colors)], linestyle='--', alpha=0.85, linewidth=2, label=f'{name} ({freq:.1f}Hz)')
    axes[1,1].legend(loc='upper right', fontsize=8, framealpha=0.85)
    
    # Plot 6: COR Index Scores with accent colors
    cor_scores = result['cor_scores']
    fault_names = list(cor_scores.keys())
    cor_values = list(cor_scores.values())
    
    bars = axes[1,2].bar(fault_names, cor_values, color=accent_colors[:len(fault_names)], alpha=0.85, edgecolor='black', linewidth=1)
    axes[1,2].set_title('COR Index Scores', fontweight='bold', color='black')
    axes[1,2].set_ylabel('COR Index')
    axes[1,2].tick_params(axis='x', rotation=45)
    axes[1,2].grid(True, alpha=0.85, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, cor_values):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                      f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig(f'bearing_analysis_sample_{sample_idx}_complete_updated.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the analysis
    """
    csv_path = "/Users/matthew/Workspace/HuaweiCup-E/data/data_分割后_带标签样本.csv"
    
    # Process first 10 samples
    results = load_and_process_data(csv_path, num_samples=2000)
    
    # Print summary of COR scores
    print("\n=== COR Index Summary ===")
    for i, result in enumerate(results):
        sample_info = result['sample_info']
        cor_scores = result['cor_scores']
        print(f"Sample {sample_info['index']}: RPM={sample_info['rpm']}, Label={sample_info['target_label']}")
        for fault_type, cor_score in cor_scores.items():
            print(f"  {fault_type}: {cor_score:.4f}")
        print()
    
    # Plot first sample with original signal
    plot_results(results, sample_idx=1999, original_signal_data=results[1999]['original_signal'])
    
    return results

if __name__ == "__main__":
    results = main()