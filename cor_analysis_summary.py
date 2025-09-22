import pandas as pd
import numpy as np
from bearing_fault_analysis import load_and_process_data, calculate_skf6205_frequencies

def analyze_cor_patterns(results):
    """
    Analyze COR patterns across different samples
    """
    cor_data = []
    
    for result in results:
        sample_info = result['sample_info']
        cor_scores = result['cor_scores']
        
        row = {
            'sample_index': sample_info['index'],
            'rpm': sample_info['rpm'],
            'target_label': sample_info['target_label']
        }
        row.update(cor_scores)
        cor_data.append(row)
    
    cor_df = pd.DataFrame(cor_data)
    
    print("=== COR Pattern Analysis ===")
    print(f"Processed {len(cor_df)} samples")
    print(f"RPM range: {cor_df['rpm'].min()} - {cor_df['rpm'].max()}")
    print(f"Labels: {cor_df['target_label'].unique()}")
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

def calculate_fault_frequencies_summary(rpm_values):
    """
    Calculate theoretical fault frequencies for given RPM values
    """
    print("=== Theoretical Fault Frequencies ===")
    
    for rpm in rpm_values:
        fault_freqs = calculate_skf6205_frequencies(rpm)
        print(f"RPM = {rpm}:")
        for fault_type, freq in fault_freqs.items():
            print(f"  {fault_type}: {freq:.2f} Hz")
        print()

def main():
    """
    Generate comprehensive analysis summary
    """
    csv_path = "/Users/matthew/Workspace/HuaweiCup-E/data/data_分割后_带标签样本.csv"
    
    # Process more samples for better analysis
    print("Loading and processing data...")
    results = load_and_process_data(csv_path, num_samples=50)
    
    # Analyze COR patterns
    cor_df = analyze_cor_patterns(results)
    
    # Get unique RPM values
    unique_rpms = cor_df['rpm'].unique()
    calculate_fault_frequencies_summary(unique_rpms)
    
    # Identify samples with highest COR scores for each fault type
    print("=== Top Samples by COR Score ===")
    fault_types = ['BPFO', 'BPFI', 'BSF', 'FTF']
    
    for fault_type in fault_types:
        top_sample = cor_df.loc[cor_df[fault_type].idxmax()]
        print(f"{fault_type} - Highest COR Score:")
        print(f"  Sample {top_sample['sample_index']}: {top_sample[fault_type]:.4f}")
        print(f"  RPM: {top_sample['rpm']}, Label: {top_sample['target_label']}")
        print()
    
    # Save COR data to CSV for further analysis
    cor_df.to_csv('cor_analysis_results.csv', index=False)
    print("COR analysis results saved to 'cor_analysis_results.csv'")
    
    return cor_df

if __name__ == "__main__":
    cor_df = main()