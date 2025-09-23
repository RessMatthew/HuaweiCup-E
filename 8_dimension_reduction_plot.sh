# COR - PCA
python 7_dimension_reduction_plot.py --method pca \
--columns COR_BPFO COR_BPFI COR_BSF COR_FTF

# COR - t-SNE
python 7_dimension_reduction_plot.py --method tsne --only-2d \
--columns COR_BPFO COR_BPFI COR_BSF COR_FTF

# COR - UMAP
python 7_dimension_reduction_plot.py --method umap \
--columns COR_BPFO COR_BPFI COR_BSF COR_FTF

# x=4 - t-SNE
python 7_dimension_reduction_plot.py --method tsne --only-2d \
--columns COR_BPFO COR_BPFI COR_BSF high_freq_energy_ratio var rms peak_to_peak low_freq_energy_ratio

# x=3 - t-SNE
python 7_dimension_reduction_plot.py --method tsne --only-2d \
--columns COR_BPFO COR_BPFI COR_BSF high_freq_energy_ratio var rms peak_to_peak low_freq_energy_ratio std peak mid_freq_energy_ratio

# x=2 - t-SNE
python 7_dimension_reduction_plot.py --method tsne --only-2d \
--columns COR_BPFO COR_BPFI COR_BSF high_freq_energy_ratio var rms peak_to_peak low_freq_energy_ratio std peak mid_freq_energy_ratio peak_frequency

# x=1 - t-SNE
python 7_dimension_reduction_plot.py --method tsne --only-2d \
--columns COR_BPFO COR_BPFI COR_BSF high_freq_energy_ratio var rms peak_to_peak low_freq_energy_ratio std peak mid_freq_energy_ratio peak_frequency peak_amplitude skewness shape_factor

# x=4 - PCA
python 7_dimension_reduction_plot.py --method pca \
--columns COR_BPFO COR_BPFI COR_BSF COR_FTF high_freq_energy_ratio var rms peak_to_peak low_freq_energy_ratio

# x=3 - PCA
python 7_dimension_reduction_plot.py --method pca \
--columns COR_BPFO COR_BPFI COR_BSF COR_FTF high_freq_energy_ratio var rms peak_to_peak low_freq_energy_ratio std peak mid_freq_energy_ratio

# x=2 - PCA
python 7_dimension_reduction_plot.py --method pca \
--columns COR_BPFO COR_BPFI COR_BSF COR_FTF high_freq_energy_ratio var rms peak_to_peak low_freq_energy_ratio std peak mid_freq_energy_ratio peak_frequency

# x=1 - PCA
python 7_dimension_reduction_plot.py --method pca \
--columns COR_BPFO COR_BPFI COR_BSF COR_FTF high_freq_energy_ratio var rms peak_to_peak low_freq_energy_ratio std peak mid_freq_energy_ratio peak_frequency peak_amplitude skewness shape_factor
