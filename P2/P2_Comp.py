import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import fft2
import pandas as pd
from scipy.stats import entropy as calculate_entropy

# Directory containing the frames
directory = 'part2_given'  # Update this with the actual path

# List all files in the directory and sort them
files = sorted([f for f in os.listdir(directory) if f.startswith('frame')])

# Function to interpolate between two images
def interpolate_images(img1, img2):
    return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# Function to calculate the SSIM difference
def calculate_ssim_difference(img1, img2):
    return 1 - ssim(img1, img2)

# Function to calculate the MSE difference
def calculate_mse_difference(img1, img2):
    return np.mean((img1 - img2) ** 2)

# Function to calculate the entropy difference
def calculate_entropy_difference(img1, img2):
    hist1, _ = np.histogram(img1, bins=256, range=(0, 256), density=True)
    hist2, _ = np.histogram(img2, bins=256, range=(0, 256), density=True)
    return np.abs(calculate_entropy(hist1) - calculate_entropy(hist2))

# Function to calculate the Fourier difference
def calculate_fourier_difference(img1, img2):
    fft1 = np.abs(fft2(img1))
    fft2_img2 = np.abs(fft2(img2))  # Corrected name to avoid conflict
    return np.mean(np.abs(fft1 - fft2_img2))

# Function to calculate NCC (Normalized Cross-Correlation)
def calculate_ncc_difference(img1, img2):
    return np.mean((img1 - img1.mean()) * (img2 - img2.mean())) / (img1.std() * img2.std())

# Function to calculate PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr_difference(img1, img2):
    mse = calculate_mse_difference(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

# Containers for results
ssim_differences = []
mse_differences = []
entropy_differences = []
fourier_differences = []
ncc_differences = []
psnr_differences = []

# Iterate through the frames, starting from the second frame and ending at the second to last frame
for i in range(1, len(files) - 1):
    prev_image = cv2.imread(os.path.join(directory, files[i-1]), cv2.IMREAD_GRAYSCALE)
    next_image = cv2.imread(os.path.join(directory, files[i+1]), cv2.IMREAD_GRAYSCALE)
    current_image = cv2.imread(os.path.join(directory, files[i]), cv2.IMREAD_GRAYSCALE)
    
    # Interpolate between the previous and next images
    interpolated_image = interpolate_images(prev_image, next_image)
    
    # Calculate differences using various methods
    ssim_diff = calculate_ssim_difference(current_image, interpolated_image)
    mse_diff = calculate_mse_difference(current_image, interpolated_image)
    entropy_diff = calculate_entropy_difference(current_image, interpolated_image)
    fourier_diff = calculate_fourier_difference(current_image, interpolated_image)
    ncc_diff = calculate_ncc_difference(current_image, interpolated_image)
    psnr_diff = calculate_psnr_difference(current_image, interpolated_image)
    
    # Store the results
    ssim_differences.append((i, ssim_diff))
    mse_differences.append((i, mse_diff))
    entropy_differences.append((i, entropy_diff))
    fourier_differences.append((i, fourier_diff))
    ncc_differences.append((i, ncc_diff))
    psnr_differences.append((i, psnr_diff))

# Convert the differences to DataFrames
df_ssim = pd.DataFrame(ssim_differences, columns=['Frame Index', 'SSIM Difference'])
df_mse = pd.DataFrame(mse_differences, columns=['Frame Index', 'MSE Difference'])
df_entropy = pd.DataFrame(entropy_differences, columns=['Frame Index', 'Entropy Difference'])
df_fourier = pd.DataFrame(fourier_differences, columns=['Frame Index', 'Fourier Difference'])
df_ncc = pd.DataFrame(ncc_differences, columns=['Frame Index', 'NCC Difference'])
df_psnr = pd.DataFrame(psnr_differences, columns=['Frame Index', 'PSNR Difference'])

# Get the top 75 most different frames for each method
df_top_75_ssim = df_ssim.sort_values(by='SSIM Difference', ascending=False).head(75)
df_top_75_mse = df_mse.sort_values(by='MSE Difference', ascending=False).head(75)
df_top_75_entropy = df_entropy.sort_values(by='Entropy Difference', ascending=False).head(75)
df_top_75_fourier = df_fourier.sort_values(by='Fourier Difference', ascending=False).head(75)
df_top_75_ncc = df_ncc.sort_values(by='NCC Difference', ascending=False).head(75)
df_top_75_psnr = df_psnr.sort_values(by='PSNR Difference', ascending=False).head(75)

# Merge the results for comparison
df_combined = pd.DataFrame({
    'Frame Index SSIM': df_top_75_ssim['Frame Index'].values,
    'SSIM Difference': df_top_75_ssim['SSIM Difference'].values,
    'Frame Index MSE': df_top_75_mse['Frame Index'].values,
    'MSE Difference': df_top_75_mse['MSE Difference'].values,
    'Frame Index Entropy': df_top_75_entropy['Frame Index'].values,
    'Entropy Difference': df_top_75_entropy['Entropy Difference'].values,
    'Frame Index Fourier': df_top_75_fourier['Frame Index'].values,
    'Fourier Difference': df_top_75_fourier['Fourier Difference'].values,
    'Frame Index NCC': df_top_75_ncc['Frame Index'].values,
    'NCC Difference': df_top_75_ncc['NCC Difference'].values,
    'Frame Index PSNR': df_top_75_psnr['Frame Index'].values,
    'PSNR Difference': df_top_75_psnr['PSNR Difference'].values
})

# Save the combined results to a single Excel sheet
output_file_path = 'top_75_most_different_frames_all_methods.xlsx'
df_combined.to_excel(output_file_path, index=False)

print(f"The top 75 most different frames for all methods have been saved to {output_file_path}")
