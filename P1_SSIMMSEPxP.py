import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Paths to the folders
given_folder_path = r'C:\Users\Roger\Downloads\NIST\opening_release_part1\opening_release_part1\part1_given'
missing_folder_path = r'C:\Users\Roger\Downloads\NIST\opening_release_part1\opening_release_part1\part1_missing'

# Get list of files in each folder
given_files = sorted(os.listdir(given_folder_path))
missing_files = sorted(os.listdir(missing_folder_path))

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def compare_images_ssim(img1, img2):
    return ssim(img1, img2, full=True)[0]

def compare_images_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def compare_images_pixel_by_pixel(img1, img2):
    return np.sum(img1 != img2)

def find_top_positions(given_files, missing_image, method):
    scores = []
    
    for i, given_file in enumerate(given_files[:-1]):
        given_image = load_image(os.path.join(given_folder_path, given_file))
        
        if method == 'ssim':
            score = compare_images_ssim(given_image, missing_image)
        elif method == 'mse':
            score = compare_images_mse(given_image, missing_image)
        elif method == 'pixel_by_pixel':
            score = compare_images_pixel_by_pixel(given_image, missing_image)
        
        scores.append((score, i))
    
    # Sort scores (SSIM descending, MSE and Pixel ascending)
    if method == 'ssim':
        scores.sort(reverse=True, key=lambda x: x[0])
    else:
        scores.sort(key=lambda x: x[0])
    
    # Get top 3 positions with scores
    top_positions = scores[:3]
    return top_positions

def refine_position_decision(given_files, candidate_positions, missing_image, method):
    refined_positions = []
    
    for score, position in candidate_positions:
        # Compare with the frames just before and after the best match
        if position > 0:
            prev_image = load_image(os.path.join(given_folder_path, given_files[position - 1]))
        else:
            prev_image = None
        
        if position < len(given_files) - 1:
            next_image = load_image(os.path.join(given_folder_path, given_files[position + 1]))
        else:
            next_image = None
        
        if prev_image is not None:
            if method == 'ssim':
                prev_similarity = compare_images_ssim(prev_image, missing_image)
            elif method == 'mse':
                prev_similarity = compare_images_mse(prev_image, missing_image)
            elif method == 'pixel_by_pixel':
                prev_similarity = compare_images_pixel_by_pixel(prev_image, missing_image)
        else:
            prev_similarity = float('-inf')
            
        if next_image is not None:
            if method == 'ssim':
                next_similarity = compare_images_ssim(next_image, missing_image)
            elif method == 'mse':
                next_similarity = compare_images_mse(next_image, missing_image)
            elif method == 'pixel_by_pixel':
                next_similarity = compare_images_pixel_by_pixel(next_image, missing_image)
        else:
            next_similarity = float('-inf')
        
        # Determine if the missing image fits better before or after the best match
        if prev_similarity > next_similarity:
            refined_position = position  # Place before the matched frame
        else:
            refined_position = position + 1  # Place after the matched frame
        
        refined_positions.append((refined_position, score))
    
    return refined_positions

# Prepare to store results
results = []

# Process each missing frame
for missing_file in missing_files:
    missing_image = load_image(os.path.join(missing_folder_path, missing_file))
    
    ssim_candidates = find_top_positions(given_files, missing_image, method='ssim')
    ssim_final_positions = refine_position_decision(given_files, ssim_candidates, missing_image, method='ssim')
    
    mse_candidates = find_top_positions(given_files, missing_image, method='mse')
    mse_final_positions = refine_position_decision(given_files, mse_candidates, missing_image, method='mse')
    
    pixel_candidates = find_top_positions(given_files, missing_image, method='pixel_by_pixel')
    pixel_final_positions = refine_position_decision(given_files, pixel_candidates, missing_image, method='pixel_by_pixel')
    
    results.append({
        "Frame": missing_file,
        "SSIM Top 1 Position": ssim_final_positions[0][0],
        "SSIM Top 1 Score": ssim_final_positions[0][1],
        "SSIM Top 2 Position": ssim_final_positions[1][0],
        "SSIM Top 2 Score": ssim_final_positions[1][1],
        "SSIM Top 3 Position": ssim_final_positions[2][0],
        "SSIM Top 3 Score": ssim_final_positions[2][1],
        "MSE Top 1 Position": mse_final_positions[0][0],
        "MSE Top 1 Score": mse_final_positions[0][1],
        "MSE Top 2 Position": mse_final_positions[1][0],
        "MSE Top 2 Score": mse_final_positions[1][1],
        "MSE Top 3 Position": mse_final_positions[2][0],
        "MSE Top 3 Score": mse_final_positions[2][1],
        "Pixel-by-Pixel Top 1 Position": pixel_final_positions[0][0],
        "Pixel-by-Pixel Top 1 Score": pixel_final_positions[0][1],
        "Pixel-by-Pixel Top 2 Position": pixel_final_positions[1][0],
        "Pixel-by-Pixel Top 2 Score": pixel_final_positions[1][1],
        "Pixel-by-Pixel Top 3 Position": pixel_final_positions[2][0],
        "Pixel-by-Pixel Top 3 Score": pixel_final_positions[2][1]
    })

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
output_path = "frame_positions.xlsx"
df.to_excel(output_path, index=False)

output_path
