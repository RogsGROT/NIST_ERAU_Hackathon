import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import correlate2d
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from skimage.filters import sobel, scharr, prewitt
from pywt import dwt2, idwt2  # Corrected import from PyWavelets


# Paths to the folders
given_folder_path = r'C:\Users\Roger\Downloads\NIST\opening_release_part1\opening_release_part1\part1_given'
missing_folder_path = r'C:\Users\Roger\Downloads\NIST\opening_release_part1\opening_release_part1\part1_missing'

# Get list of files in each folder
given_files = sorted(os.listdir(given_folder_path))
missing_files = sorted(os.listdir(missing_folder_path))  # Process only the first 5 missing files

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# New methods

def phase_correlation(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross_power_spectrum = (f1 * f2.conj()) / np.abs(f1 * f2.conj())
    correlation = np.fft.ifft2(cross_power_spectrum)
    correlation = np.fft.fftshift(correlation)
    max_correlation = np.max(np.abs(correlation))
    return max_correlation

def normalized_cross_correlation(img1, img2):
    return cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED).max()

def image_subtraction(img1, img2):
    diff = cv2.absdiff(img1, img2)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    return np.sum(thresh)

def gradient_difference(img1, img2, method='sobel'):
    if method == 'sobel':
        grad1 = sobel(img1)
        grad2 = sobel(img2)
    elif method == 'scharr':
        grad1 = scharr(img1)
        grad2 = scharr(img2)
    elif method == 'prewitt':
        grad1 = prewitt(img1)
        grad2 = prewitt(img2)
    
    grad_diff = np.abs(grad1 - grad2)
    return np.sum(grad_diff)

def optical_flow(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.sum(mag)

def lbp_comparison(img1, img2, radius=1, n_points=8):
    lbp1 = local_binary_pattern(img1, n_points, radius, method="uniform")
    lbp2 = local_binary_pattern(img2, n_points, radius, method="uniform")
    return np.sum(np.abs(lbp1 - lbp2))

def wavelet_difference(img1, img2):
    coeffs1 = dwt2(img1, 'haar')
    coeffs2 = dwt2(img2, 'haar')
    diff = np.sum(np.abs(coeffs1[0] - coeffs2[0]))  # Compare only the approximation coefficients
    return diff

def find_top_positions(given_files, missing_image, method):
    scores = []
    
    for i, given_file in enumerate(given_files[:-1]):
        given_image = load_image(os.path.join(given_folder_path, given_file))
        
        if method == 'phase_correlation':
            score = phase_correlation(given_image, missing_image)
        elif method == 'ncc':
            score = normalized_cross_correlation(given_image, missing_image)
        elif method == 'image_subtraction':
            score = image_subtraction(given_image, missing_image)
        elif method == 'sobel':
            score = gradient_difference(given_image, missing_image, method='sobel')
        elif method == 'scharr':
            score = gradient_difference(given_image, missing_image, method='scharr')
        elif method == 'prewitt':
            score = gradient_difference(given_image, missing_image, method='prewitt')
        elif method == 'optical_flow':
            score = optical_flow(given_image, missing_image)
        elif method == 'lbp':
            score = lbp_comparison(given_image, missing_image)
        elif method == 'wavelet':
            score = wavelet_difference(given_image, missing_image)
        
        scores.append((score, i))
    
    if method in ['phase_correlation', 'ncc']:
        scores.sort(reverse=True, key=lambda x: x[0])
    else:
        scores.sort(key=lambda x: x[0])
    
    top_positions = scores[:3]
    return top_positions

def refine_position_decision(given_files, candidate_positions, missing_image, method):
    refined_positions = []
    
    for score, position in candidate_positions:
        if position > 0:
            prev_image = load_image(os.path.join(given_folder_path, given_files[position - 1]))
        else:
            prev_image = None
        
        if position < len(given_files) - 1:
            next_image = load_image(os.path.join(given_folder_path, given_files[position + 1]))
        else:
            next_image = None
        
        if prev_image is not None:
            if method == 'phase_correlation':
                prev_similarity = phase_correlation(prev_image, missing_image)
            elif method == 'ncc':
                prev_similarity = normalized_cross_correlation(prev_image, missing_image)
            elif method == 'image_subtraction':
                prev_similarity = image_subtraction(prev_image, missing_image)
            elif method in ['sobel', 'scharr', 'prewitt']:
                prev_similarity = gradient_difference(prev_image, missing_image, method=method)
            elif method == 'optical_flow':
                prev_similarity = optical_flow(prev_image, missing_image)
            elif method == 'lbp':
                prev_similarity = lbp_comparison(prev_image, missing_image)
            elif method == 'wavelet':
                prev_similarity = wavelet_difference(prev_image, missing_image)
        else:
            prev_similarity = float('-inf')
        
        if next_image is not None:
            if method == 'phase_correlation':
                next_similarity = phase_correlation(next_image, missing_image)
            elif method == 'ncc':
                next_similarity = normalized_cross_correlation(next_image, missing_image)
            elif method == 'image_subtraction':
                next_similarity = image_subtraction(next_image, missing_image)
            elif method in ['sobel', 'scharr', 'prewitt']:
                next_similarity = gradient_difference(next_image, missing_image, method=method)
            elif method == 'optical_flow':
                next_similarity = optical_flow(next_image, missing_image)
            elif method == 'lbp':
                next_similarity = lbp_comparison(next_image, missing_image)
            elif method == 'wavelet':
                next_similarity = wavelet_difference(next_image, missing_image)
        else:
            next_similarity = float('-inf')
        
        if prev_similarity > next_similarity:
            refined_position = position
        else:
            refined_position = position + 1
        
        refined_positions.append((refined_position, score))
    
    return refined_positions

results = []

for missing_file in missing_files:
    missing_image = load_image(os.path.join(missing_folder_path, missing_file))
    
    # Replace 'histogram', 'cross_correlation', etc. with new methods
    phase_corr_candidates = find_top_positions(given_files, missing_image, method='phase_correlation')
    phase_corr_final_positions = refine_position_decision(given_files, phase_corr_candidates, missing_image, method='phase_correlation')
    
    ncc_candidates = find_top_positions(given_files, missing_image, method='ncc')
    ncc_final_positions = refine_position_decision(given_files, ncc_candidates, missing_image, method='ncc')
    
    img_sub_candidates = find_top_positions(given_files, missing_image, method='image_subtraction')
    img_sub_final_positions = refine_position_decision(given_files, img_sub_candidates, missing_image, method='image_subtraction')
    
    sobel_candidates = find_top_positions(given_files, missing_image, method='sobel')
    sobel_final_positions = refine_position_decision(given_files, sobel_candidates, missing_image, method='sobel')
    
    scharr_candidates = find_top_positions(given_files, missing_image, method='scharr')
    scharr_final_positions = refine_position_decision(given_files, scharr_candidates, missing_image, method='scharr')
    
    prewitt_candidates = find_top_positions(given_files, missing_image, method='prewitt')
    prewitt_final_positions = refine_position_decision(given_files, prewitt_candidates, missing_image, method='prewitt')
    
    optical_flow_candidates = find_top_positions(given_files, missing_image, method='optical_flow')
    optical_flow_final_positions = refine_position_decision(given_files, optical_flow_candidates, missing_image, method='optical_flow')
    
    lbp_candidates = find_top_positions(given_files, missing_image, method='lbp')
    lbp_final_positions = refine_position_decision(given_files, lbp_candidates, missing_image, method='lbp')
    
    wavelet_candidates = find_top_positions(given_files, missing_image, method='wavelet')
    wavelet_final_positions = refine_position_decision(given_files, wavelet_candidates, missing_image, method='wavelet')
    
    results.append({
        "Frame": missing_file,
        "Phase Corr Top 1 Pos": phase_corr_final_positions[0][0],
        "Phase Corr Top 1 Scr": phase_corr_final_positions[0][1],
        "Phase Corr Top 2 Pos": phase_corr_final_positions[1][0],
        "Phase Corr Top 2 Scr": phase_corr_final_positions[1][1],
        "Phase Corr Top 3 Pos": phase_corr_final_positions[2][0],
        "Phase Corr Top 3 Scr": phase_corr_final_positions[2][1],
        "NCC Top 1 Pos": ncc_final_positions[0][0],
        "NCC Top 1 Scr": ncc_final_positions[0][1],
        "NCC Top 2 Pos": ncc_final_positions[1][0],
        "NCC Top 2 Scr": ncc_final_positions[1][1],
        "NCC Top 3 Pos": ncc_final_positions[2][0],
        "NCC Top 3 Scr": ncc_final_positions[2][1],
        "Img Sub Top 1 Pos": img_sub_final_positions[0][0],
        "Img Sub Top 1 Scr": img_sub_final_positions[0][1],
        "Img Sub Top 2 Pos": img_sub_final_positions[1][0],
        "Img Sub Top 2 Scr": img_sub_final_positions[1][1],
        "Img Sub Top 3 Pos": img_sub_final_positions[2][0],
        "Img Sub Top 3 Scr": img_sub_final_positions[2][1],
        "Sobel Top 1 Pos": sobel_final_positions[0][0],
        "Sobel Top 1 Scr": sobel_final_positions[0][1],
        "Sobel Top 2 Pos": sobel_final_positions[1][0],
        "Sobel Top 2 Scr": sobel_final_positions[1][1],
        "Sobel Top 3 Pos": sobel_final_positions[2][0],
        "Sobel Top 3 Scr": sobel_final_positions[2][1],
        "Scharr Top 1 Pos": scharr_final_positions[0][0],
        "Scharr Top 1 Scr": scharr_final_positions[0][1],
        "Scharr Top 2 Pos": scharr_final_positions[1][0],
        "Scharr Top 2 Scr": scharr_final_positions[1][1],
        "Scharr Top 3 Pos": scharr_final_positions[2][0],
        "Scharr Top 3 Scr": scharr_final_positions[2][1],
        "Prewitt Top 1 Pos": prewitt_final_positions[0][0],
        "Prewitt Top 1 Scr": prewitt_final_positions[0][1],
        "Prewitt Top 2 Pos": prewitt_final_positions[1][0],
        "Prewitt Top 2 Scr": prewitt_final_positions[1][1],
        "Prewitt Top 3 Pos": prewitt_final_positions[2][0],
        "Prewitt Top 3 Scr": prewitt_final_positions[2][1],
        "Optical Flow Top 1 Pos": optical_flow_final_positions[0][0],
        "Optical Flow Top 1 Scr": optical_flow_final_positions[0][1],
        "Optical Flow Top 2 Pos": optical_flow_final_positions[1][0],
        "Optical Flow Top 2 Scr": optical_flow_final_positions[1][1],
        "Optical Flow Top 3 Pos": optical_flow_final_positions[2][0],
        "Optical Flow Top 3 Scr": optical_flow_final_positions[2][1],
        "LBP Top 1 Pos": lbp_final_positions[0][0],
        "LBP Top 1 Scr": lbp_final_positions[0][1],
        "LBP Top 2 Pos": lbp_final_positions[1][0],
        "LBP Top 2 Scr": lbp_final_positions[1][1],
        "LBP Top 3 Pos": lbp_final_positions[2][0],
        "LBP Top 3 Scr": lbp_final_positions[2][1],
        "Wavelet Top 1 Pos": wavelet_final_positions[0][0],
        "Wavelet Top 1 Scr": wavelet_final_positions[0][1],
        "Wavelet Top 2 Pos": wavelet_final_positions[1][0],
        "Wavelet Top 2 Scr": wavelet_final_positions[1][1],
        "Wavelet Top 3 Pos": wavelet_final_positions[2][0],
        "Wavelet Top 3 Scr": wavelet_final_positions[2][1],
    })

# Create a DataFrame from the results with shortened column names
df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
output_path = "final_frame_positions_advanced_methods_2.xlsx"
df.to_excel(output_path, index=False)

output_path
