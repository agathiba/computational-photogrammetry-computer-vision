# Import Libraries
import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pixel coordinates
image_points = np.array([
    [795, 1581],
    [1386, 1411],
    [1340, 647],
    [745, 481],
    [776, 1024],
    [1365, 1024],
    [1240, 815],
    [1261, 1237],
    [1123, 1026]
], dtype=np.float32)

# Real-world coordinates (in cm) 
object_points = np.array([
    [0, 0],
    [40, 0],
    [40, 40],
    [0, 40],
    [0, 20],
    [40, 20],
    [30, 30],
    [30, 10],
    [20, 20]
], dtype=np.float32)

# Homography H
H, _ = cv2.findHomography(image_points, object_points)

# Inverse homography H⁻¹
H_inv = np.linalg.inv(H)

print("H =\n", H)
print("\nH_inv =\n", H_inv)

# Project image points
image_points_h = np.hstack((image_points, np.ones((image_points.shape[0], 1))))  # Add homogeneous coordinate
projected_h = (H @ image_points_h.T).T  # Apply H
projected = projected_h[:, :2] / projected_h[:, [2]]  # Convert back to non-homogeneous

# Reprojection errors
errors = object_points - projected
errors_norm = np.linalg.norm(errors, axis=1)
rmse = np.sqrt(np.mean(errors_norm**2))

for i, (real, proj, err) in enumerate(zip(object_points, projected, errors_norm)):
    print(f"Point {i+1}: Real={real}, Projected={proj.round(2)}, Error={err:.2f} cm")

print(f"\nRMSE = {rmse:.2f} cm")

# Define scaling factor and margin 
scale = 12.5            
margin_cm = 1           
margin_px = int(margin_cm * scale)

# Shift the projected content
H_translate = np.array([
    [1, 0, margin_px],  
    [0, 1, margin_px],  
    [0, 0, 1]
])

# Convert from cm to pixels
H_scale = np.array([
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1]
])

# Total homography
H_total = H_scale @ H_translate @ H

# Define canvas size in pixels 
canvas_cm = 40 + 2 * margin_cm              
output_size = (int(canvas_cm * scale) + 200, int(canvas_cm * scale) + 200)  

# Load image 
img = cv2.imread(os.path.join(BASE_DIR, "chessboard.jpg"))

# Apply homography
warped = cv2.warpPerspective(img, H_total, output_size)
cv2.imwrite(os.path.join(RESULTS_DIR, "warped_output.jpg"), warped)

# Draw points and labels
for idx, point in enumerate(image_points):
    x, y = int(point[0]), int(point[1])
    cv2.circle(img, (x, y), 10, (0, 0, 255), -1)  # Red dot
cv2.imwrite(os.path.join(RESULTS_DIR, "image_with_points.jpg"), img)