# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:55:45 2025

@author: Bantisouli Agathi
"""
# Libraries
import numpy as np
from scipy.linalg import svd, rq
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Step 1: Import data ----

image_points = np.loadtxt(os.path.join(BASE_DIR, "image_points_pix.txt"), delimiter=",")
object_points = np.loadtxt(os.path.join(BASE_DIR, "object_points_m.txt"), delimiter=",")

# Check if dimensions are 2x18 (image points) and 3x18 (object points) → OK
assert image_points.shape[0] == 2 and object_points.shape[0] == 3, "Incorrect dimensions!"

# ---- Step 2: Create the system Ap = 0 for DLT ----

num_points = object_points.shape[1]
A = []

for i in range(num_points):
    X, Y, Z = object_points[:, i]
    x, y = image_points[:, i]
    # Each point contributes two rows in matrix A based on the DLT formulation
    A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
    A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])

A = np.array(A)
#print("A = ", A)

# ---- Step 3: Solve Ap=0 using SVD ----

U, D, Vt = svd(A)
# Solution is the last row of V^T reshaped into 3x4 projection matrix P
P = Vt[-1].reshape(3, 4)

# ---- Step 4: Compute reprojections and calculate reprojection error ----

# Convert object points to homogeneous coordinates (4xN)
object_points_h = np.vstack((object_points, np.ones((1, num_points))))
# Project 3D points into 2D using projection matrix P (3xN)
projected = P @ object_points_h
# Normalize by the third coordinate to get pixel coordinates
projected /= projected[2]

# Calculate reprojection error between original and projected image points
reprojection_error = image_points - projected[:2]
squared_errors = np.sum(reprojection_error**2, axis=0)

# Degrees of freedom: number of observations minus number of parameters (36-11=25)
r = 36 - 11
# Calculate standard deviation of reprojection error (sigma_0)
sigma_o = np.sqrt(np.sum(squared_errors) / r)

# ---- Step 5: Decompose projection matrix P into intrinsic matrix K, rotation matrix R, and camera center X0 ----

# P is defined only up to scale (including sign). Choose the sign so that
# det(B) > 0, which guarantees a proper rotation matrix (det(R) = +1).
if np.linalg.det(P[:, :3]) < 0:
    P = -P

B = P[:, :3]  # Extract left 3x3 submatrix (contains intrinsic and rotation)
p4 = P[:, 3]  # Extract last column vector (translation component)
K, R = rq(B)  # Perform RQ decomposition on B to get intrinsic matrix K and rotation matrix R

# RQ decomposition is unique only up to the signs of the diagonal of K.
# Force a positive diagonal (positive focal lengths): K -> K T, R -> T R,
# where T = diag(sign(K_ii)) and T @ T = I, so K @ R is unchanged.
T = np.diag(np.sign(np.diag(K)))
K = K @ T
R = T @ R

# Normalize K so that K[2,2] = 1
K = K / K[2, 2]

# Compute camera center X0 in world coordinates using formula: X0 = -inv(B) * p4
X0 = -np.linalg.inv(B) @ p4

# ---- Step 6: Print results ----

def extract_intrinsics_and_angles(K, R):
    import cv2

    # Extract intrinsic parameters from K
    c = K[0, 0]  # focal length 
    m = (K[1, 1] / c) - 1  # relative scale deviation
    skew = K[0, 1] / c  # skew coefficient
    x0 = K[0, 2]  # principal point x
    y0 = K[1, 2]  # principal point y

    print("Intrinsic parameters:")
    print(f"  Focal length c: {c:.4f}")
    print(f"  Relative scale deviation m: {m:.4f}")
    print(f"  Skew (deviation from orthogonality): {skew:.6f}")
    print(f"  Principal point (x0, y0): ({x0:.4f}, {y0:.4f})")
    print()

    # Extract rotation angles (roll, pitch, yaw) in degrees using OpenCV's RQ decomposition
    angles = cv2.RQDecomp3x3(R)[0]
    roll, pitch, yaw = angles

    print("Rotation angles (degrees):")
    print(f"  ω (roll)  : {roll:.4f}")
    print(f"  φ (pitch) : {pitch:.4f}")
    print(f"  κ (yaw)   : {yaw:.4f}")

# Print main results
print("Projection matrix P:\n", P)
print("\nMean reprojection error σ₀: {:.4f} pixels".format(sigma_o))
print("\nCamera intrinsic matrix K:\n", K)
print("\nRotation matrix R:\n", R)
print("\nCamera center (X0):\n", X0)

# Call function to print intrinsic parameters and rotation angles
extract_intrinsics_and_angles(K, R)