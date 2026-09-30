# Computational Photogrammetry & Computer Vision: Projective Geometry, Camera Calibration & Metric Rectification

Algorithmic implementations in **Python (NumPy, OpenCV)** covering projective geometry, camera matrix decomposition, Direct Linear Transformation (DLT), and planar homography for the postgraduate course *Computer Vision and Photogrammetry*.

---

## 📌 Project Modules

### 1. Affine 2D Reconstruction via Vanishing Points & Line at Infinity
* **Objective:** Projective rectification of planar images to eliminate perspective distortion and restore geometric parallelism.
* **Mathematical Formulation:**
  * Extraction of parallel image edges via cross product of homogeneous boundary coordinates: $m_i = p_a \times p_b$.
  * Computation of vanishing points: $v_m = m_1 \times m_2$ and $v_n = n_1 \times n_2$.
  * Formulation of the vanishing line (line at infinity): $l_\infty = v_m \times v_n$.
  * Construction and application of the projective rectification matrix $H_p$ coupled with dynamic canvas re-mapping to prevent coordinate clipping.
* 📄 **[Technical Report (PDF - Greek)](./HW1_Affine_Reconstruction_Vanishing_Points.pdf)**

---

### 2. Camera Projection Matrix Estimation & Geometric Decomposition (DLT)
* **Objective:** Full camera calibration and pose estimation from corresponding 3D world coordinates ($X, Y, Z$) and 2D image coordinates ($x, y$).
* **Mathematical Workflow:**
  * **Direct Linear Transformation (DLT):** Formulated the homogeneous linear system $Ap = 0$ ($36 \times 12$ coefficient matrix from 18 ground control points).
  * **Singular Value Decomposition (SVD):** Solved for the optimal projection matrix $P$ ($A = U D V^T$, extracting the eigenvector corresponding to the smallest singular value).
  * **Reprojection Error:** Validated spatial accuracy achieving a sub-pixel root mean square reprojection error ($\sigma_0 = 0.46$ px).
  * **Matrix Decomposition (QR):** Decomposed submatrix $B_{3 \times 3}$ via QR factorization to extract:
    * **Camera Intrinsic Matrix ($K$):** Calibrated focal length ($c = 2472.02$), principal point $(x_0, y_0) = (-36.8, -17.9)$, skew, and relative scale factor.
    * **Extrinsic Parameters:** Rotation matrix $R$ (Euler angles: $\omega, \varphi, \kappa$) and camera projection center $X_0 = -B^{-1} p_4$.
* 📄 **[Technical Report (PDF - Greek)](./HW2_Projection_Matrix_DLT_Calibration.pdf)**

---

### 3. Planar Homography & Metric Image Rectification
* **Objective:** Estimation of 2D projective transformation between planar physical targets and image planes to produce metric ortho-rectified imagery.
* **Methodology:**
  * Measured control points across a planar calibration grid ($40 \times 40\text{ cm}$).
  * Computed homography $H$ and inverse transform $H^{-1}$ using OpenCV's normalized transformation algorithms.
  * Evaluated individual reprojection residuals, isolating outliers to achieve a global $RMSE = 0.17\text{ cm}$.
  * Generated true-to-scale rectified ortho-images applying combined transformation pipelines ($H_{total} = \text{Scale} \cdot \text{Translate} \cdot H$) with defined physical ground sample distance ($12.5\text{ px/cm}$).
* 📄 **[Technical Report (PDF - Greek)](./HW3_Planar_Homography_Metric_Rectification.pdf)**

---

## 🛠️ Tech Stack & Methods
* **Language:** Python
* **Libraries:** OpenCV (`cv2`), NumPy, SciPy
* **Core Topics:** Epipolar Geometry, Projective Geometry, Camera Matrix Decomposition, SVD, QR Factorization, Homography, Perspective Correction, DLT
