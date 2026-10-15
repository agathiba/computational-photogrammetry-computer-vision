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

<p align="center">
  <img src="images/affine_rectification.png" alt="Original image, rectification without correction, and affine rectification using vanishing points">
  <br><em>Original image, rectification without correction, and affine rectification using vanishing points</em>
</p>

---

### 2. Camera Projection Matrix Estimation & Geometric Decomposition (DLT)
* **Objective:** Full camera calibration and pose estimation from corresponding 3D world coordinates ($X, Y, Z$) and 2D image coordinates ($x, y$).
* **Mathematical Workflow:**
  * **Direct Linear Transformation (DLT):** Formulated the homogeneous linear system $Ap = 0$ ($36 \times 12$ coefficient matrix from 18 ground control points).
  * **Singular Value Decomposition (SVD):** Solved for the optimal projection matrix $P$ ($A = U D V^T$, extracting the eigenvector corresponding to the smallest singular value).
  * **Reprojection Error:** Validated spatial accuracy achieving a sub-pixel root mean square reprojection error ($\sigma_0 = 0.46$ px).
  * **Matrix Decomposition (RQ):** Decomposed submatrix $B_{3 \times 3}$ via RQ factorization to extract:
    * **Camera Intrinsic Matrix ($K$):** Calibrated focal length ($c = 2472.02$), principal point $(x_0, y_0) = (-36.8, -17.9)$, skew, and relative scale factor.
    * **Extrinsic Parameters:** Rotation matrix $R$ (Euler angles: $\omega, \varphi, \kappa$) and camera projection center $X_0 = -B^{-1} p_4$.
  * **Sign normalization:** RQ decomposition is unique only up to the signs of the diagonal of $K$. The code fixes the overall sign of $P$ so that $\det(B) > 0$ and multiplies $K$ and $R$ by $T = \mathrm{diag}(\mathrm{sign}(K_{ii}))$, giving positive focal lengths and a proper rotation ($\det R = +1$) without changing $P$.

  > **Note:** this normalization was added after the original report. The report lists a relative scale deviation $m = -1.998$ and a roll of $179.6°$, which came from the sign ambiguity; with the corrected decomposition, $m = -0.002$ (practically square pixels) and the reprojection error is unchanged ($\sigma_0 = 0.46$ px).
* 📄 **[Technical Report (PDF - Greek)](./HW2_Projection_Matrix_DLT_Calibration.pdf)**

---

### 3. Planar Homography & Metric Image Rectification
* **Objective:** Estimation of 2D projective transformation between planar physical targets and image planes to produce metric ortho-rectified imagery.
* **Methodology:**
  * Measured control points across a planar calibration grid ($40 \times 40\text{ cm}$).
  * Computed homography $H$ and inverse transform $H^{-1}$ from 9 control points with `cv2.findHomography` (least squares).
  * Evaluated the reprojection residual of each control point, achieving a global $RMSE = 0.17\text{ cm}$.
  * Generated true-to-scale rectified ortho-images applying combined transformation pipelines ($H_{total} = \text{Scale} \cdot \text{Translate} \cdot H$) with defined physical ground sample distance ($12.5\text{ px/cm}$).
* 📄 **[Technical Report (PDF - Greek)](./HW3_Planar_Homography_Metric_Rectification.pdf)**

<p align="center">
  <img src="images/metric_rectification_chessboard.png" width="700" alt="Planar homography: original oblique photo (left) and metric rectification (right)">
  <br><em>Planar homography: original oblique photo (left) and metric rectification (right)</em>
</p>

---

## 🛠️ Tech Stack & Methods
* **Language:** Python
* **Libraries:** OpenCV (`cv2`), NumPy, SciPy
* **Core Topics:** Projective Geometry, Vanishing Points, Camera Matrix Decomposition, SVD, RQ Factorization, Homography, Perspective Correction, DLT

---

## 📁 Repository Structure

```
├── hw1_affine_rectification/
│   ├── affine_rectification.py      # Vanishing points, line at infinity, projective rectification
│   ├── Brueghel.jpg                 # Input image
│   └── results/                     # Rectified image and comparison figure
├── hw2_dlt_calibration/
│   ├── dlt_calibration.py           # DLT, SVD, RQ decomposition into K, R, X0
│   ├── image_points_pix.txt         # 18 image points (pixels)
│   ├── object_points_m.txt          # 18 object points (metres)
│   └── results/output.txt           # Printed results
├── hw3_metric_rectification/
│   ├── select_points.py             # Helper: click control points on the image
│   ├── metric_rectification.py      # Homography, RMSE, metric rectification
│   ├── chessboard.jpg               # Input image (40 x 40 cm board)
│   └── results/                     # Rectified image and image with control points
├── images/                          # Figures used in this README
├── HW1_*.pdf, HW2_*.pdf, HW3_*.pdf  # Technical reports (Greek)
└── requirements.txt
```

## ▶️ How to Run

```bash
git clone https://github.com/agathiba/computational-photogrammetry-computer-vision.git
cd computational-photogrammetry-computer-vision
pip install -r requirements.txt

python hw1_affine_rectification/affine_rectification.py
python hw2_dlt_calibration/dlt_calibration.py
python hw3_metric_rectification/metric_rectification.py
```

* **HW1** opens a window where the four corners of the painting's frame are selected with the mouse (top-left, top-right, bottom-right, bottom-left). Set `INTERACTIVE = False` at the top of the script to reuse the points from the report instead.
* **HW2** and **HW3** run without any interaction and print their results to the console; output images are saved in each `results/` folder.
