# -*- coding: utf-8 -*-
"""
Created on Fri May 16 19:16:14 2025

@author: Bantisouli Agathi
"""
# Βιβλιοθήκες
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# True: select the 4 frame corners with the mouse
# False: reuse the points used in the report
INTERACTIVE = True
PRESET_POINTS = [[67, 21], [3795, 122], [3696, 2699], [176, 2806]]

# ---- ΒΗΜΑ 1: Φόρτωμα εικόνας ----
img = cv2.imread(os.path.join(BASE_DIR, 'Brueghel.jpg'))  
if img is None:
    raise FileNotFoundError("Η εικόνα δεν βρέθηκε.")

# Αποθήκευση των αρχικών διαστάσεων της εικόνας
orig_h, orig_w = img.shape[:2]
print("Ύψος εικόνας:", orig_h, "| Πλάτος εικόνας:", orig_w)

points = [] if INTERACTIVE else [list(p) for p in PRESET_POINTS]

# ---- ΒΗΜΑ 2: Επιλογή 4 σημείων με το ποντίκι ----
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Σημείο p{len(points)}: ({x}, {y})")

# Ρύθμιση παραθύρου OpenCV για επιλογή σημείων
if INTERACTIVE:
    cv2.namedWindow("Επιλογή Σημείων", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Επιλογή Σημείων", 1200, 800)
    cv2.setMouseCallback("Επιλογή Σημείων", select_point)

# Προβολή της εικόνας μέχρι να διαλέξεις 4 σημεία
while len(points) < 4:
    temp = img.copy()
    for p in points:
        cv2.circle(temp, (p[0], p[1]), 8, (0, 0, 255), -1)
    cv2.imshow("Επιλογή Σημείων", temp)
    if cv2.waitKey(1) & 0xFF == 27:  # Πάτημα ESC για ακύρωση
        break

if INTERACTIVE:
    cv2.destroyAllWindows()

# Εκτύπωση των τελικών σημείων
points = np.array(points)
print("Τελικά σημεία:")
for i, pt in enumerate(points):
    print(f"p{i+1} = {pt}")

# --- ΒΗΜΑ 3: Υπολογισμός γραμμών και σημείων φυγής ---

# Υπολογισμός ευθείας με εξωτερικό γινόμενο (cross product)
def compute_line(p1, p2):
    return np.cross(p1, p2)

# Μετατροπή σημείων σε ομογενείς συντεταγμένες (x, y, 1)
def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0])

# Ανάθεση των 4 σημείων σε p1, p2, p3, p4
p1, p2, p3, p4 = [to_homogeneous(pt) for pt in points]

# Υπολογισμός ευθειών του πλαισίου (παράλληλες κατευθύνσεις)
m1 = compute_line(p1, p2)  # πάνω
m2 = compute_line(p4, p3)  # κάτω
n1 = compute_line(p1, p4)  # αριστερά
n2 = compute_line(p2, p3)  # δεξιά

print("\nΓραμμές:")
print("m1 (πάνω):", m1)
print("m2 (κάτω):", m2)
print("n1 (αριστερά):", n1)
print("n2 (δεξιά):", n2)

# Σημεία φυγής από τομές αντίστοιχων γραμμών
v_m = np.cross(m1, m2)  # σημείο φυγής οριζόντιο
v_n = np.cross(n1, n2)  # σημείο φυγής κάθετο

print("\nΣημεία φυγής:")
print("v_m (οριζόντια):", v_m)
print("v_n (κάθετα):", v_n)

# Ευθεία στο άπειρο από τα 2 σημεία φυγής
l_v = np.cross(v_m, v_n)
print("Ευθεία φυγής l_v:", l_v)

# Κανονικοποίηση (ώστε το τελευταίο στοιχείο να είναι 1)
l_v = l_v / l_v[2]

# ---- ΒΗΜΑ 4: Δημιουργία projective μετασχηματισμού Hp ----
Hp = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [l_v[0], l_v[1], l_v[2]]
])
print("\nΜετασχηματισμός Hp:\n", Hp)

# Καμβάς χωρίς διόρθωση (χωρίς translate) 
raw = cv2.warpPerspective(img, Hp, (orig_w, orig_h))

# ---- ΒΗΜΑ 5: Υπολογισμός νέων ορίων καμβά ----
# Γωνίες της αρχικής εικόνας
corners = np.array([
    [0, 0],
    [orig_w, 0],
    [orig_w, orig_h],
    [0, orig_h]
], dtype='float32')

# Μετατροπή σε ομογενείς
corners_hom = np.hstack([corners, np.ones((4, 1))])

# Εφαρμογή Hp στις γωνίες
transformed_corners = (Hp @ corners_hom.T).T
transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2][:, np.newaxis]

# Υπολογισμός νέων ορίων για την εικόνα
x_min, y_min = transformed_corners.min(axis=0).astype(int)
x_max, y_max = transformed_corners.max(axis=0).astype(int)

print("\nΌρια νέου καμβά:")
print("x_min:", x_min, "| x_max:", x_max)
print("y_min:", y_min, "| y_max:", y_max)

# Μεταφορά για να αποφύγουμε αρνητικές συντεταγμένες
translate = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
])

# Συνδυασμός μετασχηματισμών (Hp και μετά μεταφορά)
Hp_corrected = translate @ Hp

# Νέο μέγεθος εικόνας
new_w = x_max - x_min
new_h = y_max - y_min
print("Νέες διαστάσεις εικόνας:", new_w, "x", new_h)

# ---- ΒΗΜΑ 6: Εφαρμογή μετασχηματισμού στην εικόνα ----
corrected = cv2.warpPerspective(img, Hp_corrected, (new_w, new_h))

# Εμφάνιση σύγκρισης
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Αρχική Εικόνα")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
plt.title("Χωρίς διόρθωση (εκτός ορίων!)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
plt.title("Με διόρθωση (σωστή προβολή)")
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=100)
plt.show()

# Αποθήκευση αποτελέσματος
cv2.imwrite(os.path.join(RESULTS_DIR, "brueghel_final.jpg"), corrected)
print("\n✅ Η εικόνα αποθηκεύτηκε ως brueghel_final.jpg")

# Extra: Υπολογισμός H_p-T για κανονικοποίηση
Hp_T = np.array([
    [1, 0, -l_v[0] / l_v[2]],
    [0, 1, -l_v[1] / l_v[2]],
    [0, 0, 1 / l_v[2]]
])

print("\nH_p-T =")
print(Hp_T)

# Έλεγχος ότι H_p-T * l_v = (0, 0, 1)^T
lv_check = Hp_T @ l_v
print("\nH_p-T * l_v =", lv_check)
