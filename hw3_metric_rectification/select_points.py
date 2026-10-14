# Import Libraries
import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load image
img = cv2.imread(os.path.join(BASE_DIR, "chessboard.jpg"))

# Image shape
orig_h, orig_w = img.shape[:2]
print("Ύψος εικόνας:", orig_h, "| Πλάτος εικόνας:", orig_w)

points = []

# Select points
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 11:
        points.append([x, y])
        print(f"Σημείο p{len(points)}: ({x}, {y})")

# Margin
cv2.namedWindow("Επιλογή Σημείων", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Επιλογή Σημείων", 800, 1000)
cv2.setMouseCallback("Επιλογή Σημείων", select_point)

while len(points) < 11:
    temp = img.copy()
    for p in points:
        cv2.circle(temp, (p[0], p[1]), 8, (0, 0, 255), -1)
    cv2.imshow("Επιλογή Σημείων", temp)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC for exit
        break

cv2.destroyAllWindows()

points = np.array(points)
print("Τελικά σημεία:")
for i, pt in enumerate(points):
    print(f"p{i+1} = {pt}")