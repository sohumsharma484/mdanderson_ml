import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Convert each annoated image to mask using one hot encoding
with open("filenames.txt", "r") as f:
    for line in f.readlines():
        print(line)
        line = line.strip()
        img = cv2.imread(line + "_ISH.jpg")   
        annotated = cv2.imread(line + "_annotated.jpg")
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)

        # HSV color ranges for each tumor region
        colorOptions = [
            ["Infiltrating Tumor", np.array([148, 100, 20], dtype="uint8"), np.array([152, 255, 255], dtype="uint8")],
            ["Perinecrotic zone", np.array([96, 100, 210], dtype="uint8"), np.array([98, 255, 255], dtype="uint8")],
            ["Leading Edge", np.array([94, 100, 20], dtype="uint8"), np.array([95, 255, 200], dtype="uint8")],
            ["Pseudopalisading cells but no visible necrosis", np.array([119, 100, 20], dtype="uint8"), np.array([121, 255, 255], dtype="uint8")],
            ["Cellular Tumor", np.array([60, 100, 20], dtype="uint8"), np.array([62, 255, 255], dtype="uint8")],
            ['Necrosis', np.array([0, 0, 0], dtype="uint8"), np.array([10, 10, 10], dtype="uint8")],
            ['Microvascular proliferation', np.array([4, 50, 20], dtype="uint8"), np.array([6, 255, 255], dtype="uint8")],
            ['Hyperplastic blood vessels', np.array([11, 50, 20], dtype="uint8"), np.array([12, 255, 255], dtype="uint8")],
            ['Pseudopalisading cells around necrosis', np.array([83, 50, 20], dtype="uint8"), np.array([85, 255, 255], dtype="uint8")]
        ]

        # 0 will represet white background
        mask = np.zeros((img.shape[0], img.shape[1]))

        i = 1
        for colorOption in colorOptions:
            color_mask = cv2.inRange(annotated, colorOption[1], colorOption[2])
            mask[color_mask > 0] = i
            i += 1
        cv2.imwrite(line + "_mask.png", mask) 
