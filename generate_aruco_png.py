""" Python script to generate 50 aruco markers as png 
    asked gemini/google ai this: "generate 50 aruco marker"
"""
import os
import cv2

# Create an output directory for the markers
output_dir = "aruco_markers_50"
os.makedirs(output_dir, exist_ok=True)

# Load the predefined dictionary (4x4 bit grid, contains 50 unique IDs)
# If using an older version of OpenCV, use: cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Set the pixel dimensions for the exported image (e.g., 400x400 pixels)
marker_size_pixels = 400

# Generate 50 markers (IDs 0 to 49)
for marker_id in range(50):
    # Generate the marker image
    # If using an older version of OpenCV, use: cv2.aruco.drawMarker(...)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
    
    # Save to the target directory
    file_path = os.path.join(output_dir, f"marker_{marker_id}.png")
    cv2.imwrite(file_path, marker_img)

print(f"Successfully generated 50 markers in the '{output_dir}' directory!")
