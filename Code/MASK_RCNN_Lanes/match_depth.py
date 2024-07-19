import json
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON file containing (x, y) coordinates
json_path = '/home/krunalbhatt/study/Sem2/Computer_vision/project/P3/lane_detect/outputs/inference/scene7/lane_scene_7_uv.json'
with open(json_path, 'r') as file:
    contours_data = json.load(file)

# Read depth images
depth_img_path = '/home/krunalbhatt/study/Sem2/Computer_vision/project/P3/kush_data/P3Data/Sequences/scene7/Undist/2023-03-03_11-21-43-front_undistort_marigold/depth_bw'

for i, frame in enumerate(contours_data, start=1):
    # Load the corresponding depth image
    depth_img = cv2.imread(os.path.join(depth_img_path, f'image{i}_pred.png'), cv2.IMREAD_UNCHANGED)
    # Iterate over each detection in the frame
    for key, value in frame.items():
        print(f"Processing {key}")
        for detection in value:
            label = detection['label']
            for contour in detection['contours']:
                for point in contour:
                    x, y = point
                    # Get the depth value at the (x, y) coordinate
                    depth_value = int(depth_img[x, y])
                    # Append the depth value to the point
                    point.append(depth_value)
            detection['label'] = label

# plt.plot(x_values, y_values)
# plt.title('Line Joining Points')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()
# Define the path to the new JSON file
new_json_path = '/home/krunalbhatt/study/Sem2/Computer_vision/project/P3/lane_detect/outputs/inference/scene7/lane_scene_7_uvz.json'

# Write the modified data to the new JSON file
with open(new_json_path, 'w') as file:
    json.dump(contours_data, file)


