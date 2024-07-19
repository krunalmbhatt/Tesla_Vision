import json
import numpy as np
import os
import glob
from PIL import Image

# Load the JSON file containing (x, y, z) coordinates
json_path = '/home/krunalbhatt/study/Sem2/Computer_vision/project/P3/lane_detect/outputs/inference/scene7/lane_scene_7_uvz.json'
with open(json_path, 'r') as file:
    contours_data = json.load(file)


# Iterate over each frame in the JSON data
frame = contours_data[1]

#########################IMAGE TO CAM COORDS CONVERSION########################

cam_mat = np.array([[1.594657424424391e+03, 0, 6.552961052379301e+02],
                          [0, 1.607694179766480e+03, 4.143627123354900e+02],
                          [0, 0, 1]]).reshape(3,3)

def img_to_cam(u_and_v, camera_matrix, depth):
   
    if not isinstance(u_and_v, np.ndarray):
        u_and_v = np.array(u_and_v)
    image_coords_homogeneous = np.append(u_and_v, 1)

    inv_cam_mat = np.linalg.inv(camera_matrix)

    cam_xy = depth * np.dot(inv_cam_mat, image_coords_homogeneous)

    # Rotation in blender camera frame
    R = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])
    cam_xy = np.dot(R, cam_xy)

    # Appending 1 for homogeneous representation
    camera_homogeneous = np.append(cam_xy, 1)

    # RT MATRIX
    H = np.array([[0.0, 0.0, -1.0, 0.0],
                  [-1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    camera_coords = np.dot(H, camera_homogeneous)

    # Convert camera coordinates to non-homogeneous coordinates
    camera_non_homogeneous = camera_coords[:3]

    # Set the z coordinate to 0
    # camera_non_homogeneous[2] = 0

    return camera_non_homogeneous

##############################################################################
output_data = []

# Iterate over each frame in the JSON data
for frame_number, frame in enumerate(contours_data):
    print(f"Processing frame {frame_number}")
    output_frame = {}
    # Iterate over each detection in the frame
    for key, value in frame.items():
        print(f"Processing {key}")
        output_detection = []
        for detection in value:
            label = detection['label']
            output_contour = []
            output_detection.append({'label': label})
            for contour in detection['contours']:
                for point in contour:
                    x, y, z = point
                    # Convert image coordinates to camera coordinates
                    camera_coords = img_to_cam([x, y], cam_mat, z)
                    output_contour.append(camera_coords.tolist())
                output_detection.append({'contours': output_contour})
            output_frame[key] = output_detection
    output_data.append(output_frame)

# Save the output data to a new JSON file
with open('/home/krunalbhatt/study/Sem2/Computer_vision/project/P3/lane_detect/outputs/inference/scene7/lane_scene_7_3D.json', 'w') as file:
    json.dump(output_data, file)