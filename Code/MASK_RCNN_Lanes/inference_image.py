import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import glob
import os
import json

from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', 
    '--input', 
    required=True, 
    help='path to the input data'
)
parser.add_argument(
    '-t', 
    '--threshold', 
    default=0.5, 
    type=float,
    help='score threshold for discarding detection'
)
parser.add_argument(
    '-w',
    '--weights',
    default='out/checkpoint.pth',
    help='path to the trained wieght file'
)
parser.add_argument(
    '--show',
    action='store_true',
    help='whether to visualize the results in real-time on screen'
)
parser.add_argument(
    '--no-boxes',
    action='store_true',
    help='do not show bounding boxes, only show segmentation map'
)
args = parser.parse_args()

OUT_DIR = os.path.join('outputs', 'inference', 'scene8')
# OUT_DIR = os.path.join('outputs', 'inference')
os.makedirs(OUT_DIR, exist_ok=True)

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    weights=None, num_classes=91
)

model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

# initialize the model
ckpt = torch.load(args.weights)
model.load_state_dict(ckpt['model'])
# set the computation device
device = torch.device('cuda')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()
# print(model)

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

############################# ADDED ##########################################
def contours_to_dict(contours):
    # Convert contours to a format that can be JSON serialized
    contours_list = []
    for contour in contours:
        # contour is an array of shape (num_points, 1, 2) containing x, y coordinates
        # We reshape it to (num_points, 2) for easier processing later
        contour = contour.reshape(-1, 2)
        points_list = contour.tolist()  # Convert numpy array to a list
        contours_list.append(points_list)
    return contours_list

results = []

image_paths = glob.glob(os.path.join(args.input, '*.png'))
image_paths.sort()
for image_path in image_paths:
    image_name = os.path.basename(image_path).split('.')[0]
    print(image_path)
    image = Image.open(image_path)
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
    
    # transform the image
    image = transform(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    
    masks, boxes, labels = get_outputs(image, model, args.threshold)
    
    # print(masks)

    detections = []
    for i, mask in enumerate(masks):
        # Assuming the mask is a tensor with shape [1, height, width] and binary output
        # Convert it to a binary (0 or 255) image
        mask_array = mask.squeeze()
        mask_array = np.array(mask_array * 255, dtype=np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assuming labels[i] gives the label for the ith mask
        # label = labels[i]  # Convert tensor to integer and then to label string
        contour_points = contours_to_dict(contours)  # Convert contours to a list of points
        # print("before:",contour_points)
        #flip each pair to y,x for each list in list of lists
        contour_points = [[[y,x] for x,y in sublist] for sublist in contour_points]
        # print("after:",contour_points)
        label_index = labels[i]  # Convert tensor to integer


        ## CONVERT CONTOUR POINTS TO CAMERA COORDINATES
        # Pass each countour_points to image_to_camera function and store the result in a new list
        # for i, group in enumerate(contour_points):
        #     for j, point in enumerate(group):
        #         u, v = point
        #         XYZ = image_to_camera([u, v], camera_matrix, scale)
        #         # Replace the old coordinate with the transformed one
        #         contour_points[i][j] = XYZ.tolist()


        detection = {
            "label": labels[i],
            "contours": contour_points,
        }
        detections.append(detection)

    results.append({
        image_name: detections,
    })

    # result = draw_segmentation_map(orig_image, masks, boxes, labels, args)
    
    # # visualize the image
    # if args.show:
    #     cv2.imshow('Segmented image', np.array(result))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    
    # # set the save path
    # save_path = f"{OUT_DIR}/{image_path.split(os.path.sep)[-1].split('.')[0]}.png"
    # cv2.imwrite(save_path, result)

# Save the results to a JSON file
    json_file_path = os.path.join(OUT_DIR, 'lane_scene_8_uv.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)  # `indent=4` for pretty-printing

