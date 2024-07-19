# Einstein Vision

## Introduction
This document provides instructions for setting up and running the code for Einstein Vision. The project relies on several key libraries: Detic, Detectron2, SoftRas, YOLO3D, YoloPV2, Mask R-CNN, and Zoedepth. Each of these libraries is essential for the project's execution, offering a range of functionalities from object detection to 3D rendering.

## Prerequisites
Before running the project code, ensure you have the following libraries installed:

- Detic
- Detectron2
- SoftRas
- YOLO3D
- YoloPV2
- Mask R-CNN
- Zoedepth

## Installation Instructions

### Step 1: Clone the Repositories
Begin by cloning the repositories for each required library. Follow the installation instructions provided in their respective README files to set up each library correctly.

### Step 2: Install Dependencies
Each library may require specific dependencies. Ensure all dependencies are installed by following the installation guides provided with each library.

### Step 3: Library-Specific Setup
After installing the libraries and their dependencies, follow any post-installation instructions to configure each library. This may include setting up environment variables, downloading pre-trained models, or other library-specific setup steps.

## Configuring the Project

### Replacing Specific Files
To integrate all libraries into the project seamlessly, certain files, which were included in the submission need to be replaced


## Running the Code

After completing the setup and configuration, you can now run the project code. Ensure all paths and dependencies are correctly set up before execution. Specific lines in the code have to be changed to the absolute path. Phase 1/2 Output can be run from Detic using Run2.py. Phase 3 Outputs can be run from YOLO3D in the inference.py file. MASK-RCNN will br available on [this](https://debuggercafe.com/lane-detection-using-mask-rcnn/) link. Please follow the installation guidelines and then use the inference.py provided by us to generate an inference using command line argument available on the link.

## Troubleshooting

If you encounter any issues during setup or execution, please refer to the respective library's documentation for troubleshooting guidance. Ensure all libraries are updated to their latest versions to avoid compatibility issues.

