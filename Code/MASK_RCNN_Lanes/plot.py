##################PLOTS CURVES###########################

#import bpy
#import json
#import numpy as np

## Load the JSON file
#with open('/home/krunalbhatt/study/Sem2/Computer_vision/project/P3/lane_detect/outputs/inference/scene11/scene_11_3D.json', 'r') as file:
#    data = json.load(file)

## Extract points for a specific frame
#frame_key = 'frame_0001'  # adjust this to the frame you want
#points = np.array([item['contours'] for item in data[0][frame_key]])

## Calculate the best fit line
#x = points[:,0]
#y = points[:,1]
#slope = (len(x)*np.sum(x*y) - np.sum(x)*np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x)**2)
#intercept = (np.sum(y) - slope*np.sum(x)) / len(x)

## Create a new line object in Blender
#bpy.ops.object.add(type='CURVE', location=(0,0,0))
#line = bpy.context.object

## Set the line's points to the ones we calculated
#line.data.splines.new('POLY')
#line.data.splines[0].points.add(len(points)-1)
#for i, point in enumerate(points):
#    x, y, z = point
#    line.data.splines[0].points[i].co = (x, y, z, 1)

##########################UPDATED LANES #######################
#import bpy
#import json
#import numpy as np

## Load the points from the JSON file
#with open('/home/krunalbhatt/study/Sem2/Computer_vision/project/P3/lane_detect/outputs/inference/scene11/scene_11_3D.json', 'r') as f:
#    data = json.load(f)

## Extract points for a specific frame
#frame = data[0]

## Iterate over each detection in the frame
#for key, value in frame.items():
#    print(f"Processing {key}")
#    points = []
#    for detection in value:
#        for point in detection["contours"]:
#            print(point)
#            x, y, z = point
#            points.append((x, y, z))  # append the float points directly

#    points = np.array(points)

#    # Create a new line object in Blender for each set of points
#    bpy.ops.object.add(type='CURVE', location=(0,0,0))
#    line = bpy.context.object

#    # Set the line's points to the ones we calculated
#    line.data.splines.new('POLY')
#    line.data.splines[0].points.add(len(points)-1)
#    for i, point in enumerate(points):
#        # Check if point is iterable and has exactly three items
#        if isinstance(point, (list, tuple)) and len(point) == 3:
#            x, y, z = point
#            line.data.splines[0].points[i].co = (x, y, z, 1)


#############PLOTS THE GOOD LANES#######################
import bpy
import json
import numpy as np

# Load the points from the JSON file
with open('/home/krunalbhatt/study/Sem2/Computer_vision/project/P3/lane_detect/outputs/inference/scene11/scene_11_3D.json', 'r') as f:
    data = json.load(f)

# Extract points for a specific frame
frame = data[0]

# Iterate over each detection in the frame
for key, value in frame.items():
    print(f"Processing {key}")
    for i in range(0, len(value), 2):  # assuming label and contours always come in pairs
        label = value[i].get('label', 'no_label')
        print(f"Processing label: {label}")
        points = []
        if i+1 < len(value) and "contours" in value[i+1]:
            for point in value[i+1]["contours"]:
                print(point)
                x, y, z = point
                points.append((x, y, z))  # append the float points directly

        points = np.array(points)

        # Create a new curve data block
        curveData = bpy.data.curves.new(f'{label}_curve', type='CURVE')
        curveData.dimensions = '2D'

        # Create a new NURBS curve object
        polyline = curveData.splines.new('NURBS')
        polyline.points.add(len(points)-1)  # add points
        for i, coord in enumerate(points):
            x,y,z = coord/100
            polyline.points[i].co = (-x, -z, 0, 1)

        # Create a new object with the curve data block
        curveOB = bpy.data.objects.new(f'{label}_curve', curveData)

        # Link the curve object to the scene collection
        bpy.context.collection.objects.link(curveOB)

        # Convert the curve to a mesh
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        curveOB.select_set(True)  # Select the curve object
        bpy.context.view_layer.objects.active = curveOB  # Make the curve the active object
        bpy.ops.object.convert(target='MESH')

        # Enter edit mode
        bpy.ops.object.mode_set(mode='EDIT')

        # Select all mesh vertices
        bpy.ops.mesh.select_all(action='SELECT')

        # Fill the mesh
        bpy.ops.mesh.edge_face_add()

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create a new material
        mat = bpy.data.materials.new(name=f"{label}_Mat")
        mat.diffuse_color = (1, 1, 1, 1)  # white color

        # Assign it to the object
        if len(curveOB.data.materials):
            # assign to 1st material slot
            curveOB.data.materials[0] = mat
        else:
            # no slots
            curveOB.data.materials.append(mat)