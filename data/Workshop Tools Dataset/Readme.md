# Workshop Tools Dataset
Motivated by the need for a dataset that also includes inertial information about the objects, we contribute the following dataset. It contains 20 common workshop tools, and for each object:
- a watertight triangular surface mesh;
- a synthetic colored surface point-cloud;
- ground truth inertial parameters;
- ground truth part-level segmentation; and 
- a grasping reference frame.

## List of Objects
- Adjustable Wrench
- Bent Jaw Pliers
- C Clamp
- Electronic Caliper
- Hacksaw 
- Machinist Hammer
- Nut Screwdriver
- Ruler
- Socket Wrench
- Vise Grip
- Allen Key
- Box Wrench
- Clamp
- File
- Hammer
- Measuring Tape
- Pliers
- Rubber Mallet
- Screwdriver
- Vise Clamp

## List of Files per Object
Each object has its on dedicated folder containing the following files:
- `Frames.png` :  
Picture of the point-cloud in point-cloud.ply with the mesh reference frames, the grasping reference frame and the centre of mass. The mesh reference frame is the frame all points are expressed relative to and the world "Origin" is written over its origin. The grasping reference frame was manually defined such as to express how a human worker would intuitively grasp the object. The centre of mass of the object is visualized as a purple ball.
- `Inertia.txt` :  
Geometric and mass properties of the object as computed with the CAD software used to produce the triangular mesh. This file is not directly used by our software but can be more easily understood by humans.
- `Inertia.yaml` :  
This file is read by our software to obtain the ground truth inertial properties as well as the transform that relates the grasping reference frame with respect to the mesh frame.
- `Materials.txt` :  
Human-readable notes about the materials, and therefore the mass densities, used for the object parts.  
- `mesh.ply` :  
Binary millimiter-scaled triangular mesh with colored vertices and the following header:
    ```
    ply
    format binary_little_endian 1.0
    comment SOLIDWORKS generated,length unit = millimeters
    element vertex 10913
    property float x
    property float y
    property float z
    element face 21642
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    property list uchar int vertex_indices
    end_header  
    ```
- `point-cloud.ply` :  
    Colored and part-labeled point cloud with the following format header:
    ```
    ply
    format ascii 1.0
    comment Created by Open3D
    element vertex 2000
    property float32 x
    property float32 y
    property float32 z
    property float32 red
    property float32 green
    property float32 blue
    property uint8 segmentation
    end_header
    ```
- `reconstructed_mesh.ply` :  
    Triangular mesh reconstructed from the point-cloud in `point-cloud.ply` using the method described in the paper referenced below. Can be compared to the original mesh in `mesh.ply` to evaluate the quality of the reconstruction.
    ```
    ply
    format ascii 1.0
    comment VCGLIB generated
    element vertex 256
    property double x
    property double y
    property double z
    element face 500
    property list uchar int vertex_indices
    end_header
    ```

## Citation
If you used any part of this software in your work, please cite our paper:  
```
@inproceedings{Nadeau_PartSegForInertialIdent_2023, 
    AUTHOR    = {Philippe Nadeau AND Matthew Giamou AND Jonathan Kelly}, 
    TITLE     = { {The Sum of Its Parts: Visual Part Segmentation for Inertial Parameter Identification of Manipulated Objects} }, 
    BOOKTITLE = {Proceedings of the {IEEE} International Conference on Robotics and Automation {(ICRA'23})},
    YEAR      = {2023}, 
    ADDRESS   = {London, UK}, 
    MONTH     = {May}, 
    DOI       = {}
}
```

## License
Copyright (c) 2023 Philippe Nadeau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.