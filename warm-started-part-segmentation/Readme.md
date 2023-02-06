# Part-level Object Segmentation
This repository contains code that loads a colored point-cloud, perform shape reconstruction, initial clustering from local surface information, and part-level segmentation from global shape information.

## Dependencies
The following softwares will need to be installed prior to the execution of our implementation: 
- Python 3
- [TetGen](https://wias-berlin.de/software/index.jsp?id=TetGen&lang=1)
- [MeshLab](https://www.meshlab.net/#download)
- [Open3D](http://www.open3d.org/docs/release/getting_started.html)
- [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/installation.html)
- [MatPlotLib](https://matplotlib.org/stable/users/getting_started/)
- [Numpy](https://numpy.org/install/)

## Files Description
- `Supervoxels_and_HTC.py` :  
Runs our proposed part-level segmentation algorithm on all objects from our contributed dataset and generate a part-level segmentation of the object.
- `HierarchicalTetClustering.py` :  
Our implementation of the algorithm (HTC) proposed in "Hierarchical convex approximation of 3D shapes for fast region selection".
- `SupervoxelSegmentation.py` :  
Our implementation of the algorithm proposed in "Toward better boundary preserved supervoxel segmentation for 3D point clouds".
- `VolumeReconstruction.py` :  
Our shape reconstruction routine that relies MeshLab.
- `EfficientTetrahedronSearch.py` :  
Module used by our implementation of HTC to improve runtime.
- `tetgen1.6.0/` :  
Directory that contains the source code for TetGen version 1.6 such that it can be easily built and used by our Python implementation.

## Initial Setup
- Install dependencies listed above
- Build TetGen
    ```bash
    cd warm-started-part-segmentation/tetgen1.6.0
    mkdir build  
    cd build  
    cmake ..  
    make  
    ```

## How to Run on Our Dataset
```bash
python3 Supervoxels_and_HTC.py
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