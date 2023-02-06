# Data Repository
This repository contains source code and data related to the paper titled _The Sum of Its Parts: Visual Part Segmentation for Inertial Parameter Identification of Manipulated Objects_, which appears in the proceedings of the International Conference on Robotics and Automation 2023.

The structure of this data repository is as follows:
- `warm-started-part-segmentation` :  
Source code of our implementation of the part-level object segmentation algorithm used in this work.
- `inertial-parameters-identification` :  
Source code for the inertial parameter identification with three estimation algorithms implemented (HPS, OLS, GEO).
- `data` :  
Main data repository that is required to run the above programs.
    - `Workshop Tools Dataset` :  
    Contains our contributed dataset of 20 workshop tools with a watertight mesh, a colored surface point-cloud, ground truth inertial parameters, gruond truth part-level segmentation and a grasping reference frame.
    - `Simulations` :  
    Contains synthetic data from the manipulation of the object by a simulated manipulator following a stop-and-go motion trajectory described in the paper.
    - `Segmentations` :  
    Contains labeled point-clouds resulting from running our proposed part-segmentation algorithm on the contributed dataset along with pictures comparing it to the groud-truth segmentation.

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