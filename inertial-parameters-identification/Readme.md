# Inertial Parameter Identification
This repository contains code that loads kinematic and dynamic data, process it and uses it for the estimation of the mass, centre of mass and inertia tensor of a manipulated object.

## Dependencies
The following softwares will need to be installed prior to the execution of our implementation: 
- Python 3
- [Open3D](http://www.open3d.org/docs/release/getting_started.html)
- [Numpy](https://numpy.org/install/)
- [SpatialMath](https://github.com/petercorke/spatialmath-python)
- [CvxPy](https://www.cvxpy.org/install/index.html)
- [MOSEK](https://www.mosek.com/downloads/)
- [With-Respect-To](https://github.com/PhilNad/with-respect-to)
- [ROS](http://wiki.ros.org/ROS/Installation)

## Files Description
- `RunExperiments.py` :  
Runs experiments on the 20 objects from our dataset, with 4 noise levels, and with 3 algorithms (i.e., HPS, GEO, OLS) for a total of 240 executions. Saves the results in a pickle file.
- `AnalyzeExperiments.py` :  
Uses the data from `RunExperiments.py` and generate a table that sumarizes the average performance of each algorithm for each noise level.
- `InertialParametersEstimation.py` :  
Loads the information about the manipulated object and instantiate an implemented estimator (HPS, GEO, or OLS) to process the data and estimate the inertial parameters.
- `KinoDynamicLoader.py` :  
Loads and saves kinematics and dynamics data from pickle files.
- `DataLoader.py` :  
Interface to load all data that the estimation might need.
- `HPS.py` :  
Implementation of the Homogeneous Part Segmentation (HPS) algorithm, our proposed method.
- `GEO.py` :  
Implementation of the method proposed in "Geometric Robot Dynamic Identification: A Convex Programming Approach".
- `OLS.py` :  
Implementation of the Ordinary Least Squares (OLS) method proposed in "Estimation of Inertial Parameters of Manipulator Loads and Links".

## Initial Setup
Install the dependencies listed above.

## How to Run on Our Dataset
```bash
python3 RunExperiments.py
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