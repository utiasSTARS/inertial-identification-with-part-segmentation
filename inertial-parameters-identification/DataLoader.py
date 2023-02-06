from pathlib import Path
import yaml
import numpy as np
import open3d as o3d
from spatialmath import SE3
from spatialmath.base import skew
import with_respect_to as WRT
from KinoDynamicLoader import Record, Loader

class DataLoader:
    '''
    Load the content of a .PKL file containing Records.
    Load a .YAML file containing inertia information and the pose of the grasp frame
        with respect to the mesh frame (X_MG_M).
    Load a .PLY file containing the object point-cloud and its segmentation
        expressed relative to an arbitrary frame (mesh). The grasping frame is defined
        with respect to that frame (X_MG_M).
    
    During manipulation the TCP is assumed to coincide with the grasping frame (no slipping)
    such that the point-cloud can be expressed wrt TCP with X_TM_T = X_MG_M.inv()
    and then tracked while manipulating the object with X_WM_W = X_WT_W @ X_TM_T
    or expressed relative to the sensor as X_SM_S = X_ST_S @ X_MG_M.inv()
    '''
    def __init__(self, world_name:str, pkl_file_path:Path, yaml_file_path:Path, ply_file_path:Path) -> None:
        self.world_name = world_name
        real_world_name = 'xarm-id-traj'
        sim_world_name  = 'trajsim'
        self.load_transformations(world_name)
        self.load_pkl(pkl_file_path)
        self.load_yaml(yaml_file_path)
        self.load_ply(ply_file_path)
    
    def verify_path(self, file_path:Path, extension:str):
        if not file_path.exists():
            print('ERROR: Data file path does not exist: '+str(file_path))
            raise FileNotFoundError
        if file_path.is_dir():
            print('ERROR: Data file path points to a directory: '+str(file_path))
            raise ValueError
        if file_path.suffix.upper() != extension.upper():
            print('ERROR: Data file path must point to a '+extension.upper()+' file: '+str(file_path))
            raise ValueError

    #Compute the inertia tensor wrt sensor frame from the inertia tensor wrt center of mass.
    # J_C_M: Inertia tensor computed at the location of the center of mass about axes aligned with the object.
    # p_SC_S: Position of the center of mass of the object with respect to the sensor frame.
    # R_SM_S: Orientation of the object with respect to the sensor frame.
    # mass: Mass of the object.
    def inertia_wrt_sensor(self, J_C_M, p_SC_S, R_SM_S, mass):
        #1. Uses a similarity transform to align the axes of the I_com to the sensor frame. (Rotation)
        J_C_S = R_SM_S @ J_C_M @ R_SM_S.T
        #2. Uses the parallel axis theorem to express the inertia tensor wrt the sensor. (Translation)
        J_S_S = J_C_S - mass * skew(p_SC_S) @ skew(p_SC_S)
        return J_S_S

    #Set an attribute from loaded parameters
    def loadParam(self, data, category, param):
        name = category+'_'+param
        if isinstance(data[category][param],str):
            exec('self.%s = "%s"' % (name,data[category][param]))
        if isinstance(data[category][param],int):
            exec("self.%s = %d" % (name,data[category][param]))
        if isinstance(data[category][param],float):
            exec("self.%s = %f" % (name,data[category][param]))
        if isinstance(data[category][param],list):
            exec("self.%s = np.array(%s)" % (name,data[category][param]))

    def load_yaml(self, yaml_file_path:Path):
        self.verify_path(yaml_file_path, '.YAML')
        
        yaml_file_path = yaml_file_path.resolve().as_posix()
        d = yaml.load(open(yaml_file_path,"r"),Loader=yaml.Loader)
        categories  = d.keys()
        for category in categories:
            params = d[category].keys()
            for param in params:
                self.loadParam(d, category, param)

        
        p_MC_M = self.OBJECT_COM_WRT_MESH
        J_C_M  = self.OBJECT_INERTIA_WRT_COM
        #Pose of object-grasp wrt object-mesh
        self.X_MG_M = SE3(self.OBJECT_GRASP_WRT_MESH, check=False)
        #Pose of object-mesh wrt ft-sensor assuming that tcp is coincident with object-grasp
        self.X_SM_S = self.X_ST_S @ self.X_MG_M.inv()
        #The TCP is assumed to be coincident with the grasping frame (object-grasp)
        # such that we can do compute the position of centre of mass relative to sensor
        # with:
        p_SC_S = (self.X_SM_S.A @ np.block([p_MC_M,1]))[0:3]  
        #Compute inertia relative to sensor
        J_S_S = self.inertia_wrt_sensor(J_C_M, p_SC_S, self.X_SM_S.R, self.OBJECT_MASS)
        #Inertial parameters wrt sensor
        self.inertial_params = np.block([self.OBJECT_MASS, self.OBJECT_MASS*p_SC_S, np.block([J_S_S[0,0:3],J_S_S[1,1:3],J_S_S[2,2:3]])])

    def load_ply(self, ply_file_path:Path):
        '''
        Load a .PLY file with the a format similar to this:
            ply
            format ascii 1.0
            element vertex 1931
            property float32 x
            property float32 y
            property float32 z
            property float32 red
            property float32 green
            property float32 blue
            property float32 volume
            property uint8 segmentation
            end_header
        '''
        self.verify_path(ply_file_path, '.PLY')

        pcd = o3d.t.io.read_point_cloud(ply_file_path.resolve().as_posix())
        seg    = pcd.point['segmentation'].numpy()
        vol    = pcd.point['volume'].numpy()
        #All points are initially defined relative to object-mesh
        # so we redefined them relative to ft-sensor
        points_M = pcd.point['points'].numpy()
        points_S = self.X_SM_S.A @ np.vstack((points_M.T, np.ones((1,len(points_M)))))
        #Then we append the segmentation ID to each point
        self.seg_point_cloud_S = np.hstack((points_S.T[:,0:3], seg))
        #Then we append the volume to each point. 
        # This is the volume of the tetrahedron, whose centroid is the point.
        self.seg_point_cloud_S = np.hstack((self.seg_point_cloud_S, vol))
        #Extends of the bounding box of the object expressed in the ft-sensor frame
        self.extents_S = self.X_SM_S.R @ (pcd.get_max_bound() - pcd.get_min_bound()).numpy()

    def load_pkl(self, pkl_file_path:Path):
        '''
        Reads a .PKL containing a list of Records, each of which contains at least:
        - Force/Torque Sensor pose (SE3)
        - Force/Torque Sensor velocities (SpatialVelocity)
        - Force/Torque Sensor accelerations (6D)
        - Forces and Torques (SpatialForce)
        with all information relative to world, the optimization is based on this information.
        '''
        self.verify_path(pkl_file_path, '.PKL')

        #Load content of .PKL
        self._loader = Loader()
        self._loader.load(pkl_file_path)
        self.records = self._loader.records

    def load_transformations(self, world_name:str):
        '''
        Load key transformations from awith-respect-to database.
        '''
        db = WRT.DbConnector()
        self.X_ST_S = SE3(db.In(world_name).Get('tcp').Wrt('ft-sensor').Ei('ft-sensor'))
        self._db = db
