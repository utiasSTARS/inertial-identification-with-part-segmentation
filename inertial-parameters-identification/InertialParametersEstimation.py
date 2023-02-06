from OLS import OLS
from HPS import HPS
from GEO import GEO
from pathlib import Path
import numpy as np
from spatialmath.base import skew
from DataLoader import DataLoader

class Estimator:
    def __init__(self, world_name:str, pkl_file_path:Path, yaml_file_path:Path, ply_file_path:Path, estimator='HPS') -> None:
        self.estimation_algorithm = estimator

        self.load_data(world_name, pkl_file_path, yaml_file_path, ply_file_path)

        if estimator == 'OLS':
            self.estimator = OLS()
        if estimator == 'HPS':
            self.estimator = HPS(self.dl.seg_point_cloud_S)
        if estimator == 'GEO':
            #The prior uses the correct mass and the shape
            # of the point-cloud to get a prior on COM and Inertia.
            prior_mass = self.dl.inertial_params[0]
            points = self.dl.seg_point_cloud_S[:,0:3]
            prior_com = np.mean(points, axis=0)
            prior_J = np.zeros((3,3))
            for p in points:
                prior_J += - skew(p) @ skew(p)
            prior_J = prior_J/len(points) - skew(prior_com) @ skew(prior_com)
            prior_phi = prior_mass*np.block([1,prior_com, prior_J[0,0:3], prior_J[1,1:3] , prior_J[2,2]])

            self.estimator = GEO('Entropic', prior_phi, alpha=1e-5)
    
    def load_data(self, world_name:str, pkl_file_path:Path, yaml_file_path:Path, ply_file_path:Path):
        '''
        Load all data for all files.
        '''
        self.dl = DataLoader(world_name, pkl_file_path, yaml_file_path, ply_file_path)

    def process_data(self, v_noise_sd=[0,0], a_noise_sd=[0,0], ft_noise_sd=[0,0]):
        '''
        Iteratively go over the timesteps and express everything relative to the ft-sensor frame
        before updating the estimator with this new information.
        v_noise_sd: Std. Dev. of the noise for the linear and angular velocities, respectively.
        a_noise_sd: Std. Dev. of the noise for the linear and angular accelerations, respectively.
        ft_noise_sd: Std. Dev. of the noise for the forces and torques, respectively.
        '''
        i = 0
        rec = self.dl._loader.getRecord()
        while rec is not None:
            #Add synthetic noise to the signals
            rec.add_velocity_noise(v_noise_sd)
            rec.add_acceleration_noise(a_noise_sd)
            rec.add_wrench_noise(ft_noise_sd)
            #Pose of ft-sensor wrt world expressed in world
            X_WS_W = rec.ft_pose
            #Express data relative to ft-sensor
            rec = rec.expressed_in_sensor_frame()
            #Parse record
            X_WS_S  = rec.ft_pose           #Pose of ft-sensor wrt world expressed in ft-sensor
            lv_S_S  = rec.ft_vel.A[0:3]     #Linear velocity of ft-sensor
            av_S_S  = rec.ft_vel.A[3:6]     #Angular velocity of ft-sensor
            la_S_S  = rec.ft_acc[0:3]       #Linear acceleration of ft-sensor
            aa_S_S  = rec.ft_acc[3:6]       #Angular acceleration of ft-sensor
            f_S_S   = rec.ft_value.A[0:3]   #Forces expressed in ft-sensor
            t_S_S   = rec.ft_value.A[3:6]   #Torques expressed in ft-sensor
            #Gravity expressed in the world frame
            g_w = np.array([0,0,-9.81])
            #Gravity expressed in the ft-sensor frame
            g_s = X_WS_W.inv().R @ g_w
            #With the HPS algorithm, use only the poses when the robot is not moving much.
            if self.estimation_algorithm == 'HPS' \
                and np.linalg.norm(rec.ft_acc[0:3]) < 1 \
                and np.linalg.norm(rec.ft_acc[3:6]) < 1 \
                or self.estimation_algorithm != 'HPS':
                #Update estimator with new information
                self.estimator.update(g_s, la_S_S, av_S_S, aa_S_S, f_S_S, t_S_S)
            #Get next record
            rec = self.dl._loader.getRecord()
            i += 1

    def estimate(self):
        estimation = self.estimator.estimate().T[0]
        return estimation

    def get_condition_number(self):
        cond_number = self.estimator.compute_condition_number()
        return cond_number

    def evaluate(self, estimated_inertial_parameters, metric='size-based'):
        '''
        Evaluate how good the estimation was based on a metric.
        '''
        est_mass, est_com, est_inertia    = self.split_inertial_params(estimated_inertial_parameters)
        true_mass, true_com, true_inertia = self.split_inertial_params(self.dl.inertial_params)

        if metric == 'geodesic':
            error = self.GeodesicErrorMetric(est_inertia, est_com, est_mass, true_inertia, true_com, true_mass)
        if metric == 'size-based':
            mass_error    = self.MassErrorMetric(est_mass, true_mass)
            com_error     = self.CentreOfMassErrorMetric(est_com, true_com, self.dl.extents_S)
            inertia_error = self.InertiaErrorMetric(est_inertia, true_inertia, true_mass, self.dl.extents_S)
            mean_com_error      = np.mean(com_error)
            mean_inertia_error  = np.mean(np.block([inertia_error[0,0:3],inertia_error[1,1:3],inertia_error[2,2]]))
            error = [mass_error, mean_com_error, mean_inertia_error]
        if metric == 'relative':
            mass_error    = 100*np.abs(est_mass-true_mass)/true_mass
            com_error     = 100*np.divide(np.abs(est_com-true_com),np.abs(true_com))
            inertia_error = 100*np.divide(np.abs(est_inertia-true_inertia), np.abs(true_inertia))
            mean_com_error      = np.mean(com_error)
            mean_inertia_error  = np.mean(np.block([inertia_error[0,0:3],inertia_error[1,1:3],inertia_error[2,2]]))
            error = [mass_error, mean_com_error, mean_inertia_error]
        if metric == 'rmse':
            squared_errors = np.ndarray((6,0))
            #Create a temporary OLS estimator to build noiseless data matrices easily
            real_est = self.estimator
            temp_est = OLS()
            self.estimator = temp_est
            self.dl._loader.index_of_next_record = 0
            self.process_data()
            #Shorthand
            A = self.estimator.stacked_data_matrices
            b = self.estimator.stacked_wrench_vectors
            #Iterate over each timestep and compute the squared errors using the
            # estimated inertial parameters.
            for i in range(self.estimator.k_step):
                A_k = A[i*6:(i+1)*6,:]
                b_k = b[i*6:(i+1)*6,:]
                se = (b_k - A_k @ estimated_inertial_parameters.reshape((10,1)))**2
                squared_errors = np.hstack((squared_errors,se))
            #Root-Mean-Squared-Error
            error = np.mean(squared_errors, axis=1)**0.5
            error = error.tolist()
            #Reset the estimator to the original
            self.estimator = real_est
        return error

    def split_inertial_params(self, inertial_param_vector):
        '''
        Divide a vector of ten inertial parameters into the individual
        mass, centre of mass and inertia tensor components.
        '''
        v = inertial_param_vector
        m   = v[0]
        com = v[1:4]/m
        J   = np.ndarray((3,3))
        J[0,0:3] = v[4:7]
        J[0:3,0] = J[0,0:3]
        J[1,1:3] = v[7:9]
        J[1:3,1] = J[1,1:3]
        J[2,2]   = v[9]
        return m, com, J

    #See Wensing & Slotine 2018
    def ProjectOntoManifold(self, inertia, com, mass):
        '''
        Projection of the inertial parameters onto the positive semi-definite manifold.
        This 4x4 matrix is called "pseudoinertia" and needs to be positive definite.
        '''
        proj = np.zeros((4,4))
        proj[0:3, 0:3] = 0.5*np.trace(inertia) * np.eye(3) - inertia
        proj[0:3, 3] = mass * com
        proj[3, 0:3] = mass * com.T
        proj[3,3] = mass

        return proj

    def GeodesicErrorMetric(self, est_inertia, est_com, est_mass, true_inertia, true_com, true_mass, approximative=False):
        '''
        Compute a distance that is (proportional to or exactly) the geodesic distance between the estimated solution and the ground truth
        on the underlying Riemannian manifold.
        Returns -1 if the solution is not physically plausible/consistent.
        
        approximative : If True, will compute an approximation that is proportional to the Geodesic distance and faster to compute.
        '''
        estimation  = self.ProjectOntoManifold(est_inertia, est_com, est_mass)
        truth       = self.ProjectOntoManifold(true_inertia, true_com, true_mass)

        #Verify physical consistency
        est_plausible = self.isSymmetricPositiveDefinite(estimation)
        true_plausible = self.isSymmetricPositiveDefinite(truth)

        if est_plausible and true_plausible:
            p = np.linalg.inv(estimation) @ truth
            if approximative:
                #This is proportional but not equal to the geodesic distance
                # Scale invariant and unit-less scalar metric that is faster to compute.
                # See Sundaralingham 2020 and Lee & Park 2019
                approx_inertial_error = np.abs(4-np.trace(p))
                inertial_error = approx_inertial_error
            else:
                #This is the (exact) Riemannian geodesic distance between two SPD matrices
                # See Lee, Wensing, and Park, 2019, eqn. 17
                geodesic_inertial_error = np.sqrt(0.5*np.sum([np.log(l)**2 for l in np.linalg.eigvals(p)]))
                inertial_error = geodesic_inertial_error
        else:
            inertial_error = -1
        return inertial_error

    def isSymmetricPositiveDefinite(self, matrix):
        '''
        Return True if the matrix is Symmetric Positive Definite, otherwise return False.
        '''
        if np.allclose(matrix, matrix.T):
            try:
                np.linalg.cholesky(matrix)
                return True
            except np.linalg.LinAlgError:
                return False
        return False

    #Returns the standard percentage relative error
    def MassErrorMetric(self, estimated_mass, true_mass):
        return 100*( abs(estimated_mass-true_mass)/true_mass )

    #Uses the extends of the bounding-box of the object to put the error in perspective
    def CentreOfMassErrorMetric(self, estimated_com_c, true_com_c, true_boundingbox):
        error_vector = np.array([0,0,0])
        for i in range(0,3):
            try:
                error_vector[i] = 100*( abs(estimated_com_c[i]-true_com_c[i]) / abs(true_boundingbox[i]) )
            except OverflowError:
               error_vector[i] = np.Inf
        return error_vector

    #Computes the error metric of the inertia tensor
    def InertiaErrorMetric(self, estimated_Inertia_tensor, true_Inertia_tensor, true_mass, true_boundingbox):
        a = abs(true_boundingbox)
        error_matrix = np.zeros((3,3))
        for i in range(0,3):
            for j in range(0,3):
                if i == j:
                    delta = 1
                else:
                    delta = 0
                numer = estimated_Inertia_tensor[i,j] - true_Inertia_tensor[i,j]
                denom = (true_mass/12) * (delta*(a[0]**2 + a[1]**2 + a[2]**2) - a[i]*a[j])
                error_matrix[i,j] = 100*abs(numer/denom)
        return error_matrix

