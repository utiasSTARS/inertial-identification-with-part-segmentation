import cvxpy as cp
import numpy as np
import mosek as mo 
from spatialmath.base import skew
import open3d as o3d
'''
Homogeneous Part Segmentation for Inertial Parameter Identification
'''

class HPS:
    def __init__(self, segmented_point_cloud, batch_size=1E10):
        '''
        Assumes that the segmented_point_cloud array has a row per point
        with the three leftmost columns defining the position of the point
        relative to the ft-sensor and the rightmost column is the segmentation
        identifier associated with the point.
        '''
        self.BATCH_SIZE = batch_size
        self.segmented_point_cloud = segmented_point_cloud
        
        #Timestep counter
        self.k_step = 0

        #Define centroids
        self.part_centroids = self.define_centroids()

        #self.visualize_centroids()

        #Verify that we dont have too many segments/parts.
        if self.qty_parts > 4:
            print('Got {} segments. The number of parts is limited to 4 such that only stop-and-go motions are needed.'.format(self.qty_parts))
            raise ValueError
        
        #Used to report result
        self.estimation_result = np.zeros((self.qty_parts,1))

        #Accumulates data matrices
        self.stacked_data_matrices = np.ndarray((0,self.qty_parts))
        self.stacked_wrench_vectors= np.ndarray((0,1))

    def define_centroids(self):
        '''
        From the segmented point cloud, find the centroid of each segments/parts.
        '''
        #Column with the segmentation IDs
        seg_col = self.segmented_point_cloud[:,3]
        seg_ids = np.unique(seg_col)
        part_centroids = np.ndarray((3,len(seg_ids)))
        self.parts = []
        for i,seg_id in enumerate(seg_ids):
            #Find all points sharing the same segmentation ID
            idx  = self.segmented_point_cloud[:,3] == seg_id
            part = self.segmented_point_cloud[idx]
            self.parts.append(part)
            #Each tetrahedron has a different volume. To compute the centroid of the volume
            # of the part, it is required to take the volume of each tetrahedron into account.
            # To do so, the centroids are weighted with their corresponding volume.
            total_volume  = 0
            part_centroid = np.zeros((3,)) 
            for j in range(len(part)):
                tet_centroid = part[j,0:3]
                tet_volume   = part[j,4]
                total_volume += tet_volume
                part_centroid+= tet_centroid*tet_volume
            part_centroid /= total_volume
            part_centroids[:,i] = part_centroid
        #Memorize the number of centroids/segments/parts
        self.qty_parts = len(seg_ids)
        return part_centroids
    
    def visualize_centroids(self):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.segmented_point_cloud[:,0:3]))

        n_centroids = self.part_centroids.shape[1]
        centroids_list = []
        for i in range(n_centroids):
            com_sphere     = o3d.geometry.TriangleMesh.create_sphere(0.005)
            com_sphere.paint_uniform_color([1,0,1])
            position = self.part_centroids[:,i]
            com_sphere.translate(position)
            centroids_list.append(com_sphere)

        object_frame   = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries(centroids_list+[pcd, object_frame])

    def update(self, g_s, acc_lin_s_s, vel_ang_s_s, acc_ang_s_s, forces_s, torques_s):
        #Update the steps counter
        self.k_step += 1

        #Regressor matrix
        A_s = self.HPS_data_matrix(g_s)

        #Wrench matrix
        F_s = np.vstack((forces_s.reshape((3,1)), torques_s.reshape((3,1))))

        #Stack matrices.
        self.stacked_data_matrices = np.vstack((self.stacked_data_matrices, A_s))
        self.stacked_wrench_vectors= np.vstack((self.stacked_wrench_vectors, F_s))

        return self.estimation_result

    def estimate(self)->np.ndarray:

        #Shorthand
        A = self.stacked_data_matrices
        b = self.stacked_wrench_vectors

        #Respect the batch size
        lower = max([0,self.k_step-self.BATCH_SIZE])
        upper = self.k_step
        A = A[lower*6:,:]
        b = b[lower*6:,:]

        #Optimization variables
        part_masses     = cp.Variable(self.estimation_result.shape)

        #Objective function
        J_LS = cp.sum_squares(A @ part_masses - b)

        #Constraints: Each part has a positive mass
        constraints  = []
        for t in range(self.qty_parts):
            constraints += [part_masses[t] >= 0]

        #Solve the problem
        obj  = cp.Minimize(J_LS)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={mo.iparam.intpnt_solve_form : mo.solveform.dual})
        
        #Memorize part masses
        self.part_masses = part_masses.value

        #Get solver statistics
        setup_time = 0 if prob.solver_stats.setup_time is None else prob.solver_stats.setup_time
        solve_time = 0 if prob.solver_stats.solve_time is None else prob.solver_stats.solve_time
        total_time = setup_time + solve_time
        #print(total_time)

        #Compute inertial parameters assuming that the position of the points in the
        # point-cloud were defined relative to the ft-sensor frame.
        mass    = np.sum(self.part_masses)
        p_SC_S  = self.compute_com()
        J_S_S   = self.compute_inertia_matrix()
        inertial_params = np.block([[mass, mass*p_SC_S, J_S_S[0,0:3], J_S_S[1,1:3], J_S_S[2,2:3]]]).T

        return inertial_params

    def HPS_data_matrix(self, gravity):
        #Regressor matrix
        A = np.ndarray((6,self.qty_parts))

        #Each column is built the same way
        for i in range(self.qty_parts):
            A[0:3,i] = gravity
            A[3:6,i] = -skew(gravity) @ self.part_centroids[:,i]

        return A

    def compute_com(self):
        '''
        Compute the centre of mass given the mass of the parts already
        estimated and in self.part_masses
        '''
        total_weight = np.sum(self.part_masses)
        com = np.zeros(3,)
        #Compute the weighted average of the part centroids
        # with the weight being the part weight.
        for i,part in enumerate(self.parts):
            part_weight = self.part_masses[i]
            centroid = self.part_centroids[:,i]
            com += part_weight*centroid
        com = com / total_weight
        return com

    def compute_inertia_matrix(self):
        '''
        Compute the inertia matrix relative to the reference point used
        to define the points in the point-cloud.
        '''
        #Define a weight vector
        weights = np.ndarray((len(self.segmented_point_cloud),1))
        for i,part in enumerate(self.parts):
            part_weight = self.part_masses[i]
            points = part[:,0:3]
            seg_id = part[0,3]
            volumes= part[:,4]
            #Find all points sharing the same segmentation ID
            idx  = self.segmented_point_cloud[:,3] == seg_id
            #Compute the total volume of the part
            total_volume = np.sum(volumes)
            #Compute density of part
            mass_density = part_weight / total_volume
            #Assumes density homogeneity
            weight_per_point = volumes*mass_density
            weights[idx] = weight_per_point.reshape((len(weight_per_point),1))

        #Compute the inertia tensor components
        pos = self.segmented_point_cloud[:,0:3]
        I_xx = np.transpose(weights).dot((pos[:,1]**2 + pos[:,2]**2))
        I_yy = np.transpose(weights).dot((pos[:,0]**2 + pos[:,2]**2))
        I_zz = np.transpose(weights).dot((pos[:,0]**2 + pos[:,1]**2))
        I_xy = np.transpose(weights).dot((-pos[:,0] * pos[:,1]))
        I_xz = np.transpose(weights).dot((-pos[:,0] * pos[:,2]))
        I_yz = np.transpose(weights).dot((-pos[:,1] * pos[:,2]))

        I = np.array([[I_xx[0], I_xy[0], I_xz[0]],[I_xy[0], I_yy[0], I_yz[0]],[I_xz[0], I_yz[0], I_zz[0]]])
        return I