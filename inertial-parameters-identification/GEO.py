from time import time
import cvxpy as cp
import numpy as np
import scipy as sp
import mosek as mo 

#Method proposed in: T. Lee, P. M. Wensing and F. C. Park, "Geometric Robot Dynamic Identification: A Convex Programming Approach,"
#  in IEEE Transactions on Robotics, vol. 36, no. 2, pp. 348-365, April 2020.
#Python port of the MATLAB code provided at: https://github.com/alex07143/Geometric-Robot-DynID

class GEO:
    def __init__(self, distance_type, phi_prior, alpha, gamma=0, batch_size=1E10):

        self.DISTANCE_TYPE = distance_type
        self.REG_ALPHA     = alpha
        self.REG_GAMMA     = gamma
        self.BATCH_SIZE    = batch_size
        self.PHI_PRIOR     = phi_prior 

        #Acumulates data matrices
        self.stacked_data_matrices = np.ndarray((0,10))
        self.stacked_wrench_vectors= np.ndarray((0,1))
        self.information_vector    = np.ndarray((0,1))

        #Used to report result
        self.estimation_result = np.zeros((10,1))
        
        #Timestep counter
        self.k_step = 0

    #Pseudoinertia matrix in the body-fixed reference frame
    #Defined in: Wensing, P. M., Kim, S., & Slotine, J. J. E. (2017). Linear matrix inequalities for physically consistent inertial parameter identification: A statistical perspective on the mass distribution. IEEE Robotics and Automation Letters, 3(1), 60-67.
    #A necessary and sufficient condition for the physical consistency of the solution is the positive-definitedness of the pseudo-inertia matrix.
    #It ensures that the rotational inertia is SPD and that it also respects triangle inqualities.
    def pseudoInertia(self, mass, com, inertia):
        p = np.zeros((4,4))
        p[0:3,0:3]  = 0.5*np.trace(inertia)*np.identity(3) - inertia
        p[3,3]      = mass
        p[0:3,3]    = mass*com
        p[3,0:3]    = mass*com
        return p
    
    def phi2pseudoInertia(self, phi):
        p = phi
        return np.array([   [0.5*(p[4]+p[7]+p[9])-p[4], -p[5], -p[6], p[1]],
                            [-p[5], 0.5*(p[4]+p[7]+p[9])-p[7], -p[8], p[2]],
                            [-p[6], -p[8], 0.5*(p[4]+p[7]+p[9])-p[9], p[3]],
                            [p[1], p[2], p[3], p[0]]])

    def phi2params(self, phi):
        mass = phi[0]
        com  = phi[1:4]/mass
        inertia = np.array([[phi[4], phi[5], phi[6]],[phi[5], phi[7], phi[8]],[phi[6], phi[8], phi[9]]])
        return mass, com, inertia

    def params2phi(self, mass, com, inertia):
        phi = np.zeros((10,1))
        phi[0] = mass
        phi[1:4] = mass*com.reshape((3,1))
        phi[4] = inertia[0,0]
        phi[5] = inertia[0,1]
        phi[6] = inertia[0,2]
        phi[7] = inertia[1,1]
        phi[8] = inertia[1,2]
        phi[9] = inertia[2,2]
        return phi[:,0]

    #Matrix representation (pullback form) of the affine-invariant Riemannian metric defined on P(4) to R^10 coordinate and evaluated on phi
    def pullback_metric(self, phi):
        g = np.zeros((10,10))
        P_inv = np.linalg.inv(self.phi2pseudoInertia(phi))
        for i in range(0,10):
            for j in range(0,10):
                e_i = np.zeros((10,1))
                e_i[i] = 1
                e_j = np.zeros((10,1))
                e_j[j] = 1
                
                V_i = self.phi2pseudoInertia(e_i).reshape((4,4))
                V_j = self.phi2pseudoInertia(e_j).reshape((4,4))

                g[i,j] = np.trace(P_inv @ V_i @ P_inv @ V_j)
        return g

    #Matrix square root of the constant pullback metric 
    def pullback_metric_sqrt(self, phi):
        g = self.pullback_metric(phi)
        return sp.linalg.sqrtm(g)

    
    def update(self, g_s, acc_lin_s_s, vel_ang_s_s, acc_ang_s_s, forces_s, torques_s):
        #Update the steps counter
        self.k_step += 1

        #The way the data matrix is built, g_s must be inverted
        A_s = self.buildDataMatrix(-g_s, acc_lin_s_s, vel_ang_s_s, acc_ang_s_s)

        #Wrench matrix
        F_s = np.vstack((forces_s.reshape((3,1)), torques_s.reshape((3,1))))

        #Stack matrices.
        self.stacked_data_matrices = np.vstack((self.stacked_data_matrices, A_s))
        self.stacked_wrench_vectors= np.vstack((self.stacked_wrench_vectors, F_s))

        return self.estimation_result

    def estimate(self):
        start = time()
        #Shorthand
        A = self.stacked_data_matrices
        b = self.stacked_wrench_vectors

        #Respect the batch size
        lower = max([0,self.k_step-self.BATCH_SIZE])
        upper = self.k_step
        A = A[lower*6:,:]
        b = b[lower*6:,:]

        #Number of observations used
        N = int(b.shape[0]/6)
        #Variance of the observations [Fx,Fy,Fz,Tx,Ty,Tz]
        variances = np.diag([1,1,1,1,1,1])

        #Regularization weighting scalar gamma. Gamma should be chosen in function of
        #the uncertainty of the prior relative to the uncertainty of the measurements.
        #Its usually hand-tuned as the uncertainty in the prior is hard to measure.
        alpha = self.REG_ALPHA
        if self.REG_GAMMA > 0:
            gamma = self.REG_GAMMA
        else:
            #Uses the Kronecker tensor product to build a matrix with repetitions
            #of the variances along the diagonal.
            Sigma = np.kron(np.identity(N),variances)
            gamma = alpha * np.trace(A.T @ np.linalg.inv(Sigma) @ A)

        #Weight A and b based on the inverse of the covariance matrix (the information).
        #Inverting can become really expensive to compute as Sigma gets larger, so we record the
        # computed information value such that we dont compute it twice.
        information_vector = np.array([1/np.sqrt(var) for var in np.tile(np.diag(variances), int((A.shape[0] - self.information_vector.shape[0])/6))]).reshape(-1,1)
        self.information_vector = np.vstack((self.information_vector, information_vector))
        information = np.diag(self.information_vector.T[0])
        A = information @ A
        b = information @ b

        #Inverse of the pseudo-inertia matrix of the prior parameters
        pseudoinertia_prior = self.phi2pseudoInertia(self.PHI_PRIOR)
        pseudoinertia_prior_inv = np.linalg.inv(pseudoinertia_prior)

        #Optimization variables
        phi     = cp.Variable((10,1))
        P       = cp.Variable((4,4),PSD=True)
        constraints = [ P[0,0] == self.phi2pseudoInertia(phi)[0,0],
                        P[0,1] == self.phi2pseudoInertia(phi)[0,1],
                        P[0,2] == self.phi2pseudoInertia(phi)[0,2],
                        P[0,3] == self.phi2pseudoInertia(phi)[0,3],
                        P[1,0] == self.phi2pseudoInertia(phi)[1,0],
                        P[1,1] == self.phi2pseudoInertia(phi)[1,1],
                        P[1,2] == self.phi2pseudoInertia(phi)[1,2],
                        P[1,3] == self.phi2pseudoInertia(phi)[1,3],
                        P[2,0] == self.phi2pseudoInertia(phi)[2,0],
                        P[2,1] == self.phi2pseudoInertia(phi)[2,1],
                        P[2,2] == self.phi2pseudoInertia(phi)[2,2],
                        P[2,3] == self.phi2pseudoInertia(phi)[2,3],
                        P[3,0] == self.phi2pseudoInertia(phi)[3,0],
                        P[3,1] == self.phi2pseudoInertia(phi)[3,1],
                        P[3,2] == self.phi2pseudoInertia(phi)[3,2],
                        P[3,3] == self.phi2pseudoInertia(phi)[3,3]
                    ]
        #Objective function
        J_LS = cp.sum_squares(A @ phi - b)
        
        #Bregman divergence associated with a minus log-determinant
        if self.DISTANCE_TYPE == "Entropic":
            J_reg = -cp.log_det(P) + cp.trace(pseudoinertia_prior_inv @ P)
        
        #Constant pullback metric
        if self.DISTANCE_TYPE == "ConstantPullback":
            J_reg = cp.sum_squares( self.pullback_metric_sqrt(self.PHI_PRIOR) @ (phi - self.PHI_PRIOR.reshape((10,1)) ))
        
        #Euclidean metric
        if self.DISTANCE_TYPE == "Euclidean":
            J_reg = cp.sum_squares( phi - self.PHI_PRIOR.reshape((10,1)) )

        #Solve the problem
        obj  = cp.Minimize(J_LS + gamma * J_reg)
        prob = cp.Problem(obj, constraints)
        setup_time = time() - start
        prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={mo.iparam.intpnt_solve_form : mo.solveform.dual})
        #Returns the optimal value.
        self.estimation_result = phi.value

        #Get solver statistics
        solve_time = 0 if prob.solver_stats.solve_time is None else prob.solver_stats.solve_time
        total_time = setup_time + solve_time
        #print(total_time)

        return self.estimation_result
    
    #Build the data matrix that relates inertial parameters to forces and torques
    #in the sensor frame such that Ax=b where A is the data matrix, x is the vector
    #of inertial parameters and b is the wrench vector.
    # b = [Fx,Fy,Fz,Tx,Ty,Tz]
    # x = [m,m*Cx,m*Cy,m*Cz,Ixx,Ixy,Ixz,Iyy,Iyz,Izz]
    #Reference: Kubus 2008
    #Each argument is a 3-dim vector expressed in the sensor frame
    def buildDataMatrix(self, gravity, linear_acceleration, angular_velocity, angular_acceleration):
        g  = gravity
        la = linear_acceleration
        av = angular_velocity
        aa = angular_acceleration

        x = 0
        y = 1
        z = 2

        A = np.zeros((6,10))
        A[0,0:4] = [la[x]-g[x], -av[y]**2-av[z]**2, av[x]*av[y]-aa[z], av[x]*av[z]+aa[y]]
        A[1,0:4] = [la[y]-g[y], av[x]*av[y]+aa[z], -av[x]**2-av[z]**2, av[y]*av[z]-aa[x]]
        A[2,0:4] = [la[z]-g[z], av[x]*av[z]-aa[y], av[y]*av[z]+aa[x], -av[y]**2-av[x]**2]
        A[3,0:4] = [0, 0, la[z]-g[z], g[y]-la[y]]
        A[4,0:4] = [0, g[z]-la[z], 0, la[x]-g[x]]
        A[5,0:4] = [0, la[y]-g[y], g[x]-la[x], 0]
        A[3,4:10]= [aa[x], aa[y]-av[x]*av[z], aa[z]+av[x]*av[y], -av[y]*av[z], av[y]**2-av[z]**2, av[y]*av[z]]
        A[4,4:10]= [av[x]*av[z], aa[x]+av[y]*av[z], av[z]**2-av[x]**2, aa[y], aa[z]-av[x]*av[y], -av[x]*av[z]]
        A[5,4:10]= [-av[x]*av[y], av[x]**2-av[y]**2, aa[x]-av[y]*av[z], av[x]*av[y], aa[y]+av[x]*av[z], aa[z]]

        return A