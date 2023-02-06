import cvxpy as cp
import numpy as np
import mosek as mo 

#Method proposed in: Atkeson, C. G., An, C. H., & Hollerbach, J. M. (1986). Estimation of Inertial Parameters of Manipulator Loads and Links.
#The International Journal of Robotics Research, 5(3), 101–119. 

class OLS:
    def __init__(self, batch_size=1E10):
        self.BATCH_SIZE = batch_size

        #Acumulates data matrices
        self.stacked_data_matrices = np.ndarray((0,10))
        self.stacked_wrench_vectors= np.ndarray((0,1))

        #Used to report result
        self.estimation_result = np.zeros((10,1))
        
        #Timestep counter
        self.k_step = 0
    
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

        #Shorthand
        A = self.stacked_data_matrices
        b = self.stacked_wrench_vectors

        #Respect the batch size
        lower = max([0,self.k_step-self.BATCH_SIZE])
        upper = self.k_step
        A = A[lower*6:,:]
        b = b[lower*6:,:]

        #Optimization variables
        phi     = cp.Variable((10,1))

        #Objective function
        J_LS = cp.sum_squares(A @ phi - b)

        #Solve the problem
        obj  = cp.Minimize(J_LS)
        prob = cp.Problem(obj)
        prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={mo.iparam.intpnt_solve_form : mo.solveform.dual})
        
        #Returns the optimal value.
        self.estimation_result = phi.value

        #Get solver statistics
        setup_time = 0 if prob.solver_stats.setup_time is None else prob.solver_stats.setup_time
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
    
    def compute_condition_number(self):
        '''
        Compute the condition number of the regressor matrix.
        A condition number of 1 is perfectly well-conditioned and a condition
        number much greater means that the regressor matrix is ill-conditioned
        meaning that a small noise will have a large impact on the identification.
        '''
        #Scale the regressor according to Gautier & Khalil 1992
        reshaped_regressor = self.stacked_data_matrices.reshape((-1,6,10))
        dividend  = reshaped_regressor - np.min(reshaped_regressor,axis=0)
        divisor = np.max(reshaped_regressor,axis=0) - np.min(reshaped_regressor,axis=0)
        scaled_regressor   = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)
        reshaped_scaled_regressor = scaled_regressor.reshape((-1,10))
        u, s, vh = np.linalg.svd(scaled_regressor, full_matrices=False)
        #Condition number is the ratio of the largest singular value to the smallest
        condition_number = np.max(s)/np.min(s)
        return condition_number