'''
Loads and saves kinematics and dynamics data in Pickle files.
'''
from time import time
from threading import Thread, Event
import numpy as np
import pickle
from pathlib import Path
from spatialmath.spatialvector import SpatialVelocity, SpatialForce
from spatialmath import SE3, UnitQuaternion
from spatialmath.base import skew
import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped, AccelStamped, WrenchStamped
import with_respect_to as WRT

class Record:
    '''
    A record contains data about kinematics and dynamics
    relative to the world reference frame for one time step.

    - Force/Torque Sensor pose (SE3)
    - Force/Torque Sensor velocities (SpatialVelocity)
    - Force/Torque Sensor accelerations (6D)
    - Forces and Torques (SpatialForce)

    - Object pose (SE3)
    - Object velocities (SpatialVelocity)
    - Object accelerations (6D)

    - EE pose (SE3)
    - EE velocities (SpatialVelocity)
    - EE accelerations (6D)
    
    '''
    def __init__(self) -> None:
        self.timestamp = time()
        self.ft_pose = SE3()
        self.ft_vel  = SpatialVelocity()
        #NOTE: The concatenation of linear and angular accelerations
        # is NOT a spatial acceleration vector.
        # See: http://dx.doi.org/10.1109/MRA.2010.937853
        self.ft_acc  = np.zeros(6)
        self.ft_value= SpatialForce()

        self.ee_pose = SE3()
        self.ee_vel  = SpatialVelocity()
        self.ee_acc  = np.zeros(6)

        self.object_pose = SE3()
        self.object_vel  = SpatialVelocity()
        self.object_acc  = np.zeros(6)

    def add_Gaussian_noise(self, signal, mean=0, std_dev=1):
        '''
        Add Gaussian noise to a signal and return the result.
        '''
        noise = std_dev * np.random.standard_normal(signal.shape) + mean
        return signal + noise

    def add_zero_mean_Gaussian_noise(self, values, std_dev=[1,1]):
        '''
        Add zero-mean Gaussian noise to the linear and rotational parts of values
        with the standard deviations in std_dev, respectively.
        '''
        l = self.add_Gaussian_noise(values[0:3].reshape((3,1)), mean=0, std_dev=std_dev[0])
        a = self.add_Gaussian_noise(values[3:6].reshape((3,1)), mean=0, std_dev=std_dev[1])
        return np.vstack((l,a))

    def add_velocity_noise(self, std_dev=[1,1]):
        '''
        Add zero-mean Gaussian noise to the velocities.
        '''
        self.ft_vel = SpatialVelocity(self.add_zero_mean_Gaussian_noise(self.ft_vel.A, std_dev))
        self.ee_vel = SpatialVelocity(self.add_zero_mean_Gaussian_noise(self.ee_vel.A, std_dev))
        self.object_vel = SpatialVelocity(self.add_zero_mean_Gaussian_noise(self.object_vel.A, std_dev))
    
    def add_acceleration_noise(self, std_dev=[1,1]):
        '''
        Add zero-mean Gaussian noise to the accelerations.
        '''
        self.ft_acc = self.add_zero_mean_Gaussian_noise(self.ft_acc, std_dev)
        self.ee_acc = self.add_zero_mean_Gaussian_noise(self.ee_acc, std_dev)
        self.object_acc = self.add_zero_mean_Gaussian_noise(self.object_acc, std_dev)

    def add_wrench_noise(self, std_dev=[1,1]):
        '''
        Add zero-mean Gaussian noise to the wrench.
        '''
        self.ft_value = SpatialForce(self.add_zero_mean_Gaussian_noise(self.ft_value.A, std_dev))

    def expressed_in_sensor_frame(self):
        '''
        Expresses all values stored in the Record in the sensor frame
        and return data as a new Record.
        '''
        #Pose of ft-sensor wrt world
        X_WS_W = self.ft_pose
        X_SW_S = X_WS_W.inv()

        #Pose of ee wrt world
        X_WE_W = self.ee_pose

        #Pose of object-mesh wrt world
        X_WM_W = self.object_pose

        #Kinematics relative to world
        v_WM_W = self.object_vel
        a_WM_W = self.object_acc
        v_WS_W = self.ft_vel
        a_WS_W = self.ft_acc
        v_WE_W = self.ee_vel
        a_WE_W = self.ee_acc

        #Wrench measured by ft-sensor expressed in world
        w_S_W  = self.ft_value

        #Pose of ee wrt world expressed in ft-sensor
        X_WE_S = SE3.Rt(X_WE_W.R, t=X_SW_S.R @ X_WE_W.t)

        #Pose of object-mesh wrt world expressed in ft-sensor
        X_WM_S = SE3.Rt(X_WM_W.R, t=X_SW_S.R @ X_WM_W.t)

        #Pose of ft-sensor wrt world expressed in ft-sensor
        X_WS_S = SE3.Rt(X_WS_W.R, t=X_SW_S.R @ X_WS_W.t)

        #Jacobian that performs a change of coordinate frame to get 
        # kinematics expressed in ft-sensor from the one expressed in world
        jac_SW = np.block([ [X_SW_S.R, np.zeros((3,3))],
                            [np.zeros((3,3)), X_SW_S.R]])

        #Velocity of ee wrt world expressed in ft-sensor
        v_WE_S = jac_SW @ v_WE_W
        #Acceleration of ee wrt world expressed in ft-sensor
        a_WE_S = jac_SW @ a_WE_W

        #Velocity of object-mesh wrt world expressed in ft-sensor
        v_WM_S = jac_SW @ v_WM_W
        #Acceleration of object-mesh wrt world expressed in ft-sensor
        a_WM_S = jac_SW @ a_WM_W

        #Velocity of ft-sensor wrt world expressed in ft-sensor
        v_WS_S = jac_SW @ v_WS_W
        #Acceleration of ft-sensor wrt world expressed in ft-sensor
        a_WS_S = jac_SW @ a_WS_W

        #Wrench sensed at ft-sensor expressed in ft-sensor
        # Simple here because there is no lever arm effect.
        # See Corke p. 186 and Modern Robotics p. 93
        w_S_S  = jac_SW @ w_S_W

        #Store new data in a Record
        rec = Record()
        rec.ft_pose = X_WS_S
        rec.ft_vel  = SpatialVelocity(v_WS_S)
        rec.ft_acc  = a_WS_S
        rec.ft_value= SpatialForce(w_S_S)
        rec.object_pose= X_WM_S
        rec.object_vel = SpatialVelocity(v_WM_S)
        rec.object_acc = a_WM_S
        rec.ee_pose   = X_WE_S
        rec.ee_vel    = SpatialVelocity(v_WE_S)
        rec.ee_acc    = a_WE_S

        return rec

class RosRecorder(Thread):
    '''
    Listen to  specific topics and record their content into Records.
    '''
    def __init__(self, world_name='xarm-id-traj', ee_name='ee', ft_name='sensor') -> None:
        '''
        world_name is the with_respect_to world name in which the pose
        of the object and force/torque sensor are defined relative to.

        ee_name: name of the end-effector frame in world_name. The position of this frame
                    should be on the flange of the robot.
        ft_name: name of the force/torque sensor frame in world_name.
        '''
        Thread.__init__(self, group=None, target=None, name=None, args=None, kwargs=None, daemon=True)
        self._stop_event = Event()
        if rosgraph.is_master_online():
            #Initialize the ROS node
            #Setting the queue_size=1 make sure one Subscriber does not accumulate delays compared to a faster one
            rospy.init_node('xArm_ROS_Kinematics', anonymous=True)
            self.sub_ee_pose = rospy.Subscriber('EEPose', PoseStamped, self.new_ee_pose, queue_size=1)
            self.sub_ee_vel = rospy.Subscriber('EEVelocity', AccelStamped, self.new_ee_vel, queue_size=1)
            self.sub_ee_acc = rospy.Subscriber('EEAcceleration', AccelStamped, self.new_ee_acc, queue_size=1)
            self.sub_ft_val = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.new_ft_val, queue_size=1)
        else:
            #ROS is not up
            print('ROS does not seems to be up. Closing KinoDynamic ROS Recorder.')
            self.join()
        
        db = WRT.DbConnector()
        self.world_name = world_name
        self.ee_name    = ee_name
        self.ft_name    = ft_name

        try:
            #Pose of the Force/Torque sensor with respect to the end-effector and expressed in the end-effector
            self.X_ES_E = SE3(db.In(world_name).Get(ft_name).Wrt(ee_name).Ei(ee_name))
            #TODO: Remove below when not needed anymore
            if world_name == 'xarm-id-traj':
                #Use meters instead of millimeters
                self.X_ES_E.t /= 1000
        except RuntimeError as e:
            print(str(e)+'Closing KinoDynamic ROS Recorder.')
            self.join()

        #Instanciate a Loader that will be used to set and save Records.
        self._loader = Loader()
        #Initialize temporary memory slots
        self.reset_data()

    def reset_data(self):
        '''
        Set all temporary memories to None.
        Each memory slot is a tuple where the [0] value is a timestamp
        and the [1] value is the pose/vel/acc/wrench
        '''
        self.ft_val  = None
        self.ee_pose = None
        self.ee_vel  = None
        self.ee_acc  = None
        self.ft_pose = None
        self.ft_vel  = None
        self.ft_acc  = None 

    def stop(self):
        self._stop_event.set()
    def stopped(self):
        return self._stop_event.is_set()
    def join(self):
        Thread.join(self)
        return None
    def run(self):
        while not self.stopped() and not rospy.is_shutdown(): 
            # Check if we have a complete Record (ee_pose, ee_vel, ee_acc, ft_val)
            #If so, then verify that all timestamps are within the same timeframe.
            #If so, then make a new Record with Loader and then set all data to None
            '''
            s = [self.ee_pose, self.ee_vel, self.ee_acc, self.ft_pose, self.ft_vel, self.ft_acc, self.ft_val]
            tstamps = [ss[0].to_sec() for ss in s if ss is not None]
            if len(tstamps) == len(s):
                mini = min(tstamps)
                deltas = [ss[0].to_sec()-mini for ss in s if ss is not None]
                print(deltas)
            else:
                print(len(tstamps))
            #else:
            #    print([i for i,ss in enumerate(s) if ss is None])
            '''
            

            if self.recent(self.ee_pose, self.ee_vel, self.ee_acc, self.ft_pose, self.ft_vel, self.ft_acc, self.ft_val):
                #Build Record
                rec = Record()
                rec.timestamp  = self.ee_pose[0].to_time()
                rec.ee_pose    = self.ee_pose[1]
                rec.ee_vel     = SpatialVelocity(self.ee_vel[1])
                rec.ee_acc     = self.ee_acc[1]
                rec.ft_pose     = self.ft_pose[1]
                rec.ft_vel      = SpatialVelocity(self.ft_vel[1])
                rec.ft_acc      = self.ft_acc[1]
                rec.ft_value    = SpatialForce(self.ft_val[1])
                #Append Record
                self._loader.setRecord(rec)
                print('Record added')
                #Reset memory slots
                self.reset_data()

    def recent(self, *args, max_timediff=1/50):
        '''
        Return True if all the arguments have a timestamp that is within max_timediff seconds
        of the other timestamps.
        '''
        timestamps = []
        for a in args:
            if a is None:
                return False
            else:
                timestamps.append(a[0].to_sec())
        timediff = max(timestamps) - min(timestamps)
        if timediff > max_timediff:
            if len(args) > 6:
                print(timediff)
            return False
        else:
            return True

    def propagate_rigid_body_velocity(self, X_WA_W:SE3, X_WB_W:SE3, v_WA_W:SpatialVelocity):
        '''
        From the velocity of a point A on a rigid body, and the transform between
        the pose of A and the one of a second point B, compute the velocity
        of B from the one of A.
        X_WA_W: Pose of A wrt world expressed in world
        X_WB_W: Pose of B wrt world expressed in world
        v_WA_W: Velocity of A wrt world expressed in world
        NOTE: All points on a rigid body have the same angular velocity and
              angular acceleration. However, if a rigid-body is rotating, 
              not all points have the same linear velocity and linear acceleration.
        '''
        #Pose of B wrt A expressed in A
        X_AB_A = X_WA_W.inv() @ X_WB_W
        #Assumes that B does not move relative to A (both points are on the same rigid body)
        lv_AB_A, av_AB_A = [np.zeros((3,1)),np.zeros((3,1))]
        #Linear and angular velocities of A wrt world expressed in world
        lv_WA_W = v_WA_W.A[0:3]
        av_WA_W = v_WA_W.A[3:6]
        #Linear velocity of B relative to world
        lv_WB_W = lv_WA_W + (X_WB_W.R @ lv_AB_A).reshape((3,)) + skew(av_WA_W) @ (X_WA_W.R @ X_AB_A.t)
        #Angular velocity of B relative to world
        av_WB_W = av_WA_W + (X_WA_W.R @ av_AB_A).reshape((3,))
        #Build a spatial velocity vector
        v_WB_W = np.hstack((lv_WB_W, av_WB_W))
        return v_WB_W
    
    def propagate_rigid_body_acceleration(self, X_WA_W:SE3, X_WB_W:SE3, v_WA_W:SpatialVelocity, v_WB_W:SpatialVelocity, a_WA_W):
        '''
        From the acceleration of a point A on a rigid body, and the transform between
        the pose of A and the one of a second point B, compute the acceleration
        of B from the one of A.
        X_WA_W: Pose of A wrt world expressed in world
        X_WB_W: Pose of B wrt world expressed in world
        v_WA_W: Velocity of A wrt world expressed in world
        v_WB_W: Velocity of A wrt world expressed in world
        a_WA_W: Acceleration of A wrt world expressed in world
        NOTE: All points on a rigid body have the same angular velocity and
              angular acceleration. However, if a rigid-body is rotating, 
              not all points have the same linear velocity and linear acceleration.
        '''
        #Pose of B wrt A expressed in A
        X_AB_A = X_WA_W.inv() @ X_WB_W
        #Assumes that B does not move relative to A (both points are on the same rigid body)
        lv_AB_A, av_AB_A, la_AB_A, aa_AB_A = [np.zeros((3,1)) for i in range(4)]
        #Linear and angular velocity of A wrt world expressed in world
        lv_WA_W = v_WA_W.A[0:3]
        av_WA_W = v_WA_W.A[3:6]
        #Linear and angular velocity of B wrt world expressed in world
        lv_WB_W = v_WB_W.A[0:3]
        av_WB_W = v_WB_W.A[3:6]
        #Linear and angular acceleration of A wrt world expressed in world
        la_WA_W = a_WA_W[0:3].reshape((3,1))
        aa_WA_W = a_WA_W[3:6].reshape((3,1))
        #Linear acceleration of B relative to world
        la_WB_W = la_WA_W +\
                    X_WB_W.R @ la_AB_A +\
                    skew(av_WB_W) @ (X_WB_W.R @ lv_AB_A) +\
                    skew(aa_WA_W) @ (X_WA_W.R @ X_AB_A.t.reshape((3,1))) +\
                    skew(av_WA_W) @ (X_WA_W.R @ lv_AB_A + skew(av_WA_W) @ (X_WA_W.R @ X_AB_A.t.reshape((3,1))))
        #Angular acceleration of B relative to world
        aa_WB_W = aa_WA_W +\
                X_WA_W.R @ aa_AB_A +\
                skew(av_WA_W) @ (X_WA_W.R @ av_AB_A)
        #Build an acceleration vector
        a_WB_W = np.hstack((la_WB_W.reshape((3,)), aa_WB_W.reshape((3,))))
        return a_WB_W

    def new_ee_pose(self, msg:PoseStamped):
        '''
        Compute ft-sensor pose from ee pose.
        '''
        timestamp = msg.header.stamp
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        ox = msg.pose.orientation.x
        oy = msg.pose.orientation.y
        oz = msg.pose.orientation.z
        ow = msg.pose.orientation.w
        q = UnitQuaternion(ow, [ox,oy,oz])
        X_WE_W = SE3.Rt(R=q.SO3(), t=[px,py,pz])
        X_WS_W = X_WE_W @ self.X_ES_E

        #print('EEPose: {}'.format(msg.header.seq))

        self.ft_pose = (timestamp, X_WS_W)
        self.ee_pose = (timestamp, X_WE_W)

        #If a recent ee velocity is available, then it's now possible
        # to compute ft velocity as well. This happens if the velocity ROS
        # message is received before the pose message.
        if self.recent(self.ee_pose, self.ee_vel):
            msg = AccelStamped()
            msg.header.stamp    = self.ee_vel[0]
            msg.accel.linear.x  = self.ee_vel[1][0]
            msg.accel.linear.y  = self.ee_vel[1][1]
            msg.accel.linear.z  = self.ee_vel[1][2]
            msg.accel.angular.x = self.ee_vel[1][3]
            msg.accel.angular.y = self.ee_vel[1][4]
            msg.accel.angular.z = self.ee_vel[1][5]
            self.new_ee_vel(msg)

    def new_ee_vel(self, msg:AccelStamped):
        '''
        Compute ft-sensor velocity from ee velocity.
        '''
        timestamp = msg.header.stamp
        lv_x = msg.accel.linear.x
        lv_y = msg.accel.linear.y
        lv_z = msg.accel.linear.z
        av_x = msg.accel.angular.x
        av_y = msg.accel.angular.y
        av_z = msg.accel.angular.z
        self.ee_vel = (timestamp, np.array([lv_x,lv_y,lv_z,av_x,av_y,av_z]))

        #print('EEVel: {}'.format(msg.header.seq))

        if self.recent(self.ee_pose, self.ft_pose):
            X_WE_W = self.ee_pose[1]
            X_WS_W = self.ft_pose[1]
            v_WE_W = SpatialVelocity([lv_x,lv_y,lv_z,av_x,av_y,av_z])
            v_WS_W = self.propagate_rigid_body_velocity(X_WE_W, X_WS_W, v_WE_W)
            self.ft_vel = (timestamp, v_WS_W)

        #If a recent ee acceleration is available, then it's now possible
        # to compute ft acceleration as well. This happens if the acceleration ROS
        # message is received before the velocity message.
        if self.recent(self.ee_vel, self.ee_acc):
            msg = AccelStamped()
            msg.header.stamp    = self.ee_acc[0]
            msg.accel.linear.x  = self.ee_acc[1][0]
            msg.accel.linear.y  = self.ee_acc[1][1]
            msg.accel.linear.z  = self.ee_acc[1][2]
            msg.accel.angular.x = self.ee_acc[1][3]
            msg.accel.angular.y = self.ee_acc[1][4]
            msg.accel.angular.z = self.ee_acc[1][5]
            self.new_ee_acc(msg)

    def new_ee_acc(self, msg:AccelStamped):
        '''
        Compute ft-sensor acceleration from ee acceleration.
        '''
        timestamp = msg.header.stamp
        la_x = msg.accel.linear.x
        la_y = msg.accel.linear.y
        la_z = msg.accel.linear.z
        aa_x = msg.accel.angular.x
        aa_y = msg.accel.angular.y
        aa_z = msg.accel.angular.z
        self.ee_acc = (timestamp, np.array([la_x,la_y,la_z,aa_x,aa_y,aa_z]))

        if self.recent(self.ee_pose, self.ft_pose, self.ee_vel, self.ft_vel):
            X_WE_W = self.ee_pose[1]
            X_WS_W = self.ft_pose[1]
            v_WE_W = SpatialVelocity(self.ee_vel[1])
            a_WE_W = self.ee_acc[1]
            v_WS_W = SpatialVelocity(self.ft_vel[1])
            a_WS_W = self.propagate_rigid_body_acceleration(X_WE_W, X_WS_W, v_WE_W, v_WS_W, a_WE_W)
            self.ft_acc = (timestamp, a_WS_W)
        else:
            s = [self.ee_pose, self.ft_pose, self.ee_vel, self.ft_vel]
            print([i for i,ss in enumerate(s) if ss is None])
            #print([ss[0].to_sec() for ss in s if ss is not None])
            #print('EEAcc: {}'.format(abs(self.ee_pose[0].to_sec()-self.ee_vel[0].to_sec())))

    def new_ft_val(self, msg:WrenchStamped):
        timestamp = msg.header.stamp
        fx = msg.wrench.force.x 
        fy = msg.wrench.force.y 
        fz = msg.wrench.force.z 
        tx = msg.wrench.torque.x
        ty = msg.wrench.torque.y
        tz = msg.wrench.torque.z
        self.ft_val = (timestamp, np.array([fx,fy,fz,tx,ty,tz]))

class Loader:
    def __init__(self) -> None:
        '''
        Initialize data structures.
        '''
        self.records = []
        self.index_of_next_record = 0

    def load(self, input_pkl_file_path:Path) -> bool:
        '''
        Load the Pickle file with its Path specified in input_pkl_file_path and
        store its content. Return True if success, False otherwise.
        '''
        if input_pkl_file_path.exists() and not input_pkl_file_path.is_dir():
            path = input_pkl_file_path.resolve().as_posix()
            with open(path, "rb") as input_file:
                self.records = pickle.load(input_file)
                input_file.close()
                return True
        else:
            return False

    def save(self, output_pkl_file_path:Path) -> bool:
        '''
        Save the records in the file with its Path specified in 
        output_pkl_file_path. If the parent directories in the Path
        do not exist, try to create them. Return True if success, False otherwise.
        '''
        try:
            output_pkl_file_path.parent.mkdir(parents=True, exist_ok=True)
            path = output_pkl_file_path.resolve().as_posix()
            with open(path, "wb") as output_file:
                self.records = pickle.dump(self.records, output_file, pickle.HIGHEST_PROTOCOL)
                output_file.close()
                return True
        except:
            return False

    def getRecord(self, Nth:int=None) -> Record:
        if Nth is None:
            #If index is not given, assume the user wants
            # the following record.
            Nth = self.index_of_next_record
            self.index_of_next_record += 1
        if Nth < len(self.records) and Nth >= 0:
            return self.records[Nth]
        else:
            return None

    def setRecord(self, record:Record, Nth:int=None):
        if Nth is None:
            #Append a new record
            self.records.append(record)
        elif Nth <= len(self.records) and Nth >= 0:
            #Replace an existing record
            self.records[Nth] = record
