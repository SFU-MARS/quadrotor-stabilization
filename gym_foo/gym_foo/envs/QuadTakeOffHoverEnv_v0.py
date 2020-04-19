# Required general libraries
import numpy as np
import rospy
import time
import gym
from gym import spaces
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import gazebo_env
from baselines import logger

# Required ROS msgs
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Imu
from mav_msgs.msg import Actuators
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

# Required ROS services
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState


# ttr Engine for the use of TTR reward
from ttr_engine.ttr_helper import ttr_helper


class QuadTakeOffHoverEnv_v0(gazebo_env.GazeboEnv):
    def __init__(self, rew, **kwargs):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "crazyflie2_without_controller.launch")

        # --- Max episode steps ---
        self.max_steps = 300
        self.step_counter = 0

        # --- Take off and Hover identifier --
        self.isTakeoff = True
        self.isHover   = False

        # --- Specification of TTR reward ---
        if rew == 'ttr':
            self.ttr_helper = ttr_helper()
            self.ttr_helper.setup()


        # --- Specification of maximum motor speed, from crazyflie manual ---
        self.max_motor_speed = 2618

        # --- Specification of target ---
        self.target_height = 0.5
        self.target_takeoff_vel = np.array([0, 0, 1])
        self.target_hover_vel = np.array([0, 0, 0])

        # rospy.init_node("QuadTakeOffHover", anonymous=True, log_level=rospy.INFO)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.enable_motor = rospy.Publisher('/crazyflie2/command/motor_speed', Actuators, queue_size=1)



    def reset(self, reset_args=None):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
            # initialize pose position
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = 0, 0, 0.015

            # initialize pose orientation
            # Note: gazebo rotation order: roll, pitch, yaw
            # the angle w.r.t x-axis: roll in gazebo, [-np.pi/2, np.pi/2]
            # the angle w.r.t y-axis: pitch in gazebo, [-np.pi/2, np.pi/2]
            # the angle w.r.t z-axis: yaw in gazebo, [0, 2*np.pi]
            roll, pitch, yaw = 0, 0, np.random.uniform(0, 2*np.pi)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_euler(roll, pitch, yaw)

            # initialize twist
            # the reset value should be referring to some empirical value
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.linear.z = 0, 0, 0
            twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, 0

            reset_state = ModelState()
            reset_state.model_name = "crazyflie2"
            reset_state.pose = pose
            reset_state.twist = twist
            self.set_model_state(reset_state)
        except rospy.ServiceException as e:
            print("# /gazebo/reset_simulation call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # read observation
        Pose_data = None
        Imu_data  = None
        Odom_data = None
        while Pose_data is None or Imu_data is None or Odom_data is None:
            Pose_data = rospy.wait_for_message("/crazyflie2/pose_with_covariance", PoseWithCovarianceStamped, timeout=5)
            time.sleep(0.01)
            Imu_data  = rospy.wait_for_message('/crazyflie2/ground_truth/imu', Imu, timeout=5)
            time.sleep(0.01)
            # Odom_data = rospy.wait_for_message('/crazyflie2/ground_truth/odometry', Odometry, timeout=5)
            Odom_data = self.get_model_state(model_name="crazyflie2")
            time.sleep(0.01)

        # Pause simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed!")

        obsrv = self.get_obsrv(Pose_data, Imu_data, Odom_data)
        self.pre_obsrv = obsrv

        # --- reset these ---
        self.isTakeoff = True
        self.isHover = False
        self.step_counter = 0

        return obsrv

    def get_obsrv(self, Pose_data, Imu_data, Odom_data):
        # we don't include any state variables w.r.t 2D positions (x,y)
        # valid state variables: [z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w]

        # Hight to ground
        self.x = Pose_data.pose.pose.position.x
        self.y = Pose_data.pose.pose.position.y
        z = Pose_data.pose.pose.position.z

        # linear velocities
        # vx = Odom_data.twist.twist.linear.x
        # vy = Odom_data.twist.twist.linear.y
        # vz = Odom_data.twist.twist.linear.z
        vx = Odom_data.twist.linear.x
        vy = Odom_data.twist.linear.y
        vz = Odom_data.twist.linear.z


        # roll, pitch, yaw
        roll, pitch, yaw = euler_from_quaternion([Imu_data.orientation.x, Imu_data.orientation.y, Imu_data.orientation.z, Imu_data.orientation.w])

        # angular velocities of roll, pitch, yaw
        roll_w, pitch_w, yaw_w = Imu_data.angular_velocity.x, Imu_data.angular_velocity.y, Imu_data.angular_velocity.z

        # print(np.array([z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w]))

        return np.array([z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w])

    def step(self, action):
        # action is 4-dims representing drone's four motor speeds
        action = np.asarray(action)

        # --- check if the output of policy network is nan ---
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # --- transform action from network output into environment limit, use ref=[-2,2] or [-1,1]---
        ref_action = spaces.Box(low=-1, high=1, shape=(4,))
        action = np.clip(action, ref_action.low, ref_action.high)
        env_action = self.action_space.low + (self.action_space.high - self.action_space.low) * (action - ref_action.low) * 1.0 / (ref_action.high - ref_action.low)
        clipped_env_ac = np.clip(env_action.copy(), self.action_space.low, self.action_space.high)
        # print("real action:", clipped_env_ac)

        # --- apply motor speed to crazyflie ---
        cmd_msg = Actuators()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.angular_velocities = clipped_env_ac
        self.enable_motor.publish(cmd_msg)

        # --- run simulator to collect data ---
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        Pose_data = None
        Imu_data = None
        Odom_data = None
        while Pose_data is None or Imu_data is None or Odom_data is None:
            Pose_data = rospy.wait_for_message("/crazyflie2/pose_with_covariance", PoseWithCovarianceStamped)
            time.sleep(0.01)
            Imu_data = rospy.wait_for_message('/crazyflie2/ground_truth/imu', Imu)
            time.sleep(0.01)
            # Odom_data = rospy.wait_for_message('/crazyflie2/ground_truth/odometry', Odometry)
            Odom_data = self.get_model_state(model_name='crazyflie2')
            time.sleep(0.01)

        # --- pause simulator to process data ---
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # --- deal with obsrv and reward ---
        obsrv = self.get_obsrv(Pose_data, Imu_data, Odom_data)
        self.pre_obsrv = obsrv

        reward = 0
        done = False
        suc  = False
        self.step_counter += 1

        # --- determine reward-to-go for possible situations, using TTR for takeoff only
        if self.isTakeoff:
            reward += -10.0 * self.ttr_helper.interp(obsrv) - 2.0 * np.abs(self.target_takeoff_vel[2] - obsrv[3])
            if self.step_counter > 30:
                reward += -200
                logger.log("Failed to take off!")
                done = True
            elif obsrv[0] >= self.target_height:
                reward += 400
                logger.log("good to take off and ready to hover!")
                self.isTakeoff = False
                self.isHover = True
        elif self.isHover:
            reward = -10.0 * self.ttr_helper.interp(obsrv)
            if self.pre_obsrv[0] < 0.2:
                logger.log('clash in area too low, failed')
                reward -= 200.0  # penalty clash
                done = True
            elif self.step_counter > 50:
                reward += 400
                suc = True
                logger.log("good to hover for a while !!")


        # --- determine reward-to-go for possible situations (oldest version)
        # if self.isTakeoff:
        #     reward = -(1.5 * np.linalg.norm(self.target_height - self.pre_obsrv[0]) + 0.6 * np.linalg.norm(self.target_takeoff_vel-self.pre_obsrv[1:4]))
        #     if self.step_counter > 20:
        #         reward += -50
        #         logger.log("Failed to take off!")
        #         done = True
        #     elif self.pre_obsrv[0] >= self.target_height:
        #         reward += 50
        #         logger.log("good to take off and ready to hover!")
        #         self.isTakeoff = False
        #         self.isHover = True
        # elif self.isHover:
        #     reward = -(1.5 * np.linalg.norm(self.target_height - self.pre_obsrv[0]) + 0.8 * np.linalg.norm(self.target_hover_vel - self.pre_obsrv[1:4]))
        #     if self.pre_obsrv[0] < 0.2:
        #         logger.log('Clash Last for : {} steps, failed'.format(self.step_counter))
        #         reward -= 100.0  # penalty clash
        #         done = True
        #     elif self.step_counter > 50:
        #         reward += 100
        #         suc = True
        #         logger.log("good to hover for {} steps!!".format(self.step_counter))

        if self.in_collision(self.pre_obsrv):
            reward += -200
            done = True

        if self.step_counter >= self.max_steps:
            done = True

        return np.array(np.copy(obsrv)), reward, done, {'suc':suc}

    def in_collision(self, obsrv):
        if self.x >=2 or self.x <= -2 or self.y >= 2 or self.y <= -2:
            print("in collision, x and y out of range!")
            return True
        elif obsrv[4] >= np.pi/2:
            print("in collision, roll angle out of range")
            return True
        else:
            return False

    # observation space: [z,vx,vy,vz,roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate]
    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(10,))

    @property
    def action_space(self):
        return spaces.Box(low=2000, high=self.max_motor_speed, shape=(4,))


if __name__ == "__main__":
    env = QuadTakeOffHoverEnv_v0()
    obs = env.reset()
    while True:
        print("obs:", obs)
        if obs[0] > 1:
            obs = env.reset()
        obs, _, _, _ = env.step([0.9, 0.9, 0.9, 0.9])