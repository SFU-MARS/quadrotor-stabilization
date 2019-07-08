
import numpy as np

import rospy
import time
import gym
from gym import spaces

from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ContactsState

from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench

from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import ApplyBodyWrench
from tf.transformations import euler_from_quaternion, quaternion_from_euler


from utils import board_to_world, init_quad, apply_wrench_to_quad

from brs_engine.FullQuad_brs_engine import *


class QuadFallingDownEnv_v0(gym.Env):
    def __init__(self, **kwargs):
        # Note: need to be compatible with the quadrotor.urdf. Here I use crazyflie params
        self.mass = 0.027
        self.weight = self.mass * 9.81
        self.max_lift = 0.058 * 9.81

        # self.step_counter = 0
        # self.max_steps = 100

        self.collision_reward = -400
        self.goal_reward = 1000

        self.reward_type = kwargs['reward_type']
        self.brs_engine = None
        if self.reward_type == 'ttr':
            self.brs_engine = FullQuad_brs_engine()
        print(self.reward_type)
        print(self.brs_engine)

        self.high_level_goal = False

        rospy.init_node("quad_falling_down", anonymous=True, log_level=rospy.INFO)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

    def reset(self, reset_args=None):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
            # initialize pose
            pose = Pose()
            pose.position.x, pose.position.y = np.random.uniform(low=-2, high=2), np.random.uniform(low=-2, high=2)
            pose.position.z = np.random.uniform(3,7)

            # Note: gazebo rotation order: roll, pitch, yaw
            # the angle w.r.t x-axis: roll in gazebo
            # roll = np.random.uniform(-np.pi/2, np.pi/2)
            roll = 0
            # the angle w.r.t y-axis: pitch in gazebo
            # pitch = np.random.uniform(-np.pi/2, np.pi/2)
            # pitch = np.random.uniform(-0.1 + np.pi/4, 0.1 + np.pi/4)
            pitch = np.pi/4
            
            # the angle w.r.t z-axis: yaw in gazebo
            # yaw = np.random.uniform(0, 2*np.pi)
            yaw = 0
            ox, oy, oz, ow = quaternion_from_euler(roll, pitch, yaw)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = ox,oy,oz,ow

            # initialize twist
            # the reset value should be referring to some empirical value
            # vx =  np.random.uniform(-1,1)
            # vy =  np.random.uniform(-1,1)
            # vz =  np.random.uniform(-1,1)
            vx = 0.4
            vy = 0.4
            vz = 0.4
            roll_w = np.random.uniform(-np.pi/6, np.pi/6)
            pitch_w = np.random.uniform(-np.pi/6, np.pi/6)
            yaw_w = np.random.uniform(-np.pi/6, np.pi/6)
            # roll_w = 0
            # pitch_w = 0
            # yaw_w = 0

            twist = Twist()
            twist.linear.x, twist.linear.y, twist.linear.z =  vx, vy, vz
            twist.angular.x, twist.angular.y, twist.angular.z = roll_w, pitch_w, yaw_w

            reset_state = ModelState()
            reset_state.model_name = "quadrotor"
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
        dynamic_data = None
        while dynamic_data is None:
            rospy.wait_for_service("/gazebo/get_model_state")
            try:
                dynamic_data = self.get_model_state(model_name="quadrotor")
            except rospy.ServiceException as e:
                print("/gazebo/unpause_physics service call failed")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed!")

        obsrv = self.get_obsrv(dynamic_data)
        self.pre_obsrv = obsrv

        return obsrv


    def step(self, action):
        # action is 4-dims representing drone's four thrusts
        # --- check if the output of policy network is nan --- 
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))
        
        # --- transform action from network output into environment limit, use ref=[-2,2] ---
        # print("action before clipped:", action)
        ref_action = spaces.Box(low=-1, high=1, shape=(4,))
        env_action = self.action_space.low + (self.action_space.high - self.action_space.low) * (action - ref_action.low) * 1.0 / (ref_action.high - ref_action.low)
        # print("env_action:", env_action)
        clipped_env_ac = np.clip(env_action.copy(), self.action_space.low, self.action_space.high) 
        
        # print("action after clipped:", clipped_env_ac)
        

        # --- apply action to quadcoptor ---
        pre_roll = self.pre_obsrv[4]
        pre_pitch = self.pre_obsrv[5]
        pre_yaw = self.pre_obsrv[6]

        pre_roll_w = self.pre_obsrv[7]
        pre_pitch_w = self.pre_obsrv[8]

        # print("roll_w:", pre_roll_w)
        # print("pitch_w:", pre_pitch_w)

        rospy.wait_for_service('/gazebo/apply_body_wrench')
        _ = apply_wrench_to_quad(self.apply_wrench, clipped_env_ac, pre_roll, pre_pitch, pre_yaw)

        # --- run simulator to collect data ---
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        dynamic_data = None
        contact_data = None
        while dynamic_data is None and contact_data is None:
            rospy.wait_for_service('/gazebo/get_model_state')
            try:
                contact_data = rospy.wait_for_message('/gazebo_ros_bumper', ContactsState, timeout=50)
                dynamic_data = self.get_model_state(model_name="quadrotor")
            except rospy.ServiceException as e:
                print("/gazebo/get_model_state service call failed!")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")
        
        # --- deal wiht obsrv and reward assignment ---
        obsrv = self.get_obsrv(dynamic_data)
        # print("obsrv:", obsrv)
        self.pre_obsrv = obsrv

        reward = 0
        # put penalty on invalid action
        # reward += -np.sum(np.abs(action - clipped_env_ac))

        if self.reward_type == 'sparse':
            reward += 0
        elif self.reward_type == 'ttr':
            ttr = self.brs_engine.evaluate_ttr(obsrv)
            reward += -ttr
        else:
            pass

        # --- additional rewards on angular velocity ---
        if -0.17 <= pre_roll_w <= 0.17 and -0.17 <= pre_pitch_w <= 0.17:
            reward += 0
        else:
            tmp_reward_w = -1.0 * (abs(pre_roll_w - (-0.17)) + abs(pre_roll_w - 0.17) + abs(pre_pitch_w - (-0.17)) + abs(pre_pitch_w - 0.17))
            # print("penalty only for angular velocity:", tmp_reward_w)
            reward += tmp_reward_w
        # ----------------------------------------------

        done = False
        suc  = False

        if self.in_obst(contact_data):
            reward += self.collision_reward
            done = True

        # --- one-level reward setting ---
        # if self.in_goal(obsrv):
            # reward += self.goal_reward
            # done = True
            # suc  = True
        # -----------------------------------

        # --- two-level reward setting ---
        if self.in_goal(obsrv):
            reward += self.goal_reward
            done = True
            suc  = True
        elif self.in_half_goal(obsrv):
            reward += self.goal_reward / 2.
        else:
            pass
        # -------------------------------

        # print("reward:", reward)
        #if self.out_of_time():
        #    print("episode out of max length")
        #    done = True

        # self.step_counter += 1
        # print("reward:%f" %reward)

        return np.array(np.copy(obsrv)), reward, done, suc, {}

    def get_obsrv(self, dynamic_data):
        # we don't include any state variables w.r.t 2D positions (x,y)
        # But we include the distance to walls around
        d_front = 5.0 - dynamic_data.pose.position.y 
        d_rear  = 5.0 + dynamic_data.pose.position.y
        d_left  = 5.0 + dynamic_data.pose.position.x
        d_right = 5.0 - dynamic_data.pose.position.x
        d_top   = 10  - dynamic_data.pose.position.z
        z = dynamic_data.pose.position.z
        vx =  dynamic_data.twist.linear.x
        vy =  dynamic_data.twist.linear.y
        vz =  dynamic_data.twist.linear.z

        roll_w = dynamic_data.twist.angular.x
        pitch_w = dynamic_data.twist.angular.y
        yaw_w = dynamic_data.twist.angular.z

        ox = dynamic_data.pose.orientation.x
        oy = dynamic_data.pose.orientation.y
        oz = dynamic_data.pose.orientation.z
        ow = dynamic_data.pose.orientation.w

        roll, pitch, yaw = euler_from_quaternion([ox, oy, oz, ow])

        return np.array([z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w, d_front, d_rear, d_left, d_right, d_top])

    def in_obst(self, contact_data):

        if len(contact_data.states) != 0:
            if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
                return True
        else:
                return False

    def in_goal(self, obsrv):
        pitch = obsrv[5]
        roll = obsrv[4]
        pitch_w = obsrv[8]
        roll_w = obsrv[7]
        vx = obsrv[1]
        vy = obsrv[2]
        vz = obsrv[3]
        
        if -0.348 <= pitch <= 0.348 and -0.348 <= roll <= 0.348 and -0.17 <= pitch_w <= 0.17 and -0.17 <= roll_w <= 0.17:
            print("reach goal!!")
            return True
        else:
            return False
        '''
        if -0.348 <= pitch <= 0.348 and -0.348 <= roll <= 0.348 and -0.17 <= pitch_w <= 0.17 and -0.17 <= roll_w <= 0.17 and -0.1 <= vx <= 0.1 and -0.1 <= vy <= 0.1 and -0.1 <= vz <= 0.1:
            print("reach the goal!!")
            return True
        else:
            return False
        '''

    def in_half_goal(self, obsrv):
        pitch = obsrv[5]
        roll = obsrv[4]
        pitch_w = obsrv[8]
        roll_w = obsrv[7]
        vx = obsrv[1]
        vy = obsrv[2]
        vz = obsrv[3]
        
        if (-0.17 <= pitch_w <= 0.17 and -0.17 <= roll_w <= 0.17):
            print("reach half goal!")
            return True
        else:
            return False

    def out_of_time(self):
        if self.step_counter > self.max_steps:
            self.step_counter = 0
            return True
        else:
            return False

    # observation space: [z,vx,vy,vz,roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate]
    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(15,))

    @property
    def action_space(self):
        # it's not feasible to set the lower limit to the weight of quad since sometimes we may require one of the motor to be zero for highly rolling or something
        return spaces.Box(low=0, high=self.max_lift/4., shape=(4,))
