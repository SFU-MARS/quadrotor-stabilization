#!/usr/bin/python3.5

from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.spaces.box import Box
import numpy as np

import rospy
import time


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



class QuadFallingDown(Env, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        # Note: need to be compatible with the quadrotor.urdf. Here I use crazyflie params
        self.mass = 0.027
        self.weight = self.mass * 9.81
        self.max_lift = 0.042 * 9.81

        self.step_counter = 0
        self.max_steps = 100

        self.collision_reward = -400
        self.goal_reward = 1000

        self.reward_type = 'sparse'

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
            pitch = np.random.uniform(-0.1 + np.pi/4, 0.1 + np.pi/4)
            # the angle w.r.t z-axis: yaw in gazebo
            # yaw = np.random.uniform(0, 2*np.pi)
            yaw = 0
            ox, oy, oz, ow = quaternion_from_euler(roll, pitch, yaw)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = ox,oy,oz,ow

            # initialize twist
            # the reset value should be referring to some empirical value
            vx =  np.random.uniform(-1,1)
            vy =  np.random.uniform(-1,1)
            vz =  np.random.uniform(-1,1)

            # roll_w = np.random.uniform(-np.pi/6, np.pi/6)
            # pitch_w = np.random.uniform(-np.pi/6, np.pi/6)
            # yaw_w = np.random.uniform(-np.pi/6, np.pi/6)
            roll_w = 0
            pitch_w = 0
            yaw_w = 0

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
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # print("action before clipped:", action)
        pre_roll = self.pre_obsrv[4]
        pre_pitch = self.pre_obsrv[5]
        pre_yaw = self.pre_obsrv[6]

        # Note: may require clipping if using Gaussian distribution as action model
        # actually the clippling here makes no difference since normalized() func helps with it
        # no harm to keep it here.
        a = action.copy()
        clipped_action = np.clip(a, self.action_space.bounds[0][0], self.action_space.bounds[1][0])
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        _ = apply_wrench_to_quad(self.apply_wrench, clipped_action, pre_roll, pre_pitch, pre_yaw)

        # print("action after clipped:", action)
        # run simulator to collect data
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
                    contact_data = rospy.wait_for_message('/gazebo_ros_bumper', ContactsState, timeout=5)
                    dynamic_data = self.get_model_state(model_name="quadrotor")
                except rospy.ServiceException as e:
                    print("/gazebo/get_model_state service call failed!")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(dynamic_data)
        self.pre_obsrv = obsrv


        reward = 0
        # put penalty on invalid action
        reward += -np.sum(np.abs(action - clipped_action))


        if self.reward_type == 'sparse':
            reward = 0
        elif self.reward_type == 'ttr':
            ttr = self.brsEngine.evaluate_ttr(obsrv)
            reward = -ttr
        else:
            pass

        done = False

        if self.in_obst(contact_data):
            reward += self.collision_reward
            done = True

        if self.in_goal(obsrv):
            reward += self.goal_reward
            done = True

        if self.out_of_time():
            done = True

        self.step_counter += 1

        # print("reward:%f" %reward)

        return Step(observation=np.copy(obsrv), reward=reward, done=done)

    def get_obsrv(self, dynamic_data):
        # we don't include any state variables w.r.t 2D positions (x,y)
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

        return np.array([z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w])

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

        # pitch and roll are around 45 degrees, angular vel is around 10 degrees per sec.
    #   if -0.5 <= pitch <= 0.5 and -0.5 <= roll <= 0.5 and -0.17 <= pitch_w <= 0.17 and -0.17 <= roll_w <= 0.17 and -0.1 <= vx <= 0.1 and -0.1 <= vy <= 0.1 and -0.1 <= vz <= 0.1:
    #       return True
    #   else:
    #       return False
        if -0.17 <= pitch <= 0.17 and -0.17 <= roll <= 0.17 and -0.17 <= pitch_w <= 0.17 and -0.17 <= roll_w <= 0.17 and -0.1 <= vx <= 0.1 and -0.1 <= vy <= 0.1 and -0.1 <= vz <= 0.1:
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
        return Box(low=-np.inf, high=np.inf, shape=(10,))

    @property
    def action_space(self):
        # it's not feasible to set the lower limit to the weight of quad since sometimes we may require one of the motor to be zero for highly rolling or something

        return Box(low=0, high=self.max_lift/4., shape=(4,))
