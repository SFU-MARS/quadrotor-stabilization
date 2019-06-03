from garage.envs.base import Env
from garage.envs.base import Step
from akro import Box
import numpy as np

import rospy
import time


from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench

from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import ApplyBodyWrench
from tf.transformations import euler_from_quaternion, quaternion_from_euler


from utils import board_to_world, init_quad, apply_wrench_to_quad

class QuadFallingDown(Env):
	def __init__(self):
		# Note: need to be compatible with the quadrotor.urdf
		self.mass = 0.027
		self.weight = self.mass * 9.81
		self.max_lift = 0.042 * 9.81

		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
	
		self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
	
	def reset(self):

		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.reset_proxy()
			# initialize pose 
			pose = Pose()
			pose.position.x, pose.position.y = np.random.uniform(low=-10, high=10), np.random.uniform(low=-10, high=10)
			pose.position.z = np.random.uniform(3,10)
			
			# Note: gazebo rotation order: roll, pitch, yaw
			# the angle w.r.t x-axis: roll in gazebo
			roll = np.random.uniform(-np.pi/2, np.pi/2)
			# the angle w.r.t y-axis: pitch in gazebo
			pitch = np.random.uniform(-np.pi/2, np.pi/2) 
			# the angle w.r.t z-axis: yaw in gazebo
			yaw = np.random.uniform(0, 2*np.pi) 
			ox, oy, oz, ow = quaternion_from_euler(roll, pitch, yaw)
			pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = ox,oy,oz,ow

			# initialize twist
			# the reset value should be referring to some empirical value
			vx =  np.random.uniform(-1,1)
			vy =  np.random.uniform(-1,1)
			vz =  np.random.uniform(-1,1)
			roll_w = np.random.uniform(-np.pi/6, np.pi/6)
			pitch_w = np.random.uniform(-np.pi/6, np.pi/6) 
			yaw_w = np.random.uniform(-np.pi/6, np.pi/6) 
		
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
		dynamic_date = None
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

		pre_roll = self.pre_obsrv[4]
		pre_pitch = self.pre_obsrv[5]
		pre_yaw = self.pre_obsrv[6]
		
		# Note: may require clipping if using Gaussian distribution as action model
		rospy.wait_for_service('/gazebo/apply_body_wrench')
		_ = apply_wrench_to_quad(self.apply_wrench, pre_roll, pre_pitch, pre_yaw)
	
		# run simulator to collect data
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except rospy.ServiceException as e:
			print("/gazebo/unpause_physics service call failed")
		

		dynamic_data = None
		while dynamic_data is None:
				rospy.wait_for_service('/gazebo/get_model_state')
				try:
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

		if self.reward_type == 'sparse':
			reward = 0
		elif self.reward_type == 'ttr':
			ttr = self.brsEngine.evaluate_ttr(obsrv)
			reward = -ttr
		else:
			pass

		done = False
		
		if self.in_obst():
			reward += collision_reward
			done = True

		if self.in_goal(obsrv):
			reward += goal_reward
			done = True
		
		return np.asarray(obsrv), reward, done, {}
				
	def get_obsrv(self, dynamic_data)
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
	
	def	in_obst(self):
		# Not implemented ...
		pass

	def in_goal(self, obsrv):
		# Not implemented
		pass
		
	# observation space: [z,vx,vy,vz,roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate]
	@property
	def observation_space(self):
		return Box(low=-np.inf, high=np.inf, shape=(10,))

	@property
	def action_space(self):
		return Box(low=self.weight/4., high=self.max_lift/4., shape=(4,))




