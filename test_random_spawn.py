#!/usr/bin/python3.5

from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SpawnModel


import rospy
import time
import numpy as np

# All the params below should be consistent with that on world file
# env params
RG = [-4,4]

# obstacle params
OBST_SIZE = 1.2
OBST_NUM = 3

# start and goal params
START_SIZE = 1.0
START_LOC = [3.182320, -3.339730, 3.0]
GOAL_SIZE = 1.5
GOAL_LOC = [4.0, 0.0, 9.0]


class RandObstacle(object):
	def __init__(self, size, rg):
		x = np.random.uniform(rg[0],rg[1])
		# random range for z from [-5,5] to [0,10]
		z = np.random.uniform(rg[0],rg[1]) + 5
		y = 0
		
		self.loc = (x,y,z)
		self.size = size
	
	def get_size(self):
		return self.size

	def get_loc(self):
		return self.loc

class Start(object):
	def __init__(self, size, loc):
		self.loc = loc
		self.size = size
	
	def get_size(self):
		return self.size
	
	def get_loc(self):
		return self.loc

class Goal(object):
	def __init__(self, size, loc):
		self.loc = loc
		self.size = size
	
	def get_size(self):
		return self.size

	def get_loc(self):
		return self.loc

def check_distance(obj1, obj2):
	offset = 0.2

	if np.sqrt(np.square(obj1.get_loc()[0] - obj2.get_loc()[0]) + np.square(obj1.get_loc()[2] - obj2.get_loc()[2])) >= obj1.get_size() * np.sqrt(2) / 2 + obj2.get_size() * np.sqrt(2) / 2 + offset:
		return True
	else:
		return False

def gen_obstacles(num, size, rg):
	# rg = RG
	# num: number of obstacles
	# size: integer 

	obsts = []
	start = Start(START_SIZE, START_LOC)
	goal  = Goal(GOAL_SIZE, GOAL_LOC)

	while True:
		tmp_obst = RandObstacle(size, rg)
		flag = True
		if len(obsts) >= 0 and len(obsts) < num:
			for j in range(len(obsts)):
				if not check_distance(obsts[j], tmp_obst) or not check_distance(start, tmp_obst) or not check_distance(goal, tmp_obst):
					flag = False
					break
			if flag:
				obsts.append(tmp_obst)

		else:
			break

	return obsts

def spawn_obstacles(obsts):
	assert len(obsts) > 0
	
	obst_states = []

	for obst in obsts:
		pose = Pose()
		pose.position.x = obst.get_loc()[0]
		pose.position.y = 0
		pose.position.z = obst.get_loc()[2]
		pose.orientation.x = 0
		pose.orientation.y = 0
		pose.orientation.z = 0
		pose.orientation.w = 0
		
		obst_state = ModelState()
		obst_state.model_name = "obstacle_" + str(obsts.index(obst)+1)
		obst_state.pose = pose
		obst_states.append(obst_state)
	
	return obst_states

def spawn_start_goal():
		start_pose = Pose()
		goal_pose = Pose()

		start_pose.position.x, goal_pose.position.x = START_LOC[0], GOAL_LOC[0]
		start_pose.position.y, goal_pose.position.y = START_LOC[1], GOAL_LOC[1]
		start_pose.position.z, goal_pose.position.z = START_LOC[2], GOAL_LOC[2]
		
		start_pose.orientation.x, goal_pose.orientation.x = 0, 0
		start_pose.orientation.y, goal_pose.orientation.y = 0, 0
		start_pose.orientation.z, goal_pose.orientation.z = 0, 0
		start_pose.orientation.w, goal_pose.orientation.w = 0, 0

		start_state = ModelState()
		goal_state = ModelState()
		start_state.model_name = "start"
		goal_state.model_name = "goal"
		start_state.pose = start_pose
		goal_state.pose = goal_pose

		return start_state, goal_state

#if __name__ == "__main__":
#	spawn_obstacles(num=3, size=0.5, rg=[-5,5])	

if __name__ == "__main__":
	# You have to initialize node at first when using rospy.
	# the node name could be set as you wish. 
	# Actually the node here means your own code file
	rospy.init_node("random_spawn", anonymous=True, log_level=rospy.INFO)
	print("rospy node names:", rospy.get_namespace())	
	srv_unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
	print("do I get here??")
	srv_pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
	srv_reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
	srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_model', SpawnModel)
	srv_get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
	srv_set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
	
	rospy.wait_for_service('/gazebo/reset_simulation')

	print("do I get here??")
	try:
		srv_reset_proxy()
		# spawn start and goal for only one time at beginning
		start_st, goal_st = spawn_start_goal()
		time.sleep(0.05)
		srv_set_model_state(start_st)
		time.sleep(0.05)
		srv_set_model_state(goal_st)
		# Do randomly spawning obstacles multiple times
		N = 20
		idx = 0
		while idx <= N:

			# use time.sleep instead rospy.sleep in case of any time backward exceptions
			time.sleep(5.)

			rospy.wait_for_service('/gazebo/pause_physics')
			try:
				srv_pause()
			except rospy.ServiceException as e:
				print("/gazebo/pause_physics service call failed")
			
			# generate a few obstacles 			
			obsts = gen_obstacles(num=OBST_NUM, size=OBST_SIZE, rg=RG)
			print(obsts)
			
			# retrieve spawning info 
			obsts_st = spawn_obstacles(obsts)
			
			print(obsts_st)
			idx += 1
 			
			# spawn objects using ros service
			for ost in obsts_st:
				time.sleep(0.05)
				srv_set_model_state(ost)
				

			rospy.wait_for_service('/gazebo/unpause_physics')
			try:
				srv_unpause()
			except rospy.ServiceException as e:
				print("/gazebo/unpause_physics service call failed") 

	except rospy.ServiceException as serv_e:
		print("# Resets the state of the environment and returns an initial observation.")
	
	# Unpause simulation to make observation
	rospy.wait_for_service('/gazebo/unpause_physics')
	try:
		srv_unpause()
	except rospy.ServiceException as e:
		print("/gazebo/unpause_physics service call failed")
 
	print("random spawning obstacles success!!!")



