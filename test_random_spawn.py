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


class rand_obstacle(object):
	def __init__(self, size, rg):
		x = np.random.uniform(rg[0],rg[1])
		z = np.random.uniform(rg[0],rg[1])

		self.loc = (x,z)
		self.size = size
	

def check_distance(obst1, obst2):
	offset = 0.1
	if np.sqrt(np.square((obst1.loc[0] - obst2.loc[0])) + np.square(obst1.loc[1] - obst2.loc[1])) >= obst1.size * np.sqrt(2) / 2 + obst2.size * np.sqrt(2) / 2 + offset:
		return True
	else:
		return False

def spawn_obstacles(num, size, rg):
	# rg = [-5,5]
	# num: number of obstacles
	# size: integer 

	obsts = []
	while True:
		tmp_obst = rand_obstacle(size, rg)
		flag = True
		if len(obsts) >= 0 and len(obsts) < num:
			for j in range(len(obsts)):
				if not check_distance(obsts[j], tmp_obst):
					flag = False
					break
			if flag:
				obsts.append(tmp_obst)

		else:
			break
	
	for i in range(len(obsts)):
		print("obstacle locs:", obsts[i].loc)



	
if __name__ == "__main__":
	# You have to initialize node at first when using rospy.
	# the node name could be set as you wish. 
	# Actually the node here means your own code file
	
#	rospy.init_node("random_spawn", anonymous=True, log_level=rospy.INFO)
#	print("rospy node names:", rospy.get_namespace())	
#	srv_unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
#	print("do I get here??")
#	srv_pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
#	srv_reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
#	srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_model', SpawnModel)
#	srv_get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
#	srv_set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
#	
#	rospy.wait_for_service('/gazebo/reset_simulation')
#
#	print("do I get here??")
#	try:
#		srv_reset_proxy()
#		# use time.sleep instead rospy.sleep in case of any time backward exceptions
#		time.sleep(5.)
#
#
#		pose = Pose()
#		pose.position.x = 5
#		pose.position.y = 5
#		pose.position.z = 5
#
#		pose.orientation.x = 0
#		pose.orientation.y = 0
#		pose.orientation.z = 0
#		pose.orientation.w = 0
#	
#		obst_state = ModelState()
#		obst_state.model_name = "obstacle_1"
#		obst_state.pose = pose
#		srv_set_model_state(obst_state)
#
#	except rospy.ServiceException as serv_e:
#		print("# Resets the state of the environment and returns an initial observation.")
#	
#	# Unpause simulation to make observation
#	rospy.wait_for_service('/gazebo/unpause_physics')
#	try:
#		srv_unpause()
#	except rospy.ServiceException as e:
#		print("/gazebo/unpause_physics service call failed")
	
#	print("random spawning obstacles success!!!")


	spawn_obstacles(num=3, size=0.5, rg=[-5,5])
