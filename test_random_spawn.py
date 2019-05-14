#!/usr/bin/python3.5

from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SpawnModel


import rospy
import time

	
if __name__ == "__main__":
	
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
		# use time.sleep instead rospy.sleep in case of any time backward exceptions
		time.sleep(5.)


		pose = Pose()
		pose.position.x = 5
		pose.position.y = 5
		pose.position.z = 5

		pose.orientation.x = 0
		pose.orientation.y = 0
		pose.orientation.z = 0
		pose.orientation.w = 0
	
		obst_state = ModelState()
		obst_state.model_name = "obstacle_1"
		obst_state.pose = pose
		srv_set_model_state(obst_state)

	except rospy.ServiceException as serv_e:
		print("# Resets the state of the environment and returns an initial observation.")
	# Unpause simulation to make observation
	rospy.wait_for_service('/gazebo/unpause_physics')
	try:
		srv_unpause()
	except rospy.ServiceException as e:
		print("/gazebo/unpause_physics service call failed")
	
	print("random spawning obstacles success!!!")
