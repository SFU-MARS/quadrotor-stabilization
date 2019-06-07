#!/usr/bin/python3.5

from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import ApplyBodyWrench

from tf.transformations import quaternion_from_euler
import rospy
import time
import numpy as np


# distance from motor to center of quad.
# crazyflie size: 0.092 * 0.092 * 0.029. After 10 times scale, becoming 0.92 * 0.92 *0.29
L = 0.65

# coordinate transform from board frame to world frame
def board_to_world(board_coord, roll, pitch, yaw):

	Rx = np.array([[1, 0, 0],
	               [0, np.cos(roll), -np.sin(roll)],
								 [0, np.sin(roll), np.cos(roll)]])
	Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
		             [0, 1, 0],
								 [-np.sin(pitch), 0, np.cos(pitch)]])
	Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
			           [np.sin(yaw), np.cos(yaw), 0],
								 [0, 0, 1]])

	M = np.dot(Rz, np.dot(Ry, Rx))
	ret = np.dot(M, board_coord)

	return ret
		
def init_quad(srv):
	quad_state = None

	quad_pose = Pose()
	quad_pose.position.x = 0
	quad_pose.position.y = 0
	quad_pose.position.z = 3

	qu_x, qu_y, qu_z, qu_w = quaternion_from_euler(0, np.pi/2, 0)
	quad_pose.orientation.x = qu_x
	quad_pose.orientation.y = qu_y
	quad_pose.orientation.z = qu_z
	quad_pose.orientation.w = qu_w

	quad_state = ModelState()
	quad_state.model_name = "quadrotor"
	quad_state.pose = quad_pose
	
	srv(quad_state) 
	return True

def apply_wrench_to_quad(srv, action, roll, pitch, yaw):

	# pitch, roll, yaw is the relative motion between two frame

	# action: [UL, UR, LL, LR]
	wrench_UL, wrench_UR, wrench_LL, wrench_LR = Wrench(), Wrench(), Wrench(), Wrench()

	# apply linear force
	wrench_UL.force.x, wrench_UL.force.y, wrench_UL.force.z = board_to_world(np.array([0,0,action[0]]), roll=roll, pitch=pitch, yaw=yaw)
	wrench_UR.force.x, wrench_UR.force.y, wrench_UR.force.z = board_to_world(np.array([0,0,action[1]]), roll=roll, pitch=pitch, yaw=yaw)
	wrench_LL.force.x, wrench_LL.force.y, wrench_LL.force.z = board_to_world(np.array([0,0,action[2]]), roll=roll, pitch=pitch, yaw=yaw)	
	wrench_LR.force.x, wrench_LR.force.y, wrench_LR.force.z = board_to_world(np.array([0,0,action[3]]), roll=roll, pitch=pitch, yaw=yaw)

	# apply torque
	wrench_UL.torque.x, wrench_UL.torque.y, wrench_UL.torque.z = board_to_world(np.array([action[0]*L*np.sqrt(2)/2, action[0]*L*np.sqrt(2)/2, 0]), roll=roll, pitch=pitch, yaw=yaw)
	wrench_UR.torque.x, wrench_UR.torque.y, wrench_UR.torque.z = board_to_world(np.array([action[1]*L*np.sqrt(2)/2, -action[1]*L*np.sqrt(2)/2, 0]), roll=roll, pitch=pitch, yaw=yaw)
	wrench_LL.torque.x, wrench_LL.torque.y, wrench_LL.torque.z = board_to_world(np.array([-action[2]*L*np.sqrt(2)/2, action[2]*L*np.sqrt(2)/2, 0]), roll=roll, pitch=pitch, yaw=yaw)
	wrench_LR.torque.x, wrench_LR.torque.y, wrench_LR.torque.z = board_to_world(np.array([-action[3]*L*np.sqrt(2)/2, -action[3]*L*np.sqrt(2)/2, 0]), roll=roll, pitch=pitch, yaw=yaw)

	# Actually we need combine all four motors effect into one
	wrench = Wrench()
	# apply to force
	wrench.force.x = wrench_UL.force.x + wrench_UR.force.x + wrench_LL.force.x + wrench_LR.force.x
	wrench.force.y = wrench_UL.force.y + wrench_UR.force.y + wrench_LL.force.y + wrench_LR.force.y
	wrench.force.z = wrench_UL.force.z + wrench_UR.force.z + wrench_LL.force.z + wrench_LR.force.z
	# apply to torque
	wrench.torque.x = wrench_UL.torque.x + wrench_UR.torque.x + wrench_LL.torque.x + wrench_LR.torque.x
	wrench.torque.y = wrench_UL.torque.y + wrench_UR.torque.y + wrench_LL.torque.y + wrench_LR.torque.y
	wrench.torque.z = wrench_UL.torque.z + wrench_UR.torque.z + wrench_LL.torque.z + wrench_LR.torque.z
	
	
	srv(body_name="base_link", reference_frame="world", wrench=wrench, start_time=rospy.Time().now(), duration=rospy.Duration(1))
	
	return True

# test random falling down task
if __name__ == "__main__":
 	# You have to initialize node at first when using rospy.
 	# the node name could be set as you wish. 
 	# Actually the node here means your own code file
 	rospy.init_node("random_falling", anonymous=True, log_level=rospy.INFO)
 	srv_unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
 	srv_pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
 	srv_reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
 	srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_model', SpawnModel)
 	srv_get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
 	srv_set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
 	srv_apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
 
 	rospy.wait_for_service('/gazebo/reset_simulation')
 	print("do I get here??")
 	try:
 		srv_reset_proxy()
 
 		time.sleep(3)
 		init_quad(srv_set_model_state)
 
 		time.sleep(3)
 		apply_wrench_to_quad(srv_apply_wrench, [0.1,0.1,0.1,0.1], roll=0, pitch=np.pi/2, yaw=0)
 	except rospy.ServiceException as e:
 		print("# Reset simulation failed!")
 	
 	# Unpause simulation to make observation
 	rospy.wait_for_service('/gazebo/unpause_physics')
 	try:
 		srv_unpause()
 	except rospy.ServiceException as e:
 		print("/gazebo/unpause_physics service call failed")
		
		


