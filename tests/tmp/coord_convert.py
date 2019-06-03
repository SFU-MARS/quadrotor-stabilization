#!/usr/bin/python3.6

import numpy as np

# coordinate transform from board frame to world frame
def board_to_world(board_coord, pitch, roll, yaw):

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

if __name__ == "__main__":
		# board frame: x=0, y=0, z=1
		F = np.array([0, 0, 1])

		roll, pitch, yaw = 0, 0, -np.pi/0.24

		print(board_to_world(F, pitch, roll, yaw))
	
		
