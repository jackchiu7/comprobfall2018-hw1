#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 27 13:36:21 2018

@author: jackc
"""

from contextlib import contextmanager
import sys, os
import compRoboticsHW1RUN as rbc
from turtlebot_ctrl.srv import TurtleBotControl
from turtlebot_ctrl.msg import TurtleBotState

import rospy




ptList = []



@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
         
def main():

#    os.system('rosparam set goal_position [-5,-1.75]')
    
#    os.system('rosparam set initial_pose [-1,-1.75]')
    goal = [9,7]
    start = [0,9]
#    cmd = 'ROBOT_INITIAL_POSE="-x -1 -y -1.75" '
#    cmd = cmd+ 'roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/home/bhargod/catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/worlds/'
#    cmd = cmd + 'world_1.world'
#    os.system(cmd)
#    os.system('rosrun turtlebot_ctrl turtlebot_control.py')

    filePath = '/home/bhargod/catkin_ws/src/turtlebot_maps/'
    filename = filePath+'map_5.txt'

    print filename
    fdaVal = True
    visVal = False
    with suppress_stdout():
        grid, path, time = rbc.main(filename, start = start, goal = goal, fdaVal = fdaVal, vis = visVal)
    
    
    rospy.init_node("ihatethisclass")
    rospy.wait_for_service('turtlebot_control')
    turtlebot_client = rospy.ServiceProxy('turtlebot_control', TurtleBotControl)



    ptList = []
    while path:
        node = path.pop()
        pt = TurtleBotControl()
        pt.x = node.x
        pt.y = node.y
        pt.z = 0
        ptList.append(pt)
        
    
    ptList.pop(0)
    
    while ptList:
        try:
            tb = ptList.pop(0)
            response = turtlebot_client(tb)
            print response
        except rospy.ServiceException as e:
            print 'Service could not process request ' + str(e)
    
    
    
    

#    ptList.pop()
#    
#    currPoint = ptList.pop()
#    while currPoint.x != goal[0] and currPoint.y != goal[1]:   
#        
#        while not call_service_client(currPoint):
#            call_service_client(currPoint)
#        currPoint = ptList.pop()
#        
#        
#    while not call_service_client(currPoint):
#        call_service_client(currPoint)

    
#    for pt in ptList:
#        print pt.x, pt.y 
        
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
