#!/usr/bin/env python

PACKAGE_NAME = 'hw2'

# Standard Python Imports
import os
import copy
import time
import math
import numpy as np
np.random.seed(0)
import scipy

import collections
import Queue

# OpenRAVE
import openravepy
#openravepy.RaveInitialize(True, openravepy.DebugLevel.Debug)


curr_path = os.getcwd()
relative_ordata = '/models'
ordata_path_thispack = curr_path + relative_ordata


#this sets up the OPENRAVE_DATA environment variable to include the files we're using
openrave_data_path = os.getenv('OPENRAVE_DATA', '')
openrave_data_paths = openrave_data_path.split(':')
if ordata_path_thispack not in openrave_data_paths:
  if openrave_data_path == '':
      os.environ['OPENRAVE_DATA'] = ordata_path_thispack
  else:
      datastr = str('%s:%s'%(ordata_path_thispack, openrave_data_path))
      os.environ['OPENRAVE_DATA'] = datastr

#set database file to be in this folder only
relative_ordatabase = '/database'
ordatabase_path_thispack = curr_path + relative_ordatabase
os.environ['OPENRAVE_DATABASE'] = ordatabase_path_thispack

#get rid of warnings
openravepy.RaveInitialize(True, openravepy.DebugLevel.Fatal)
openravepy.misc.InitOpenRAVELogging()


#constant for max distance to move any joint in a discrete step
TRANS_PER_DIR = 0.1


class RoboHandler:
  def __init__(self):
    self.openrave_init()
    self.problem_init()



  #######################################################
  # the usual initialization for openrave
  #######################################################
  def openrave_init(self):
    self.env = openravepy.Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('HW2 Viewer')
    self.env.Load('models/%s.env.xml' %PACKAGE_NAME)
    # time.sleep(3) # wait for viewer to initialize. May be helpful to uncomment
    self.robot = self.env.GetRobots()[0]

    #set right wam as active manipulator
    with self.env:
      self.robot.SetActiveManipulator('right_wam');
      self.manip = self.robot.GetActiveManipulator()

      #set active indices to be right arm only
      self.robot.SetActiveDOFs(self.manip.GetArmIndices() )
      self.end_effector = self.manip.GetEndEffector()

  #######################################################
  # problem specific initialization
  #######################################################
  def problem_init(self):
    self.target_kinbody = self.env.GetKinBody("target")

    # create a grasping module
    self.gmodel = openravepy.databases.grasping.GraspingModel(self.robot, self.target_kinbody)
    
    # load grasps
    if not self.gmodel.load():
      self.gmodel.autogenerate()

    self.grasps = self.gmodel.grasps
    self.graspindices = self.gmodel.graspindices

    # load ikmodel
    self.ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(self.robot,iktype=openravepy.IkParameterization.Type.Transform6D)
    if not self.ikmodel.load():
      self.ikmodel.autogenerate()

    # create taskmanip
    self.taskmanip = openravepy.interfaces.TaskManipulation(self.robot)
  
    # move left arm out of way
    self.robot.SetDOFValues(np.array([4,2,0,-1,0,0,0]),self.robot.GetManipulator('left_wam').GetArmIndices() )


  #######################################################
  # Simpler search problem - uses breadth first search algorithm
  #######################################################
  def run_simple_problem(self):
    self.robot.GetController().Reset()

    # move hand to preshape of grasp
    # --- important --
    # I noted they were all the same, otherwise you would need to do this separately for each grasp!
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    
    
    self.init_transition_arrays()
#    goal = [ 0.93422058, -1.10221021, -0.2,  2.27275587, -0.22977831, -1.09393251, -2.23921746]
       
#    goal = self.convert_for_dict(goal)
#    goal = self.convert_from_dictkey(goal)
    self.start = [1.23, -1.10, -0.3, 2.37, -0.23, -1.29, -2.23]
#    self.start = [1.2, -1.10, -0.3, 2.3, -0.2, -1.2, -2.2]
    goal = [ 0.93, -1.10, -0.2,  2.27, -0.23, -1.09, -2.23] 
#    goal = [ 1.4, -1.30, -0.3, 2.3,-0.2, -1.2, -2.2]
    with self.env:
      self.robot.SetActiveDOFValues(self.start)

    # get the trajectory!
    traj = self.search_to_goal_breadthfirst([goal])

    with self.env:
      self.robot.SetActiveDOFValues(self.start)

    
    self.robot.GetController().SetPath(traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()
    
#    with self.env:
##        self.robot.SetActiveDOFValues([0.93422058, -1.10221021, -0.2,  2.27275587, -0.22977831, -1.09393251, -2.23921746])
#         self.robot.SetActiveDOFValues(self.start)
  #######################################################
  # Harder search problem - uses A* algorithm
  #######################################################
  def run_difficult_problem(self):
    self.robot.GetController().Reset()

    # move hand to preshape of grasp
    # --- important --
    # I noted they were all the same, otherwise you would need to do this separately for each grasp!
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    

    self.init_transition_arrays()
    #goals = self.get_goal_dofs(7,1)
    goals = np.array([[ 0.93422058, -1.10221021, -0.2       ,  2.27275587, -0.22977831, -1.09393251, -2.23921746],
       [ 1.38238176, -1.05017481,  0.        ,  1.26568204,  0.15001448,  1.32813949, -0.06022621],
       [ 1.16466262, -1.02175153, -0.3       ,  1.26568204, -2.62343746, -1.43813577, -0.37988181],
       [ 3.45957137, -0.48619817,  0.        ,  2.0702298 , -1.12033301, -1.33241556,  1.85646563],
       [ 1.65311863, -1.17157253,  0.4       ,  2.18692683, -2.38248898,  0.73272595, -0.23680544],
       [ 1.59512823, -1.07309638,  0.5       ,  2.26315055,  0.57257592, -1.15576369, -0.30723627],
       [ 1.67038884, -1.16082512,  0.4       ,  2.05339849, -2.0205527 ,  0.54970211, -0.4386743 ]])
    self.start = np.array([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856]) 
    with self.env:
      self.robot.SetActiveDOFValues(self.start)

    # get the trajectory!
    traj = self.search_to_goal_astar(goals)

    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])
    
    self.robot.GetController().SetPath(traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()



  #######################################################
  # finds the arm configurations (in cspace) that correspond
  # to valid grasps
  # num_goal: number of grasps to consider
  # num_dofs_per_goal: number of IK solutions per grasp
  #######################################################
  def get_goal_dofs(self, num_goals=1, num_dofs_per_goal=1):
    validgrasps,validindices = self.gmodel.computeValidGrasps(returnnum=num_goals) 

    curr_IK = self.robot.GetActiveDOFValues()

    goal_dofs = np.array([])
    for grasp, graspindices in zip(validgrasps, validindices):
      Tgoal = self.gmodel.getGlobalGraspTransform(grasp, collisionfree=True)
      sols = self.manip.FindIKSolutions(Tgoal, openravepy.IkFilterOptions.CheckEnvCollisions)

      # magic that makes sols only the unique elements - sometimes there are multiple IKs
      sols = np.unique(sols.view([('',sols.dtype)]*sols.shape[1])).view(sols.dtype).reshape(-1,sols.shape[1]) 
      sols_scores = []
      for sol in sols:
        sols_scores.append( (sol, np.linalg.norm(sol-curr_IK)) )

      # sort by closest to current IK
      sols_scores.sort(key=lambda tup:tup[1])
      sols = np.array([x[0] for x in sols_scores])
      
      # sort randomly
      #sols = np.random.permutation(sols)

      #take up to num_dofs_per_goal
      last_ind = min(num_dofs_per_goal, sols.shape[0])
      goal_dofs = np.append(goal_dofs,sols[0:last_ind])

    goal_dofs = goal_dofs.reshape(goal_dofs.size/7, 7)

    return goal_dofs


  ### TODO:REPLICATE FROM BFS ###  
  #######################################################
  # DEPTH FIRST SEARCH
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  ##########MARSH###############
  def search_to_goal_depthfirst(self, goals):
    visited_nodes = {}
    nodes = Queue.LifoQueue()
    start = self.robot.GetActiveDOFValues()
    nodes.put(start)
    print 'test'
    trajectory = np.array([])
    for g in goals: #for each of the goal states
        goal_reached = False
        while not nodes.empty() and not goal_reached: #while the queue has nodes
            currentNode = nodes.get() #pop the next node
            if np.allclose(currentNode,g): #check if we reached the current goal
                goal_reached = True                
                continue #jump out of the loop

                neighbors = self.transition_config(currentNode)
            
                for c in neighbors:
                    if not self.convert_for_dict(c) in visited_nodes.keys():
                            visited_nodes[self.convert_for_dict(c)] = currentNode
                            nodes.put(c)
                    print np.size(c)        
                     
        r= currentNode                                                      #Ankit
        while r is not start:
                trajectory = np.append(trajectory,r)
                r = visited_nodes[self.convert_for_dict(r)]
        start = g
        
        #nodes.queue.clear()
        nodes.put(start)
    trajectory = np.reshape(trajectory,(np.size(trajectory)/7,7))              #~Ankit

    print 'Found goal' + g
    return trajectory                                                           #Ankit #~Ankit

  ### TODO:CLEAN-UP ###  
  #######################################################
  # BREADTH FIRST SEARCH
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def search_to_goal_breadthfirst(self, goals):
#        print self.start
        start = np.array(self.start)
        parents = {}
        visited_nodes=set([])
        nodes = collections.deque()
        #start = np.array(self.robot.GetActiveDOFValues())
        #start = [1.23, -1.11, -0.30, 2.37, -0.23, -1.29, -2.23]
        print "start:", start
        parents[self.convert_for_dict(start)] = None
        print "parents: start: ", parents
        
        visited_nodes.add(self.convert_for_dict(start))
        print "visited_node:", visited_nodes
        
        nodes.append(start)
        currentNode = start
        print "Nodes before main loop:", nodes
        print 'test'
        
        
        trajectory = np.array([])
#        for g in goals: #for each of the goal states
#                goal_reached = False
#                while nodes and not goal_reached: #while the queue has nodes
#                        currentNode = nodes.popleft() #pop the next node
#                        print "distance to goal:", (currentNode-g)        #to check distance from goal
#                        if np.allclose(currentNode,g): #check if we reached the current goal
#                                goal_reached = True    
#                                print "Goal Reached?",goal_reached
#                                continue #jump out of the loop
#
#                        neighbors = self.transition_config(currentNode)
#                        #print "neighbors:",  neighbors
#                        for c in neighbors:
#                                #if not self.check_collision(c):
#                                if not self.convert_for_dict(c) in parents.keys():    
#                                                
#                                    parents[self.convert_for_dict(c)] = currentNode
#                                    nodes.append(c)
#                        #print c    
#            #print currentNode
#                parents[self.convert_for_dict(g)] = currentNode 
#                print "parents:", parents
#                r = g 
#                while r is not start:
#                        trajectory = np.append(trajectory,r)
#                        r = parents[self.convert_for_dict(r)]
#                        print r
#                trajectory = np.append(trajectory,r)
#                start = g
#                nodes = collections.deque()
#                nodes.append(start)
#        trajectory = np.reshape(trajectory,(np.size(trajectory)/7,7))              #~Ankit
#
#        print "trajectory", trajectory
#        return trajectory
###################################################################################
        g=goals[0]
        print 'g', g
        goal_reached = False
        while nodes and not goal_reached: #while the queue has nodes
            currentNode = nodes.popleft() #pop the next node
#            print "distance to goal:", (currentNode-g)        #to check distance from goal
            if np.allclose(currentNode,g): #check if we reached the current goal
                goal_reached = True    
                print "Goal Reached?",goal_reached
                print"Current Node at goal reached", currentNode
                print"G at goal reached", g
                continue #jump out of the loop
#            print 'curent node:', currentNode
            neighbors = self.transition_config(currentNode)
#            print "neighbors:",  neighbors
            for c in neighbors:
#              if not self.check_collision(c):
#                print 'c:', c
                if self.convert_for_dict(c) not in visited_nodes:    #test if neighbor has not been visited    
#                    if c not in 
                        visited_nodes.add(self.convert_for_dict(c))  
                        parents[self.convert_for_dict(c)] = currentNode
                        nodes.append(c)
                        #print c    
            #print currentNode
#        visited_nodes.add(self.convert_for_dict(g))
#        print 'visited nodes : ', visited_nodes
#        parents[self.convert_for_dict(g)] = currentNode 
#        print "parents:", parents
        
        r = currentNode 
        print "r", r
        print 'start', start
        print len(parents)
        print len(visited_nodes)
        a1 = parents[self.convert_for_dict(r)]
        a2 = parents[self.convert_for_dict(a1)]
        a3 = parents[self.convert_for_dict(a2)]
        print 'a1: ', a1, ' a2: ', a2, ' a3: ', a3
        while r is not start:
            trajectory = np.append(trajectory,r)
#            a = trajectory[0]
#            b = trajectory [1]
#            c = trajectory [2]
            r = parents[self.convert_for_dict(r)]
#            print r
        trajectory = np.append(trajectory,r)
        nodes = collections.deque()
        nodes.append(start)
        trajectory = np.reshape(trajectory,(np.size(trajectory)/7,7))              #~Ankit
        trajectory[0]= [ 0.93422058, -1.10221021, -0.2,  2.27275587, -0.22977831, -1.09393251, -2.23921746]
        trajectory = trajectory[::-1]
        print "trajectory", trajectory
        traj = self.points_to_traj(trajectory)
        return traj
###########################################################################################################
  ### TODO ###  
  #######################################################
  # A* SEARCH
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def search_to_goal_astar(self, goals):
    start = np.array(self.start)
    goals = np.array(goals)
    openset = Queue.PriorityQueue()
    closedset = set([])
    came_from = {}
    g_score = {}
    g_score[self.convert_for_dict(start)] = 0
    dist, goal = self.min_manhattan_dist_to_goals(start, goals)
    #goal = np.int_(goal*100)/100. 
    openset.put(self.config_to_priorityqueue_tuple(dist, start, goals))
    came_from[self.convert_for_dict(start)] = None
    while not openset.empty():
	temp = openset.get()
	current = np.array(temp[1])
	dist = temp[0]
	dist1, g = self.min_euclid_dist_to_goals(current, goals)
	if dist1<0.1:		#ankit
		print "success"
		break

	closedset.add(self.convert_for_dict(current))
	neighbors = self.transition_config(current)
	for neighbor in neighbors:
		tentative_score = g_score[self.convert_for_dict(current)] + TRANS_PER_DIR

		if self.convert_for_dict(neighbor) in closedset:
			if tentative_score >= g_score[self.convert_for_dict(neighbor)]:				
				continue

		if self.convert_for_dict(neighbor) not in g_score.keys(): 
			if not self.check_collision(neighbor):
				came_from[self.convert_for_dict(neighbor)] = current
				g_score[self.convert_for_dict(neighbor)] = tentative_score
 				dist, goal = self.min_manhattan_dist_to_goals(neighbor, goals)
				#goal = np.int_(goal*100)/100.
				openset.put(self.config_to_priorityqueue_tuple(dist, neighbor, goals))
		else:
			if tentative_score < g_score[self.convert_for_dict(neighbor)]:
				g_score[self.convert_for_dict(neighbor)] = tentative_score
	
    r = current
    print r
    trajectory = np.array([])
    #a1 = came_from[self.convert_for_dict(r)]
    #print "a1", a1
    #a2 = came_from[self.convert_for_dict(a1)]
    #a3 = came_from[self.convert_for_dict(a2)]
    #a4 = came_from[self.convert_for_dict(a3)]
    #print "a1", a1
    #print "a2", a2
    #print "a3", a3
    #print "a4", a4
    #print came_from
    #print r
    trajectory = np.append(trajectory,goal)
    r = came_from[self.convert_for_dict(r)]
    while r is not None:
       	trajectory = np.append(trajectory,r)
       	r = came_from[self.convert_for_dict(r)]
        trajectory = np.append(trajectory,r)
    trajectory = trajectory[0:np.size(trajectory)-1]
    trajectory = np.reshape(trajectory, (np.size(trajectory)/7,7))              #~Ankit

    trajectory = trajectory[::-1]
    print trajectory	
    #print "failure"
    return self.points_to_traj(trajectory)





###################################################
#  Collision Check
###################################################
  def check_collision(self, DOFs):
    current_DOFs = self.robot.GetActiveDOFValues()
    with self.env:
        self.robot.SetActiveDOFValues(DOFs)
        collision1 = self.env.CheckCollision(self.robot) 
        collision2 = self.robot.CheckSelfCollision()
        #self.robot.SetActiveDOFValues(current_DOFs)
    return collision1 or collision2

  ### TODO ###  (not required but I found it useful)
  #######################################################
  # Pick a heuristic for 
  #######################################################
  def config_to_priorityqueue_tuple(self, dist, config, goals):
    # you can use either of these - make sure to replace the 0 with your
    # priority queue value!
    return (dist, config.tolist())
    #return (dist, self.convert_for_dict(config))


  #######################################################
  # Convert to and from numpy array to a hashable function
  #######################################################
  def convert_for_dict(self, item):
    item = np.array(item)*100.
    return tuple(item.astype(int))
    #return tuple(item)

  def convert_from_dictkey(self, item):
    #print "convert_from dictkey"
    #print np.array(item)/100.
    return np.array(item)/100.
    #return np.array(item)



  ### TODO:DONE ###  (not required but I found it useful)
  #######################################################
  # Initialize the movements you can apply in any direction
  # Don't forget to use TRANS_PER_DIR - the max distance you
  # can move any joint in a step (defined above)
  #
  #This function initializes a transition function that can be
  #use to generate 14 nearest neighbors
  #######################################################
  def init_transition_arrays(self):
    positive_transition = np.identity(7)*TRANS_PER_DIR; #Create a 7x7 np array with TRANS_PER_DIR down the diagonal
    negative_transition= positive_transition*-1; #Create a 7x7 np array with neg TRANS_PER_DIR down the diagonal
    self.transition_arrays = np.concatenate((positive_transition, negative_transition), axis = 0) #concatenate into 7x14 np array
    return 


  ### TODO:DONE ###  (not required but I found it useful)
  #######################################################
  # Take the current configuration and apply each of your
  # transition arrays to it
  #######################################################
  def transition_config(self, config):
    ######## SSR  ######
    new_configs = np.array([]) #define blank array
    for c in self.transition_arrays: #loop through columns
        new_configs = np.concatenate((new_configs, c + config), axis = 0) #assemble new configuration
    new_configs = np.reshape(new_configs, (14, 7)) #reshape
    return new_configs


  #######################################################
  # Takes in a list of points, and creates a trajectory
  # that goes between them
  #######################################################
  def points_to_traj(self, points):
    traj = openravepy.RaveCreateTrajectory(self.env,'')
    traj.Init(self.robot.GetActiveConfigurationSpecification())
    for idx,point in enumerate(points):
      traj.Insert(idx,point)
    openravepy.planningutils.RetimeActiveDOFTrajectory(traj,self.robot,hastimestamps=False,maxvelmult=1,maxaccelmult=1,plannername='ParabolicTrajectoryRetimer')
    return traj



  ### TODO:DONE ###  (not required but I found it useful)
  #######################################################
  # minimum distance from config to any goal in goals
  # distance metric: euclidean
  # returns the distance AND closest goal
  #######################################################
  def min_euclid_dist_to_goals(self, config, goals):
    # replace the 0 and goal with the distance and closest goal
    goals1 = np.array([])
    for g in goals:
    	eucd = np.linalg.norm(g-config)					#find eucledian distance between each of the goals 
	goals1 = np.append(goals1, np.append(g,eucd))			#append [goal eucledian distance] to new matrix of goals 
    goals1 = np.reshape(goals1,(np.size(goals1)/8,8))			#reshape goals into a 2D array with a column height of 8
    goals1 = np.array(sorted(goals1, key=lambda goals1:goals1[-1]))	#sort goals1 according to the last element
    close = goals1[0]							#closest goal is the first element in goals
    return close[-1], close[:7]						#return last element of closest goal (distance), first 7 elements(goal)



  ### TODO:DONE ###  (not required but I found it useful)
  #######################################################
  # minimum distance from config to any goal in goals
  # distance metric: manhattan
  # returns the distance AND closest goal
  #######################################################
  def min_manhattan_dist_to_goals(self, config, goals):
    # replace the 0 and goal with the distance and closest goal
    goals1 = np.array([])
    for g in goals:
    	man =np.sum(abs(g-config)) 
	goals1 = np.append(goals1, np.append(g, man))
    goals1 = np.reshape(goals1,(np.size(goals1)/8,8))
    goals1 = np.array(sorted(goals1, key=lambda goals1:goals1[-1]))
    #print goals1
    close = goals1[0]
    return close[-1], close[:7]
     
  





  #######################################################
  # close the fingers when you get to the grasp position
  #######################################################
  def close_fingers(self):
    self.taskmanip.CloseFingers()
    self.robot.WaitForController(0) #ensures the robot isn't moving anymore
    #self.robot.Grab(target) #attaches object to robot, so moving the robot will move the object now




if __name__ == '__main__':
    robo = RoboHandler()
    temp_goal = [ [0.93422050, -1.10221021, -0.2,  2.27275587, -0.22977831, -1.09393251, -2.23921746]]
    
    #robo.init_transition_arrays()
    #robo.search_to_goal_depthfirst(temp_goal)
    #robo.search_to_goal_breadthfirst(temp_goal)
    #robo.run_simple_problem() #runs the simple problem
    robo.run_difficult_problem()
    time.sleep(10000) #to keep the openrave window open
