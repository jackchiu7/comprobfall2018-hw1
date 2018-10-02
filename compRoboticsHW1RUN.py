# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:41:49 2018

@author: jackc
"""

import numpy as np
import math
import pandas as pd 
import matplotlib.patches as ptc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from shapely.geometry.point import Point
from shapely.geometry import LineString
from shapely.geometry import LineString
from shapely import geometry
from descartes.patch import PolygonPatch
import shapely.geometry.polygon as pg
import sys
import time
#import cv2

Infinity = sys.maxint

gridVertices = []
grid = []
visibilityGraph = []

def genVector(start, stop, step):
    tempArr = []
    while start <= stop:
        tempArr.append(start)
        start = start+step
    
    return tempArr

def generateGridVertices():
    global gridVertices
    gridVertices = []
    
    iVals = genVector(grid.minX, grid.maxX, grid.interval)
    jVals = genVector(grid.minY, grid.maxY, grid.interval)
    indI = 0
    indJ = 0
    for i in iVals:
        temp = []
        indJ = 0
        for j in jVals:
            tempNode = makeNode(i,j)
            tempNode.g = Infinity
            tempNode.gridIndex = [indI, indJ]
            temp.append(tempNode)
            indJ += 1
        indI += 1
        gridVertices.append(temp)
    
#    print(len(gridVertices))
    
def getIndex(xVal, yVal):
    i = abs(xVal-grid.minX)/grid.interval
    j = abs(yVal-grid.minY)/grid.interval
    i = (int)(i)
    j = (int)(j)
    return i,j

class Grid:
    obstacles = []
    geomPoly = []
    buffedPoly = []
    boundary = [] #[minX, maxX], [minY, maxY]
    minX = 0
    maxX = 1
    minY = 0
    maxX = 1
    interval = 1
    buff = 0
    
    
    def plot(self, ax, gridInterval = .25):
        ax.axis([self.minX-gridInterval, self.maxX+gridInterval, self.minY-gridInterval, self.maxY+gridInterval])
    
        for poly in self.buffedPoly:
            patch = PolygonPatch(poly, facecolor=[.3,0,0.5], edgecolor=[0,0,0], alpha=0.4, zorder=2)
            ax.add_patch(patch)

        
        for obs in self.obstacles:
            ax.add_patch(ptc.Polygon(obs.get_xy()))
        
        ax.plot([self.minX,self.maxX],[self.minY,self.minY],'b-',linewidth = 2)
        ax.plot([self.minX,self.maxX],[self.maxY,self.maxY],'b-',linewidth = 2)
        ax.plot([self.minX,self.minX],[self.minY,self.maxY],'b-',linewidth = 2)
        ax.plot([self.maxX,self.maxX],[self.minY,self.maxY],'b-',linewidth = 2)
        
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(gridInterval))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(gridInterval))
        ax.axis([self.minX-gridInterval, self.maxX+gridInterval,
                 self.minY-gridInterval, self.maxY+gridInterval])
        ax.grid(which = 'both', linewidth = .4)     

        
#        return ax
    def contains(self, nodePt):
        x = nodePt.x
        y = nodePt.y
        if x < self.minX + self.buff or x> self.maxX - self.buff:
            return True
        if y < self.minY + self.buff or y > self.maxY - self.buff:
            return True
#        for obs in self.obstacles:
#            if obs.get_path().contains_point([nodePt.x, nodePt.y]):
#                return True
        shapelyPt = Point(x,y)
        for poly in self.buffedPoly:
            if shapelyPt.within(poly):
                return True
            
        
                     
        return False
    
    
    def containsPt(self, point):
        x = point[0]
        y = point[1]
        if x < self.minX or x> self.maxX:
            return True
        if y < self.minY or y > self.maxY:
            return True
        for obs in self.obstacles:
            if obs.get_path().contains_point([x, y]):
                return True
        
        shapelyPt = Point(x,y)
        for poly in self.buffedPoly:
            if shapelyPt.within(poly):
                return True
        
        return False
    
    
    def intersectsLine(self,node1, node2):
        
        for poly in self.buffedPoly:
            l1 = LineString([[node1.x, node1.y],[node2.x, node2.y]])
            if l1.intersects(poly):
                return True
        return False

def makeGrid(fileName, buff = .4):
    global grid
    grid = Grid()
    grid.buff = buff
    file = open(fileName)
    lines = file.readlines()
    coords = lines[0].replace('(','').replace(')','').split()
    xVals = []
    yVals = []
    for xy in coords:
        xVals.append(xy.split(',')[0])
        yVals.append(xy.split(',')[1])
    
    grid.minX = (float)(min(xVals))
    grid.maxX = (float)(max(xVals))
    grid.minY = (float)(min(yVals))
    grid.maxY = (float)(max(yVals))
    
    
    
    
    lines = lines[2::] #delete first two lines
    
    
    #Reading obstacles
#    poly = ptc.Polygon(bounds)       #create polygon based on vertices 
#    poly.get_path().contains_point([2,2]) #see if point in polygon

    while lines[0] != '---\n':
        vertices = []
        ln = lines[0]
        coords = ln.replace('(','').replace(')','').split()
        for xy in coords:
            xVal = (float)(xy.split(',')[0])
            yVal = (float)(xy.split(',')[1])
            vertices.append([xVal,yVal])
            
        obs = ptc.Polygon(vertices)
        poly = geometry.Polygon(obs.get_xy())
        grid.geomPoly.append(poly)
        grid.obstacles.append(obs)
        if grid.buff != 0:
            grid.buffedPoly.append(pg.Polygon(poly.buffer(grid.buff, cap_style = 3, join_style = 2, mitre_limit = 2).exterior))            
        else:
            grid.buffedPoly.append(poly)
        lines = lines[1::]
    
#    grid.obsLineStrings, grid.obsVertices = createLineSegments()
        
    
    return grid


def getStartGoals(fileName):
    file = open(fileName)
    lines = file.readlines()
    ctr = 0 #to get to third section of text file
    startGoals = []
    while ctr != 2:
        if(lines[0] == '---\n'):
            ctr +=1
        lines = lines[1::]
        
    for ln in lines:
        coords = ln.replace('(','').replace(')','').split()    
        temp = []
        for xy in coords:
             x = (float)(xy.split(',')[0])
             y = (float)(xy.split(',')[1])
             temp.append([x,y])
        startGoals.append(temp)
    
    return startGoals




####################################
    #coding algorithm
####################################

class Node:
    parent = None
    gridIndex = []
    ifGoal = False
    g = 0
    x = 0
    y = 0
    
    def disp(self):
        print 'x: ',self.x, '   y: ', self.y,'   parent: ', self.parent.x, ', ',self.parent.y 
    
    def pt(self):
        return [self.x, self.y]
    
    def equals(self, node):
        if self.x == node.x and self.y == node.y:
            return True
        return False
    def parentContains(self,node):
        temp = self
        while(temp.parent != None):
            if temp.parent.equals(node):
                return True
        
        return False

def makeNode(x,y):
    node = Node()
    node.x = x
    node.y = y
    return node 


def func(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(seg1,seg2):   #uses [x, y]
    [A,B] = seg1
    [C,D] = seg2
    return func(A,C,D) != func(B,C,D) and func(A,B,C) != func(A,B,D)

def succ(node): 
    successors = []
#    print('node',node.x, node.y)
    tempNode = None
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            intersects = False
            iTemp = node.gridIndex[0]+i
            jTemp = node.gridIndex[1]+j
#            print('suck',iTemp, jTemp) 
            if iTemp >=0 and jTemp >=0 and iTemp < len(gridVertices) and jTemp < len(gridVertices[0]):
                tempNode = gridVertices[iTemp][jTemp]
#                print(tempNode.x, tempNode.y)
            else:
                continue
            if tempNode != node and not grid.contains(tempNode):
#                print('hi')
                intersects = grid.intersectsLine(node, tempNode)
                
                
                if not intersects:
                    successors.append(tempNode)
            
    
#    print(len(successors))
#    print(len(successors))
    return successors

def containsNode(array, node):
    for k in array:
        coords = k[0] #coords is a node 
        if node.x == coords.x and node.y == coords.y:
            return True
    
    return False


def removeNode(array, node):
    
    for k in array:
        coords = k[0]
        if node.x == coords.x and node.y == coords.y:
            array.remove(k)
    return array


def h(curr, goal):
    if grid.contains(curr):
        return Infinity
        
    temp1 = min(abs(curr.x-goal.x),abs(curr.y-goal.y))
    temp2 = max(abs(curr.x-goal.x),abs(curr.y-goal.y))
    heur = 2**(.5)*temp1 + temp2 - temp1
    
#    heur = math.sqrt((curr.x-goal.x)**2 + (curr.y-goal.y)**2)
        
    return heur    

def hStraightLine(curr,goal):
    heur = math.sqrt((curr.x-goal.x)**2 + (curr.y-goal.y)**2)
    
    return heur
    
def c(node1, node2): #straight line distance
    temp = math.sqrt((node1.x-node2.x)**2 + (node1.y-node2.y)**2)
    return temp


def aStar(startPts, goalPts, fda = False): #grids with vertices of .1 dist
    
    startI, startJ = getIndex(startPts[0],startPts[1])
    goalI, goalJ = getIndex(goalPts[0],goalPts[1])
    
    start = gridVertices[startI][startJ]
    goal = gridVertices[goalI][goalJ]
    
    start.parent = start
    start.g = 0
    
    fringe = []
    fringe.append([start, start.g+h(start,goal)])
    #append [node, key]
    
    #sort by second value in fringe (the key)
    fringe.sort(key=lambda tup: tup[0].g, reverse = True) 
    fringe.sort(key=lambda tup: tup[1], reverse = True) 
    closed = [] #closed = list [node, key]
    
    
    while fringe:
        s = fringe.pop()    #s = [node, key]
        
        

        
        if s[0] == goal:
            return s[0]
            
        closed.append(s)
        for sNew in succ(s[0]):    #sNew are nodes
            if not containsNode(closed, sNew):
                if not containsNode(fringe,sNew):
                    sNew.g = Infinity
                    sNew.parent = None
                if not fda:
                    fringe = updateVertex(s[0],sNew,fringe,goal)
                else:
                    fringe = updateVertexFDA(s[0],sNew,fringe,goal)
    return None
    
    
    
def updateVertex(s,sNew,fringe,goal):
#    print(sNew.x, sNew.y, sNew.g)    
    
#    print(s.x, s.y)
    if s.g + c(s,sNew) < sNew.g:
        
        sNew.g = s.g + c(s,sNew)
        sNew.parent = s
        
        if containsNode(fringe,sNew):
            fringe = removeNode(fringe,sNew)
#            print('if statement',sNew.x, sNew.y, sNew.g)
        fringe.append([sNew, sNew.g+h(sNew,goal)])
        fringe.sort(key=lambda tup: tup[1], reverse = True) 
    
    
    
    return fringe


def updateVertexFDA(s,sNew,fringe,goal):
    if line_of_sight(s.parent, sNew):
        if (s.parent.g + c(s.parent,sNew)) < sNew.g:
            sNew.g = s.parent.g + c(s.parent, sNew)
            sNew.parent = s.parent
            if containsNode(fringe,sNew):
                removeNode(fringe, sNew)
            fringe.append([sNew, sNew.g + h(sNew,goal)])
            fringe.sort(key=lambda tup: tup[0].g, reverse = True) 
            fringe.sort(key=lambda tup: tup[1], reverse = True) 
    else:
        if s.g + c(s,sNew) < sNew.g:
            sNew.g = s.g + c(s,sNew)
            sNew.parent = s
            if containsNode(fringe, sNew):
                removeNode(fringe, sNew)
            fringe.append([sNew, sNew.g + h(sNew,goal)])
            fringe.sort(key=lambda tup: tup[0].g, reverse = True) 
            fringe.sort(key=lambda tup: tup[1], reverse = True) 
    
    return fringe
    

def line_of_sight(s, s2):
    if grid.intersectsLine(s,s2):
        return False
    else:
        return True

def cost(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)


def pathIntersect(pt1,pt2,grid):
    for obs in grid.obstacles:
        coords = obs.get_xy()
        for a in range(0,len(coords)-1):
            if intersect([coords[a],coords[a+1]],[pt1,pt2]):
                return True

    return False

class GraphNode:
    x = 0
    y = 0
    h = Infinity
    c = Infinity
    prev = None
    neighbors = []
    
    def disp(self):
        print 'x: ',self.x, ', y: ', self.y,'   parent: ', self.prev.x, ',',self.prev.y 

def makeGraphNode(x,y): #give it x,y coordinates
    gnode = GraphNode()
    gnode.x = x
    gnode.y = y
    gnode.neighbors = []
    return gnode
            
def makeVisbilityGraph(startPts, goalPts):
    
    start = makeGraphNode(startPts[0], startPts[1])
    goal = makeGraphNode(goalPts[0], goalPts[1])
    
    visGraph = [start,goal]   #create the graph 
    for poly in grid.buffedPoly: #add all the vertices into the graph
        coordList = list(zip(*poly.exterior.coords.xy))
        for i in range(len(coordList)-1):
            coords = coordList[i]
            if coords[0] < grid.maxX - grid.buff and coords[0] > grid.minX + grid.buff and coords[1] < grid.maxY - grid.buff and coords[1] > grid.minY + grid.buff:
                temp = makeGraphNode(coords[0],coords[1])
                visGraph.append(temp)
    
    for node1 in visGraph: #fill out the neighbors (drawing edges)
        for node2 in visGraph:
            if node1.x == node2.x and node1.y == node2.y:
                continue
            if line_of_sight_2(node1,node2):
                node1.neighbors.append(node2)
    return visGraph

def createPathVisGraph(visGraph):
    visGraph[0].c = 0  #setting source cost to 0
    goal = visGraph[1] #goal node is always second
    #implement dijkstra's
    fringe = []
    
    for node in visGraph:
        fringe.append([node,node.c])
    #sort by second value in fringe (the key)
    #dijkstra
     
    while fringe != []:
        fringe.sort(key=lambda tup: tup[1])
        curr = fringe.pop(0)[0]
        
        if curr.x == goal.x and curr.y == goal.y:
            return curr
        
        for neighbor in curr.neighbors:
            cost = c(curr,neighbor) + curr.c
            if cost < neighbor.c:
                fringe = fringeRemove(neighbor, fringe)
                neighbor.c = cost
                neighbor.prev = curr
                fringe.append([neighbor, neighbor.c])
    
                    
    return None

def createVisPathAStar(visGraph):
    visGraph[0].c = 0
    goal = visGraph[1]
    
    visGraph[0].prev = visGraph[0]
    fringe = []
    fringe.append([visGraph[0],visGraph[0].c + hStraightLine(visGraph[0],goal)])
    closed = []
    
    while fringe != []:
        curr = fringe.pop(0)
        
        if curr[0].x == goal.x and curr[0].y == goal.y:
            return curr[0]
        
        closed.append(curr)
        
        for neighbor in curr[0].neighbors:
            if not containsNode(closed, neighbor):
                if not containsNode(fringe, neighbor):
                    neighbor.c = Infinity
                    neighbor.prev = None
            
            fringe = updateVertex2(curr[0],neighbor,fringe,goal)
        
    return None
                
                    
def updateVertex2(curr, neighbor, fringe, goal):
    if curr.c + c(curr, neighbor) < neighbor.c:
        neighbor.c = curr.c + c(curr,neighbor)
        neighbor.prev = curr
        if containsNode(fringe, neighbor):
            fringe = removeNode(fringe, neighbor)
        fringe.append([neighbor, neighbor.c + hStraightLine(neighbor,goal)])
        fringe.sort(key=lambda tup: tup[1])
        
    return fringe                
        

def printNeighbors(fringe):
    for l1 in fringe:
        print 'Curr: (' + str(l1.x) + ',' + str(l1.y) + ')'
        for neighbor in l1.neighbors:
            print 'Neigbor: (' + str(neighbor.x) + ',' + str(neighbor.y) + ')'
        print '\n'

def fringeRemove(node, fringe):
    for l1 in fringe:
        if l1[0].x == node.x and l1[0].y == node.y:
            fringe.remove(l1)
    return fringe

def line_of_sight_2(node1,node2):
    l1 = LineString([[node1.x, node1.y],[node2.x, node2.y]])
    for poly in grid.buffedPoly:
        if l1.intersects(poly) and not l1.touches(poly):
            return False
    
    return True

def main(filename, start, goal, fdaVal = False, vis = False): 
    global grid
    grid = makeGrid(filename, buff = .2)
    
    interval = .25
    grid.interval = interval
        
    generateGridVertices()
        

        
    if vis == False: 
        startTime = time.time()
        result = aStar(start, goal, fda = fdaVal) 
        endTime = time.time()
        elapsedTime = endTime-startTime
            
        pt1 = result
        tempPath = [pt1]
        if pt1 != None:
            while pt1.parent != pt1:
                pt2 = pt1.parent
                tempPath.append(pt2)
                pt1 = pt2
        
    else:
        startTime = time.time()
        visGraph = makeVisbilityGraph(start, goal)
        visResult = createVisPathAStar(visGraph)
        endTime = time.time()
        tempTime = endTime-startTime
        elapsedTime = tempTime
            
            
        ptr = visResult
        visPath = [ptr]

        if(ptr == None):
            print 'No path found'
            return
            
        while ptr.prev != ptr:
            ptr2 = ptr.prev
            visPath.append(ptr2)
            ptr = ptr2
            
        tempPath = visPath

        
    return grid, tempPath, elapsedTime
    

if __name__ == '__main__':
    main('grid1.txt')
    
    
    
    
    
    
    
    
    
    
    
    
    
    