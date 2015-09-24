import numpy as np
import math
import time

from sympy import *
from sympy.mpmath import *
from sympy.matrices import Matrix

from collections import namedtuple
DHParameters = namedtuple('DHParameters', 'alpha d theta r offset')

def identity(n):
    return np.eye(n)

def skew(u):
    u_skew = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    return u_skew

def rotation_x(theta):
    Rx = np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
    return Rx

def rotation_y(theta):
    Ry = np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
    return Ry

def rotation_z(theta):
    Rz = np.array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]])
    return Rz

def rotation_dx(theta):
    Rx = np.array([[1,0,0],[0,-math.sin(theta),-math.cos(theta)],[0,math.cos(theta),-math.sin(theta)]])
    return Rx

def rotation_dy(theta):
    Ry = np.array([[-math.sin(theta),0,math.cos(theta)],[0,1,0],[-math.cos(theta),0,-math.sin(theta)]])
    return Ry

def rotation_dz(theta):
    Rz = np.array([[-math.sin(theta),-math.cos(theta),0],[math.cos(theta),-math.sin(theta),0],[0,0,1]])
    return Rz

def rotation(u,theta):
    vect_u = np.array([u]).T
    rot = np.dot(vect_u,vect_u.T)*(1-math.cos(theta)) + identity(3)*math.cos(theta) + skew(vect_u)*math.sin(theta)
    return rot

def transformation(rot,vect):
    T = np.hstack((rot,np.array([vect]).T))
    T = np.vstack((T,[0,0,0,1]))
    return T

def inverse_transform(transform):
    inv_transform = np.zeros((4,4))
    rot = transform[:-1,:-1].T
    inv_transform[:-1,:-1] = rot
    inv_transform[:-1,-1] = np.dot(-rot,transform[:-1,-1])
    inv_transform[-1,-1] = 1
    return inv_transform

def transform_point(point,transform):
    temp = np.hstack((np.array(point),1))
    t_temp = np.dot(transform,temp)
    Tpoint = np.transpose(t_temp[:-1]).tolist()
    return Tpoint 

def transform_trajectory(traj, transform):
    t_traj = []
    for x in traj:
        if len(x) == 2:
            x = np.hstack((np.array(x),0))
        temp = np.hstack((np.array(x),1))
        t_temp = np.dot(transform,temp)
        if len(traj[0]) == 2:
            t_traj.append(np.transpose(t_temp[:-2]).tolist())
        else:
            t_traj.append(np.transpose(t_temp[:-1]).tolist())
    return t_traj

def rotate_trajectory(traj, axis, theta):
    rot = rotation(axis,theta)
    vect = [0,0,0]
    trans = transformation(rot,vect)
    return transform_trajectory(traj,trans)

def quaternion_to_euler(q):
    euler = np.zeros(3)
    if q[0]*q[1] + q[2]*q[3] == 0.5:
        euler[0] = 2*math.atan2(q[0]*q[3])
        euler[2] = 0
    elif q[0]*q[1] + q[2]*q[3] == -0.5:
        euler[0] = -2*math.atan2(q[0]*q[3])
        euler[2] = 0
    else:
        euler[0] = math.atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]**2+q[2]**2))
        euler[2] = math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    euler[1] = math.asin(2*(q[0]*q[2]-q[3]*q[1]))
    return euler