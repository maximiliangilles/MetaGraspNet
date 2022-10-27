#!/usr/bin/env python
# Author : IFL, KIT

import numpy as np
import math
import trimesh
import random
import transforms3d

class Robust_Vacuum_Grasp_Model:
    def __init__(self):
        
        # Variables defined by user
        self.r = float()                # radius of the inner contact zone in cm
        self.num = int()                # number of mass-points around the radius
        self.young = float()            # youngs modulus of the gripper material in GPa
        self.my = float()               # friction coefficient between gripper and surface
        self.pressure_diff = float()    # pressure difference between the vacuum and the atmospheric pressure in bar
        self.h = float()                # possible height differences in cm
        
        # Fixed variables
        self.approach_height = 3        # height from which the gripper is projected onto the surface
        self.c1 = 0.03768               # proportional factor for the compressed springs
        self.c2 = 0.06241               # proportional factor for the remaining springs
        
        # Remaining variables wich are used in the class
        self.obj_mesh = ()              # scale is required in cm!
        self.grasping_point = ()
        self.surface_normal = ()
        
        self.mass_points = ()
        self.random_point = ()
        self.contact_points = []
        self.random_normal = ()
        self.approach_vectors = []
        self.faces_idx = ()
        self.lokal_surface_normals = ()
        
        self.mass_points_pregrasp = ()
        self.center_pregrasp = ()
        self.projected_mass_points = ()
        self.projected_center = ()
        
        self.mass_points_normal = ()
        
        self.fe = ()
        self.ff = ()
        self.fen_val = ()
        self.ffn_val = ()
        
        self.seal = True
        self.binary_score = []
    
####################    Main model    ####################
    
    # creates the mass points of the spring-mass-system with same distances on the contact ring defined by r
    def create_spring_cup(self):
        self.mass_points = [[self.r * math.cos((k-1+1/2) * 2 * math.pi/self.num), self.r * math.sin((k-1+1/2) * 2 * math.pi/self.num), 0.0] for k in range(self.num)]
    
    # transforms the mass-points to the contact point
    def rotate_spring_cup(self):
                
        # Rotate the mass points to the random point
        H = trimesh.geometry.align_vectors(self.random_normal, [0,0,1], False)
        mass_points_rot = [np.dot(H.T[:3, :3], v) for v in self.mass_points]
        
        # Translate the mass points to the random point
        self.mass_points_pregrasp = [(v + self.random_point + self.random_normal * self.approach_height) for v in mass_points_rot]
        self.center_pregrasp = self.random_point + self.random_normal * self.approach_height
    
    # projects the mass-points on the object surface
    def project_spring_cup(self):
                
        # Project the mass points on the surface
        ray_directions = [(-1) * self.random_normal for k in range(len(self.mass_points_pregrasp))]
        self.faces_idx, _, self.projected_mass_points = trimesh.ray.ray_pyembree.RayMeshIntersector(self.obj_mesh).intersects_id(
            ray_origins = self.mass_points_pregrasp,
            ray_directions = ray_directions,
            multiple_hits = False,
            return_locations = True
        )
        
        # Project the center on the surface
        _, _, self.projected_center = trimesh.ray.ray_pyembree.RayMeshIntersector(self.obj_mesh).intersects_id(
            ray_origins = [self.center_pregrasp],
            ray_directions = [(-1) * self.random_normal],
            multiple_hits = False,
            return_locations = True
        )
        
        # Calculate distances of mass points in surface normal direction
        self.mass_points_normal = [np.dot(self.projected_mass_points[j],self.random_normal) for j in range(len(self.projected_mass_points))]
                    
        # Error handling (not all points are projected)    
        if len(self.projected_mass_points) < self.num:
            self.seal = False
        if len(self.projected_center) < 1:
            self.seal = False
        else:
            self.contact_points.append(self.projected_center[0])
            
    # calculates the pressure distribution on the mass-points dependending on the distances along the normal vector
    def pressure_distribution(self):
        f_pressure = self.pressure_diff * 10**5 * math.pi * self.r**2 * 10**(-4)
        
        # define the spring stiffenes of the compressed springs with proportional factor
        k_comp = self.young * 10**6 * math.pi * self.c1 / self.num
        
        # Calculate the mass points distannces in normal direction
        mass_points_normal_max = max(self.mass_points_normal)
        dist = [(mass_points_normal_max - self.mass_points_normal[j]) * 10**(-2) for j in range(len(self.mass_points_normal))]
        sum_dist = sum(dist)
        
        # Calculate the maximum compression
        delta_l = ((f_pressure/k_comp) + sum_dist)/self.num
        
        # Calculate the compression force on each point
        self.fe = [(k_comp * (delta_l - dist[j]) * self.random_normal) for j in range(len(dist))]
    
    # calculates the forces on each mass-point caused by the remaining springs
    def force_calculation(self):

        # Calculate length pregrasp
        l_struct = 2 * self.r * math.sin(math.pi/self.num)
        l_flex = 2 * self.r * math.sin((math.pi * 2)/self.num)
        l_shear = 2 * self.r
        
        # Calculate length changes
        diff_vec_struct = [self.projected_mass_points[k-1] - self.projected_mass_points[k] for k in range(len(self.projected_mass_points))]
        diff_vec_flex = [self.projected_mass_points[k-2] - self.projected_mass_points[k] for k in range(len(self.projected_mass_points))]
        diff_vec_shear = [self.projected_mass_points[k-int(self.num/2)] - self.projected_mass_points[k] for k in range(len(self.projected_mass_points))]
        
        dl_struct = [(np.linalg.norm(diff_vec_struct[k])-l_struct)*(diff_vec_struct[k]/np.linalg.norm(diff_vec_struct[k])) for k in range(len(diff_vec_struct))]
        dl_flex = [(np.linalg.norm(diff_vec_flex[k])-l_flex)*(diff_vec_flex[k]/np.linalg.norm(diff_vec_flex[k])) for k in range(len(diff_vec_flex))]
        dl_shear = [(np.linalg.norm(diff_vec_shear[k])-l_shear)*(diff_vec_shear[k]/np.linalg.norm(diff_vec_shear[k])) for k in range(len(diff_vec_shear))]
        
        # define the spring stiffenes of the remaining springs with proportional factor
        k_struct = (self.young * 10**6 * math.pi * self.c2)/(self.num * l_struct)
        k_flex = (self.young * 10**6 * math.pi * self.c2)/(self.num * l_flex)
        k_shear = (self.young * 10**6 * math.pi * self.c2)/(self.num * l_shear)
         
        # Calculate different forces per point
        f_struct = k_struct * np.array(dl_struct) * 10**(-2)
        f_flex = k_flex * np.array(dl_flex) * 10**(-2)
        f_shear = k_shear * np.array(dl_shear) * 10**(-2)
        
        fp_struct = np.array([f_struct[i-1] - f_struct[i] for i in range(len(f_struct))])
        fp_flex = np.array([f_flex[i-2] - f_flex[i] for i in range(len(f_flex))])
        fp_shear = f_shear
        
        # Sum up forces per point
        self.ff = [np.array(fp_flex[k]) + np.array(fp_struct[k-1]) + np.array(fp_shear[k-2]) for k in range(len(fp_struct))]
        
        # calculate the part of the forces in normal direction
        self.lokal_surface_normals = [self.obj_mesh.face_normals[self.faces_idx[j]] for j in range(len(self.faces_idx))]
        self.fen_val = [np.dot(self.fe[j],self.lokal_surface_normals[j]) for j in range(len(self.fe))]
        self.ffn_val = [np.dot(self.ff[j],self.lokal_surface_normals[j-2]) for j in range(len(self.ff))]
    
    # compares the calculated forces to check if contact is sealed    
    def force_comparison(self):
        for k in range(len(self.ffn_val)):
            # check forces along the normal
            if self.ffn_val[k] > self.fen_val[k-2]:
                self.seal = False            
        
    # combines all functions above    
    def complete_calculation(self, obj_mesh, r, h, num, grasping_point, face_index, young, my, pressure_diff):
        self.approach_vectors = []
        self.contact_points = []
        self.binary_score = []
        
        self.obj_mesh = obj_mesh
        self.r = r
        self.h = h
        self.num = num
        self.grasping_point = grasping_point
        self.young = young
        self.pressure_diff = pressure_diff
        self.my = my
        self.surface_normal = self.obj_mesh.face_normals[face_index]
        
        ai = np.random.normal(0, 0.1, 10)
        aj = np.random.normal(0, 0.1, 10)
        ak = np.random.normal(0, 0.1, 10)
        bx = np.random.normal(self.grasping_point[0], r/4, 10)
        by = np.random.normal(self.grasping_point[1], r/4, 10)
        bz = np.random.normal(self.grasping_point[2], r/4, 10)
        
        for j in range(10):
            self.seal = True
            
            R = transforms3d.euler.euler2mat(ai[j], aj[j], ak[j])
            self.random_normal = np.dot(R, self.surface_normal.T)
            self.random_point = [bx[j],by[j],bz[j]]
            
            self.create_spring_cup()
            self.rotate_spring_cup()        
            self.project_spring_cup()
            
            self.approach_vectors.append(self.random_normal)
            
            if self.seal == True:
                self.simple_collison_detection()
                self.hill_check()
                self.hole_check()
                  
            if self.seal == True:
                self.pressure_distribution()
                self.force_calculation()
                self.force_comparison()
            
            if self.seal == True:
                self.binary_score.append(1)
            else:
                self.binary_score.append(0)
                
        return self.binary_score, self.approach_vectors, self.contact_points
        
####################    Extensions to prevent typical grasping problems which are not detected by the model    ####################
    
    # checks if there is an object along the surface normal with gripper size
    def simple_collison_detection(self):
        ray_directions = [self.random_normal for k in range(len(self.mass_points_pregrasp))]
        hit = trimesh.ray.ray_pyembree.RayMeshIntersector(self.obj_mesh).intersects_any(self.mass_points_pregrasp,ray_directions)
        if hit.any() == True:
            self.seal = False
    
    # checks if the center point is located significant higher than the projected mass-points
    def hill_check(self):
        contact_normal = np.dot(self.projected_center, self.random_normal)
        mass_points_normal_min = min(self.mass_points_normal)
        if contact_normal - mass_points_normal_min > self.h:
            self.seal = False
    
    # checks if there are holes in the object inside the contact zone which prevent the seal by projecting points on the surface 
    def hole_check(self):
        sampled_points = [[(self.r * math.sqrt(random.random())) * math.cos(2 * math.pi * random.random()), (self.r * math.sqrt(random.random())) * math.sin(2 * math.pi * random.random()), 0] for k in range(1000)]
        pointcloud = trimesh.points.PointCloud(sampled_points)
        
        # Rotate the sampled points
        H = trimesh.geometry.align_vectors(self.random_normal, [0,0,1], False)
        sample_rot = [np.dot(H.T[:3, :3], sample) for sample in sampled_points]
        
        # Translate the mass points
        sample_pregrasp = [(sample + self.random_point + self.random_normal * self.approach_height) for sample in sample_rot]
        
        # project points to object
        ray_directions = [(-1) * self.random_normal for k in range(len(sample_pregrasp))]
        _, _, projected_samples = trimesh.ray.ray_pyembree.RayMeshIntersector(self.obj_mesh).intersects_id(
            ray_origins = sample_pregrasp,
            ray_directions = ray_directions,
            multiple_hits = False,
            return_locations = True
        )
        
        if len(projected_samples) < len(sample_pregrasp):
            self.seal = False
