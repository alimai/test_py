# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:41:53 2024

@author: maishengli
"""


import numpy as np
from numpy import linalg
import quaternion  # 导入 numpy-quaternion
import open3d as o3d

max_bias = 0.1

def adjust_mesh(mesh, ratio):    
    point_array = np.asarray(mesh.vertices) 
    center = np.mean(point_array, axis=0)
    point_array -= center
    point_array[:,0] *= ratio#0.4 
    point_array[:,1] *= ratio#0.4 
    point_number = len(point_array)
    for vertex_index in range(point_number): 
        mesh.vertices[vertex_index] = point_array[vertex_index]+center

def test_rotate(pcd, theta_target):
    theta_used = theta_target * 0.5
    test_q = np.quaternion(np.cos(theta_used), 0, 0, np.sin(theta_used))
    
    test_vec = np.array([1, 0, 0])
    result_vec = quaternion.rotate_vectors(test_q, test_vec)
    print(result_vec)
    
    rotation_axis = np.array([0, 0, 1])  # 绕z轴旋转
    rotation_angle = theta_target
    rotation_quaternion = quaternion.from_rotation_vector(rotation_angle * rotation_axis)
    result_vec2 = quaternion.rotate_vectors(rotation_quaternion, test_vec)
    print(result_vec2)
    
    q4 = quaternion.from_euler_angles([0, 0, theta_target])
    result_vec3 = quaternion.rotate_vectors(q4, test_vec)
    print(result_vec3)
    
    for vertex_index in range(len(pcd.points)):
        pcd.points[vertex_index] = quaternion.rotate_vectors(test_q, np.asarray(pcd.points[vertex_index]))

def test_rotate2(mesh, loc):
    rotation_quaternion, translation_vector = loc
    for vertex_index in range(len(mesh.vertices)):
        mesh.vertices[vertex_index] = quaternion.rotate_vectors(rotation_quaternion, np.asarray(mesh.vertices[vertex_index])) + translation_vector

def test_regist(mesh, mesh0): 
    point_array0 = np.asarray(mesh0.vertices)
    v_center0 = np.mean(point_array0, axis=0)
    bias_array0 = point_array0 - v_center0
    
    point_number = len(mesh.vertices)
    point_array = np.asarray(mesh.vertices)
    for k in range(20):   
        v_center = np.mean(point_array, axis=0)
        bias_array = point_array - v_center
     
        b_bias_array = bias_array0 - bias_array#key points
        b_bias_array = np.clip(b_bias_array, -max_bias, max_bias)  # 将每个元素限制在 [-0.01, 0.01] 之间
        
        next_array = point_array + b_bias_array
        v_next_center = np.mean(next_array, axis=0)
        bias_next_array = next_array - v_next_center
        
        mH = np.zeros((3, 3))
        for i in range(point_number):
            v_cur_matrix =  np.transpose([bias_array[i]])
            v_next_matrix = np.array([bias_next_array[i]])
            mH += v_cur_matrix.dot(v_next_matrix) 
        mH /= point_number
        U, s, V = linalg.svd(mH)
        svd_rot = V.T.dot(U.T)  # V.T instead of V
        #svd_mov = v_center0 - svd_rot.dot(v_center)
        svd_mov1 = v_center - svd_rot.dot(v_center)
        svd_mov2 = v_center0 - v_center
        svd_mov2 = np.clip(svd_mov2, -max_bias, max_bias)
        svd_mov = svd_mov1 + svd_mov2
    
        for vertex_index in range(point_number): 
            point_array[vertex_index] = svd_rot.dot(point_array[vertex_index]) + svd_mov
        
    for vertex_index in range(point_number): 
        mesh.vertices[vertex_index] = point_array[vertex_index]

def test_regist2(mesh, points_ellipt): 
    point_array0 = np.asarray(mesh.vertices)
    
    loc_target = [np.eye(3), [0.0, 0.0, 0.0]]  
    point_array = point_array0[::30]
    point_number = len(point_array)
    for k in range(10): 
        v_center = np.mean(point_array, axis=0)
        
        b_bias_array = v_center - point_array     
        b_bias_array = np.clip(b_bias_array, -max_bias, max_bias)  
        b_bias_array[:, 2] *= 0.0#0.3

        e_bias_array = points_ellipt[0]-point_array
        for n in range(point_number): 
            point_m = point_array[n]
            point_e0 = points_ellipt[0]
            dist= linalg.norm(point_e0-point_m)
            for point_e in points_ellipt:
                dist_n = linalg.norm(point_e - point_m)
                if(dist_n < dist):
                    dist = dist_n
                    e_bias_array[n] = point_e - point_m
        e_bias_array = np.clip(e_bias_array, -max_bias, max_bias)# 将每个元素限制在 [-0.01, 0.01] 之间

        next_array = point_array + b_bias_array + e_bias_array#
        v_next_center = np.mean(next_array, axis=0)
        
        mH = np.zeros((3, 3))
        for i in range(point_number):
            v_cur_matrix =  np.transpose([point_array[i]])
            v_next_matrix = np.array([next_array[i]])
            mH += v_cur_matrix.dot(v_next_matrix) 
        mH /= point_number
        U, s, V = linalg.svd(mH)
        svd_rot = V.T.dot(U.T)  # V.T instead of V
        svd_mov = v_next_center - svd_rot.dot(v_center)
    
        loc_target[0] = svd_rot.dot(loc_target[0])
        loc_target[1] = svd_rot.dot(loc_target[1]) + svd_mov
        for index in range(point_number): 
            point_array[index] = svd_rot.dot(point_array[index]) + svd_mov
        
    svd_rot_i = linalg.inv(loc_target[0])
    for index in range(point_number): 
        point_array[index] = svd_rot_i.dot(point_array[index] - loc_target[1])
    for vertex_index in range(len(point_array0)): 
        mesh.vertices[vertex_index] = loc_target[0].dot(point_array0[vertex_index])+ loc_target[1]
        
def disp_open3d(geometry, default=False):
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    geometry.append(coor)
    if default:
        o3d.visualization.draw_geometries(geometry, mesh_show_back_face=True, mesh_show_wireframe=True)
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.mesh_show_wireframe = False
        opt.point_show_normal = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        for i in geometry:
            vis.add_geometry(i)
        vis.poll_events()
        vis.update_renderer()    
        vis.run()

if __name__ == "__main__":
    
    files_path = ["D:\\material\\邻接面优化\\test\\a1\\BC01000070937\\cache\\11.stl",
                  "D:\\material\\邻接面优化\\test\\a1\\BC01000070937\\cache\\14.stl",
                  "D:\\material\\邻接面优化\\test\\a1\\BC01000070937\\cache\\23.stl"]
    
    mesh0 = o3d.io.read_triangle_mesh(files_path[0]) 
    adjust_mesh(mesh0, 0.4)
    color = [0, 0, 0.1]
    mesh0.paint_uniform_color(color)
    
    mesh1 = o3d.io.read_triangle_mesh(files_path[1])    
    adjust_mesh(mesh1, 0.4)
    color1 = [0, 0.1, 0]
    mesh1.paint_uniform_color(color1)
    
    mesh2 = o3d.io.read_triangle_mesh(files_path[2])  
    adjust_mesh(mesh2, 0.4)  
    color2 = [0.1, 0, 0]
    mesh2.paint_uniform_color(color2)
    
    points_ellipt = []
    length_ellipt = 20
    a_ellipt = 20
    b_ellipt = 30
    a_squre = a_ellipt * a_ellipt
    b_squre = b_ellipt * b_ellipt
    c_square = a_squre * b_squre
    center_ybias = 10
    pcd_ellipt = o3d.geometry.PointCloud()  
    for i in range(length_ellipt):
        x = -a_ellipt + 2 * a_ellipt * i / (length_ellipt - 1)
        y = -np.sqrt(b_squre - x * x * b_squre / a_squre) + center_ybias
        points_ellipt += [[x, y, 0.0]]
    pcd_ellipt.points = o3d.utility.Vector3dVector(points_ellipt)
    colors_ellipt = [[0.1, 0.0, 0.1] for i in range(length_ellipt)]
    pcd_ellipt.colors = o3d.utility.Vector3dVector(colors_ellipt)
          
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    axis_actor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(axis_actor)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.3, 0.3, 0.3])
    render_option.light_on = True    
    
    vis.add_geometry(pcd_ellipt)
    mesh0.compute_vertex_normals()
    mesh1.compute_vertex_normals()
    mesh2.compute_vertex_normals()
    #vis.add_geometry(mesh0)
    vis.add_geometry(mesh1)
    vis.add_geometry(mesh2)
      
    theta = (np.pi * 0.5) * 0.5
    loc1 = [np.quaternion(), [0.0, 0.0, 0.0]]  
    loc1[0].real = np.cos(theta * 0.5)
    loc1[0].imag = np.array([1.0, 1.0, 0.0]) * np.sin(theta * 0.5)
    loc1[1] = [0.0, -10.0, 5.0]    
    mesh0_2 = o3d.geometry.TriangleMesh(mesh0.vertices, mesh0.triangles)
    test_rotate2(mesh0_2, loc1)  
    mesh0_2.paint_uniform_color(color2)
    mesh0_2.compute_vertex_normals()
    geo_handle_0 = vis.add_geometry(mesh0_2)
    vis.run()
    
    '''
    '''
    for k in range(10):
        #test_regist(mesh0_2, mesh0) 
        test_regist2(mesh0_2, points_ellipt) 
        test_regist2(mesh1, points_ellipt) 
        test_regist2(mesh2, points_ellipt) 
        mesh0_2.compute_vertex_normals()
        mesh1.compute_vertex_normals()
        mesh2.compute_vertex_normals()
    
        print("cycle ", k)
        vis.update_geometry(mesh0_2)
        vis.update_geometry(mesh1)
        vis.update_geometry(mesh2)
        vis.run()
    
    input("end.")
    vis.destroy_window()
