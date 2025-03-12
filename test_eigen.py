# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:15:29 2023

@author: maishengli
"""

import numpy as np
import quaternion # 导入 numpy-quaternion
import open3d as o3d


        
a_squre = 10 * 10
b_squre = 30 * 30
c_square = a_squre*b_squre
center_ybias = 10

def test_rotate(pcd, theta_target):
    theta_used = theta_target * 0.5
    test_q=np.quaternion(1,0,0,0)
    test_q.real = np.cos(theta_used)
    test_q.imag = np.array([0,0,1.0])*np.sin(theta_used)
    
    test_vec = np.array([1,0,0])
    result_vec = quaternion.rotate_vectors(test_q,test_vec)
    print(result_vec)
    
    rotation_axis = np.array([0, 0, 1])  # 绕z轴旋转
    rotation_angle = theta_target#np.radians(45)
    rotation_quaternion = quaternion.from_rotation_vector(rotation_angle * rotation_axis)
    result_vec2 = quaternion.rotate_vectors(rotation_quaternion,test_vec)
    print(result_vec2)
    
    #q3 = quaternion.from_rotation_matrix([[1,2,3],[1,2,3],[1,2,3]])
    q4 = quaternion.from_euler_angles([0,0,theta_target])
    result_vec3 = quaternion.rotate_vectors(q4,test_vec)
    print(result_vec3)
    
    
    #print(test_q*test_vec)
    #print(test_vec*test_q)
    #print(test_q*test_q.conjugate())
    #print(test_q*test_q)
    
    for vertex_index in range(len(pcd.points)):
        #break
        #pcd.points[vertex_index][1] += -10.0
        pcd.points[vertex_index] = quaternion.rotate_vectors(test_q,np.asarray(pcd.points[vertex_index]))
        #print(np.asarray( pcd[vertex_index]))

def test_rotate2(mesh, loc):
    for vertex_index in range(len(mesh.vertices)):
        #print(mesh.vertices[0])
        mesh.vertices[vertex_index] = quaternion.rotate_vectors(loc[0],np.asarray(mesh.vertices[vertex_index])) + loc[1]
        #ttt = svd_rot.dot(np.transpose([point_array[vertex_index]]))+svd_mov
        #mesh.vertices[vertex_index] = np.transpose(ttt)[0]
        #print(mesh.vertices[0])

        
def test_regist(mesh, mesh0): 
    point_array0 = np.asarray(mesh0.vertices)
    v_center0 = np.mean(point_array0, axis=0)
    bias_array0 = point_array0 - v_center0
    
    point_number = len(mesh.vertices)
    point_array = np.asarray(mesh.vertices)
    for k in range(20):   
        v_center = np.mean(point_array, axis=0)
        bias_array = point_array - v_center
     
        b_bias_array = bias_array0 - bias_array
        for b_bias in b_bias_array:
            for i in range(3):
                if(b_bias[i] > 0.01):
                    b_bias[i] = 0.01
                elif(b_bias[i] < -0.01):
                    b_bias[i] = -0.01
        next_array = point_array + b_bias_array
        v_next_center = np.mean(next_array, axis=0)
        bias_next_array = next_array - v_next_center
        
        mH = np.zeros((3, 3))
        for i in range(point_number):
            v_cur_matrix =  np.transpose(np.asarray([bias_array[i]])) #np.array([bias_array[i]]).T
            v_next_matrix = np.array([bias_next_array[i]])
            mH += v_cur_matrix.dot(v_next_matrix) 
        mH /= point_number
        U,s,V = np.linalg.svd(mH)
        svd_rot = V.T.dot(U.T)
        svd_mov = np.transpose([v_center0]) - svd_rot.dot(np.transpose([v_center]))
        #svd_mov = np.transpose(svd_mov)
        #svd_mov = svd_mov[0]
        print(svd_mov)
    
        for vertex_index in range(point_number): 
            ttt = svd_rot.dot(np.transpose([point_array[vertex_index]]))+svd_mov
            point_array[vertex_index] = np.transpose(ttt)[0]
        
    for vertex_index in range(point_number): 
        mesh.vertices[vertex_index] = point_array[vertex_index]
        
        
def disp_open3d(geometry,default=False):
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    geometry.append(coor)
    if(default):
        o3d.visualization.draw_geometries(geometry,mesh_show_back_face=True,mesh_show_wireframe=True)
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        #设定背景颜色
        opt = vis.get_render_option()#get_view_control()
        opt.mesh_show_back_face=True
        opt.mesh_show_wireframe=False
        opt.point_show_normal=True
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
    
    #mesh = o3d.io.read_stl_file(file_path)
    mesh0 = o3d.io.read_triangle_mesh(files_path[0]) 
    point_array0 = np.asarray(mesh0.vertices) 
    point_number0 = len(point_array0)
    center0 = [0.0,0.0,0.0]
    for vertex in point_array0: 
        center0 += vertex
    center0 -= point_array0[0]
    center0 /= point_number0
    for vertex_index in range(point_number0): 
        mesh0.vertices[vertex_index] = point_array0[vertex_index]-center0
    color = [0,0,0.1]
    mesh0.paint_uniform_color(color)
    
    mesh1 = o3d.io.read_triangle_mesh(files_path[1])    
    color1 = [0,0.1,0]
    mesh1.paint_uniform_color(color1)
    
    mesh2 = o3d.io.read_triangle_mesh(files_path[2])    
    color2 = [0.1,0,0]
    mesh2.paint_uniform_color(color2)
    
    pcd_ellipt = o3d.geometry.PointCloud()  
    length = 100
    points = []
    for i in range(length):
        x = -10.0+20.0*i/(length-1)
        y = -np.sqrt(b_squre - x*x*b_squre/a_squre)+center_ybias
        points += [[x,y,0.0]]
    pcd_ellipt.points = o3d.utility.Vector3dVector(points)
    colors = [[0.1,0.0,0.1] for i in range(length)]
    pcd_ellipt.colors = o3d.utility.Vector3dVector(colors)
          
    # 创建渲染窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    axis_actor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(axis_actor)
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	#设置点云渲染参数
    render_option.background_color = np.array([0.3, 0.3, 0.3])	#设置背景色（这里为黑色）
    render_option.light_on = True    
    
    vis.add_geometry(pcd_ellipt)	#添加点云   
    mesh0.compute_vertex_normals()
    mesh1.compute_vertex_normals()
    mesh2.compute_vertex_normals()
    vis.add_geometry(mesh0)
    vis.add_geometry(mesh1)
    vis.add_geometry(mesh2)
    #vis.run()
    
    theta = (np.pi * 0.5) * 0.5
    #test_rotate(pcd_ellipt, theta)  

    loc1 = [np.quaternion(), [0.0,0.0,0.0]]
    loc1[0].real = 1.0
    loc1[0].imag = np.array([0.0,0.0,0.0])
    loc1[1] = center0
    mesh0_1 = o3d.geometry.TriangleMesh(mesh0.vertices, mesh0.triangles)
    test_rotate2(mesh0_1, loc1)  
    mesh0_1.paint_uniform_color(color1)
    mesh0_1.compute_vertex_normals()
    vis.add_geometry(mesh0_1)
    
    loc1[0].real = np.cos(theta*0.5)
    loc1[0].imag = np.array([1.0,0.0,0.0])*np.sin(theta*0.5)
    mesh0_2 = o3d.geometry.TriangleMesh(mesh0.vertices, mesh0.triangles)
    test_rotate2(mesh0_2, loc1)  
    mesh0_2.paint_uniform_color(color2)
    mesh0_2.compute_vertex_normals()
    geo_handle_0 = vis.add_geometry(mesh0_2)
    #vis.run()
    
    for k in range(20):
        test_regist(mesh0_2, mesh0_1) 
        mesh0_2.compute_vertex_normals()
        #o3d.visualization.draw_geometries([mesh0, mesh1])      
    
        print("cycle ", k)
        vis.update_geometry(mesh0_2)
        vis.run()
        
    
           
    #vis.clear_geometries()
    print("end.")
    input("按下回车键继续...")
    vis.destroy_window()
    
    
    
    '''
//多线程测试
#include <thread>
#include <mutex>//与互斥量(mutex)相关的类

std::mutex m;
void threadfun1()
{
    std::cout << "threadfun1 - 1\r\n" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(15));
    //m.lock(); //保证线程同步的，防止不同的线程同时操作同一个共享数据
    //std::lock_guard<std::mutex> lockGuard(m);//基于作用域的，能够自解锁
    std::cout << "threadfun1 - 2" << std::endl;
    //m.unlock();
}

void threadfun2()
{
    std::cout << "threadfun2 - 1" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    //m.lock();
    std::cout << "threadfun2 - 2" << std::endl;
    //m.unlock();
}

class Test
{
public:
    Test()
    {
        std::cout << "Test()" << std::endl;
    }
    ~Test()
    {
        std::cout << "~Test()" << std::endl;
    }
    template <typename T> T func(T nParameter)
    {
        return (T)(nParameter * 2);
    }

    int a = 0;
};
int main()
{
    auto aaa = "test";
    std::cout << aaa << std::endl;
    int* bbb = nullptr;

    int ccc[] = { 1,2,3,4 };
    for (auto c : ccc)
    {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    std::vector<float> ddd = { 1.0f,2.1f };//(4,1.0f);
    for (auto& d : ddd)
    {
        d = 5.0f;
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::array<float, 4> eee = { 1.1f,2.2f, 3.3f, 4.4f };
    auto eee_s = sizeof(eee);
    auto eee_s2 = eee.size();
    std::cout << eee_s << std::endl;
    std::cout << eee_s2 << std::endl;

    //多线程测试
    std::thread t1(threadfun1);
    std::thread t2(threadfun2);

    std::cout << "******" << "th1 id: " << t1.get_id() << "******" << std::endl;
    std::cout << "******" << "th2 id: " << t2.get_id() << "******" << std::endl;
    t1.swap(t2);
    std::cout << "******" << "th1 id: " << t1.get_id() << "******" << std::endl;
    std::cout << "******" << "th2 id: " << t2.get_id() << "******" << std::endl;

    t1.join();//让主线程等待直到该子线程执行结束
    std::cout << "join t1" << std::endl;
    t2.detach();//将子分离出主线程，这样子线程可以独立地执行(即使主线程已销毁)
    std::cout << "detach t2" << std::endl << std::endl;

    

/*
————————————————
Eigen预定义了一些类型:
Matrix--二维,Vector--一维
MatrixNt = Matrix<type, N, N> 比如 MatrxXi = Matrix<int, Dynamic, Dynamic>
VectorNt = Matrix<type, N, 1> 比如 Vector2f = Matrix<float, 2, 1>
RowVectorNt = Matrix<type, 1, N> 比如 RowVector3d = Matrix<double, 1, 3>
-----------------
ArrayNt,一维array
ArrayNNt,二维array

N可以是2, 3, 4或X(Dynamic),
t可以是i(int)、f(float)、d(double)、cf(complex)、cd(complex)等。
Matrix的运算遵守矩阵运算规则，
Array则提供点对点的数组运算，比如对应系数相乘(点乘)，向量加数量(每个元素加)等
当需要线性代数类操作时，请使用Matrix；但需要元素级操作时，需要使用Array。

----
Eigen::Tensor:可创建高维向量(>2),点对点操作(类Array)
————————————————
*/

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
//#include <utility>
//#include <array>

{
    {
        Eigen::Tensor<float, 3> test_3d(2, 3, 4);//Eigen::Tensor<float, 3, Eigen::RowMajor>:默认列优先
        Eigen::TensorFixedSize<float, Eigen::Sizes<3, 4, 3>> test_fixed3d;//TensorFixedSize:固定size(定义时明确)，因此运算速度较快

        //Eigen::TensorMap---管理一块已定义的连续内存(类似指针,但可重新指定格式)
        int temp_array[128];
        Eigen::TensorMap<Eigen::Tensor<int, 4>> map_4d(temp_array, 2, 4, 2, 8);// 2 x 4 x 2 x 8 = 128
        Eigen::TensorMap<Eigen::Tensor<int, 2>> map_2d(temp_array, 16, 8);
        Eigen::TensorMap<Eigen::Tensor<float, 3>> map_3d(test_3d.data(), 4, 3, 2);

        test_3d.setValues({ { {1.2,2.2,3.2,4.2},{5.2,6.2,7.2,8.2},{1.5,3.5,5.5,7.5} },
            { {1.1,2.1,3.1,4.1},{5.1,6.1,7.1,8.1},{1.6,3.6,5.6,7.6} } });
        Eigen::Tensor<float, 2> test_3d_0(3, 4);
        Eigen::Tensor<float, 2> test_3d_00(3, 4);
        test_3d_0.setValues({ {1.2,2.2,3.2,4.2},{5.2,6.2,7.2,8.2},{1.5,3.5,5.5,7.5} });
        test_3d_00.setValues({ {1.1,2.1,3.1,4.1},{5.1,6.1,7.1,8.1},{1.6,3.6,5.6,7.6} });
        Eigen::DSizes<Eigen::DenseIndex, 3> dim(1,3,4);
        test_3d = test_3d_0.reshape(dim).concatenate(test_3d_00.reshape(dim), 0);//0:按行拼接
        std::cout << "test_3d:\n" << test_3d << std::endl;//按行(2行)输出,每行(3,4)按列(4列)输出
        std::cout << "Tensor 元素输出:" << test_3d(0, 2, 3)<< "\n" << test_3d.chip(1,0)  << std::endl;//chip(1,0)--第二(1)行(0)
        std::cout << "map_3d:\n" << map_3d << std::endl;

        Eigen::Tensor<float, 3> test_3d_2 = test_3d + 3.0f;//*3.0f;
        std::cout << "test_3d_2:\n" << test_3d_2 << std::endl;
        test_3d_2 = test_3d * test_3d;
        std::cout << "test_3d_2:\n" << test_3d_2 << std::endl;

        {/*
            Eigen::TensorFixedSize<float, Eigen::Sizes<5, 6, 7>> input_3d;
            Eigen::TensorFixedSize<float, Eigen::Sizes<3, 3, 3>> kernal_3d;
            Eigen::TensorFixedSize<float, Eigen::Sizes<3, 4, 5>> output_3d;*/
            Eigen::Tensor<float, 3> input_3d(4,5,6) ;
            Eigen::Tensor<float, 3> kernal_3d(3, 3, 3);
            Eigen::Tensor<float, 3> output_3d(2, 3, 4);
            
            input_3d.setRandom();
            kernal_3d.setValues({ {{0.25,0.5,0.25},{0.5,1.0,0.5},{0.25,0.5,0.25}},
                                {{0.5,1.0,0.5},{1.0,2,0,1.0},{0.5,1.0,0.5}},
                                {{0.25,0.5,0.25},{0.5,1.0,0.5},{0.25,0.5,0.25}} });
            
            /*Eigen::Tensor<float, 3>::Dimensions dim2_0(0, 1, 2);
            Eigen::Tensor<float, 3>::Dimensions dim2_1(0, 0, 0);
            Eigen::Tensor<float, 3>::Dimensions dim2_2(3, 3, 3);
            Eigen::DSizes<Eigen::DenseIndex, 3> dim2_0(0, 1, 2);
            Eigen::DSizes<Eigen::DenseIndex, 3> dim2_1(0, 0, 0);
            Eigen::DSizes<Eigen::DenseIndex, 3> dim2_2(3, 3, 3);*/
            Eigen::array<Eigen::DenseIndex, 3> dim2_0 = { 0, 1, 2 };
            Eigen::array<Eigen::DenseIndex, 3> dim2_1 = { 0, 0, 0 };
            Eigen::array<Eigen::DenseIndex, 3> dim2_2 = { 3, 3, 3 }; 
            output_3d = input_3d.convolve(kernal_3d, dim2_0);

            Eigen::Tensor<float, 3> input_tmp = input_3d.slice(dim2_1, dim2_2);
            std::cout << "input:\n" << input_tmp.chip(0, 0) << std::endl;
            std::cout << "input:\n" << input_tmp.chip(1, 0) << std::endl;
            std::cout << "input:\n" << input_tmp.chip(2, 0) << std::endl;
            std::cout << "convolve:\n" << output_3d(dim2_1) << std::endl;

            //Eigen::Tensor<float, 3> output_tmp = input_tmp * kernal_3d;
            //Eigen::Tensor<float, 0> convolve_manu = output_tmp.sum();//sum()--crush sometimes
            //Eigen::Tensor<float, 0> convolve_manu = (input_tmp * kernal_3d).sum();//sum()--crush in most times
            auto convolve_manu = (input_tmp * kernal_3d).sum();//sum()--work in most times
            std::cout << "convolve_manu:\n" << convolve_manu << std::endl;

            float convolve_manu2 = 0.0f;
            for (int k = 0;  k < dim2_2[0]; k++)
            {
                INT64 dim_k = dim2_1[0] + k;
                for (int j = 0; j < dim2_2[1]; j++)
                {
                    INT64 dim_j = dim2_1[1] + j;
                    for (int i = 0; i < dim2_2[0]; i++)
                    {
                        INT64 dim_i = dim2_1[2] + i;
                        convolve_manu2 += input_3d(dim_k, dim_j, dim_i) * kernal_3d(k,j,i);
                    }
                }
            }
            std::cout << "convolve_manu2:\n" << convolve_manu2 << std::endl;/**/

        }
    }

    Eigen::AngleAxisf test_angle(0.0f, Eigen::Vector3f::UnitZ());//(quant.test_angle.toRotationMatrix())//旋转量
    Eigen::Quaternionf quant(test_angle.toRotationMatrix());//(0.0, 0.0, 0.0, 1.0f);
    Eigen::Translation3f test_transl(0.0f,0.0f,0.0f);//平移量
    Eigen::Transform<float,3,1,0> test_transform = test_transl*test_angle;//test_angle*test_transl
    Eigen::Transform<float,3,1,0> test_transform2 = test_transl*quant;//quant*test_transl
    Eigen::Matrix4f test_trans_matrix = test_transform.matrix();

    Eigen::MatrixXf test_matrix(2, 3);//2行3列，列优先且不可更改
    Eigen::Matrix<float, 2, 3, Eigen::RowMajor> test_matrix_0;//指定行优先
    test_matrix.setZero();
    test_matrix << 1.0f, 2.0f, 3.0f,4.0f, 5.0f, 6.0f;//行优先赋值
    test_matrix_0 << 1.0f, 2.0f, 3.0f,4.0f, 5.0f, 6.0f;//行优先赋值(同上)
    test_matrix(1, 2) = 1.0f;
    std::cout << "test_matrix（尺寸和元素）:\n" << test_matrix.rows() << std::endl << test_matrix.cols() << std::endl << test_matrix << std::endl;
    std::cout << "test_matrix的转置矩阵:\n" << test_matrix.transpose() << std::endl;
    std::cout << "test_matrix的共轭矩阵:\n" << test_matrix.conjugate() << std::endl;
    std::cout << "test_matrix的伴随矩阵:\n" << test_matrix.adjoint() << std::endl;
    std::cout << "test_matrix的子矩阵:\n" << test_matrix.block(0,1,2,3) << std::endl;//test_matrix.block<2,3>(0,1)//可以作为左值(引用)
    std::cout << "test_matrix的一行:\n" << test_matrix.row(0) << std::endl;
    std::cout << "test_matrix的一列:\n" << test_matrix.col(0) << std::endl;
    std::cout << "test_matrix的最小值:\n" << test_matrix.minCoeff() << std::endl;
    std::cout << "test_matrix的最大值:\n" << test_matrix.maxCoeff() << std::endl;
    std::cout << "test_matrix的逐列遍历:\n" << test_matrix.colwise().maxCoeff() << std::endl;
    std::cout << "test_matrix的均值:\n" << test_matrix.mean() << std::endl;
    std::cout << "test_matrix的平方范数:\n" << test_matrix.squaredNorm() << std::endl;//所有元素平方和//可用来计算方差/标准差
    std::cout << "test_matrix的范数:\n" << test_matrix.norm() << std::endl;//所有元素平方和的根
    std::cout << "test_matrix的对角和:\n" << test_matrix.trace() << std::endl;
    std::cout << "test_matrix的内值的积:\n" << test_matrix.prod() << std::endl;//所有元素相乘

    {//can release the memory timely

        Eigen::MatrixXf test_matrix2(3, 3);
        test_matrix2 = test_matrix.transpose();
        std::cout << "test_matrix2:\n" << test_matrix2 << std::endl;

        Eigen::MatrixXf test_matrix3 = test_matrix * test_matrix2;;
        std::cout << "test_matrix3:\n" << test_matrix3 << std::endl;

        Eigen::MatrixXf test_matrix4 = test_matrix;
        std::cout << "test_matrix的数组式乘法:\n" << test_matrix.cwiseProduct(test_matrix4) << std::endl;
    }

    {
        Eigen::VectorXf test_vector(3);//列
        Eigen::RowVectorXf test_rvector(3);//行
        test_vector << 1.0f, 2.0f, 3.0f;
        test_rvector << 1.0f, 2.0f, 3.0f;
        std::cout << "test_vector:\n" << test_vector << std::endl;
        Eigen::VectorXf test_vector2 = test_matrix * test_vector;
        std::cout << "test_vector2:\n" << test_vector2 << std::endl;

        test_vector2 = test_vector;
        std::cout << "点乘 test:\n" << test_vector.dot(test_vector2) << std::endl;
        std::cout << "点乘 test2:\n" << test_vector.transpose() * test_vector2 << std::endl;

        Eigen::Vector3d test_vector3(1.0, 2.0, 3.0);
        Eigen::Vector3d test_vector4(4.0, 5.0, 6.0);
        std::cout << "叉乘 test:\n" << test_vector4.cross(test_vector3) << std::endl;
        //std::cout << "叉乘 test:\n" << test_vector.cross(test_vector2) << std::endl;//error:必须为Vector3*类型 

        Eigen::RowVectorXf test_vector5 = test_vector.transpose();
        std::cout << "逐行操作:\n" << test_matrix.rowwise() + test_vector5 << std::endl;

        //计算矩阵中哪列与目标向量距离最近。
        // find nearest neighbour
        Eigen::MatrixXf::Index index;
        (test_matrix.rowwise() - test_vector5).rowwise().squaredNorm().minCoeff(&index);
        std::cout << "Nearest neighbour is row " << index << ":" << std::endl;
        std::cout << test_matrix.row(index) << std::endl;
    }

    Eigen::ArrayXXf test_array(2, 3);//similar but different with matrix(XX--固定维数,最多两个; another difference--矩阵乘法,Array是点对点)
    test_array << 0.5f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f;
    test_array(1, 2) = 1.0f;
    std::cout << "test_array:\n" << test_array.rows() << std::endl << test_array.cols() << std::endl << test_array << std::endl;
    
    {//can release the memory timely
        Eigen::ArrayXXf test_array2 = test_array;
        std::cout << "test_array2:\n" << test_array2 << std::endl;

        Eigen::ArrayXXf test_array3 = test_array * test_array2;//diffenent with matrix
        std::cout << "test_array3:\n" << test_array3 << std::endl;

        std::cout << "test_array的转置矩阵:\n" << test_array.transpose() << std::endl;
        std::cout << "test_array的共轭矩阵:\n" << test_array.conjugate() << std::endl;
        //std::cout << "test_array的伴随矩阵:\n" << test_array.adjoint() << std::endl;//无
        std::cout << "test_array+标量:\n" << test_array+5 << std::endl;//matrix不允许
        //std::cout << "test_array与test_array3的逐点最小值:\n" << test_array.min(test_array3) << std::endl;//test_array.min编译时与Tensor有冲突
    }

    test_array = test_matrix.array();
    test_matrix = test_array.matrix();
    std::cout << "test_matrix:\n" << test_matrix.rows() << std::endl << test_matrix.cols() << std::endl << test_matrix << std::endl;
    std::cout << "test_array:\n" << test_array.rows() << std::endl << test_array.cols() << std::endl << test_array << std::endl;

    return 0;
}

    
    '''

