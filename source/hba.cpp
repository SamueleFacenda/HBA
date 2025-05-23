#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <mutex>
#include <assert.h>
#include <ros/ros.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include "ba.hpp"
#include "hba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

int pcd_name_fill_num = 0;

void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                pcl::PointCloud<PointType>& feat_pt,
                Eigen::Quaterniond q, Eigen::Vector3d t, int fnum,
                double voxel_size, int window_size, float eigen_ratio)
{
  float loc_xyz[3];
  for(PointType& p_c: feat_pt.points)
  {
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = q * pvec_orig + t;

    for(int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->vec_orig[fnum].push_back(pvec_orig);
      iter->second->vec_tran[fnum].push_back(pvec_tran);

      iter->second->sig_orig[fnum].push(pvec_orig);
      iter->second->sig_tran[fnum].push(pvec_tran);
    }
    else
    {
      OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->vec_tran[fnum].push_back(pvec_tran);
      ot->sig_orig[fnum].push(pvec_orig);
      ot->sig_tran[fnum].push(pvec_tran);

      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}

void parallel_comp(LAYER& layer, int thread_id, LAYER& next_layer)
{
  int& part_length = layer.part_length;
  int& layer_num = layer.layer_num;
  for(int i = thread_id*part_length; i < (thread_id+1)*part_length; i++)
  {
    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE); raw_pc.resize(WIN_SIZE);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    for(int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }
    
    if(layer_num != 1)
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared(); // deep copy why?

    size_t mem_cost = 0;
    // load point clouds
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      if(layer_num == 1)
        // first layer loads point clouds (at the first iteration)
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        {
          if(loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared(); // deep copy also here
        }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
      
      for(size_t j = 0; j < WIN_SIZE; j++)
      {
        if(layer.downsample_size > 0)
          downsample_voxel(*src_pc[j], layer.downsample_size);
        cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
      }
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      
      VOX_HESS voxhess(WIN_SIZE);
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);

      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
      
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;
        
        for(int j = 0; j < WIN_SIZE*(WIN_SIZE-1)/2; j++)
          layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];
        
        break;
      }
      residual_pre = residual_cur;
    }

    // transform all the clouds
    for(size_t j = 0; j < WIN_SIZE; j++)
    {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      mypcl::transform_pointcloud(*src_pc[j], *src_pc[j], t_tmp, q_tmp);
    }

    // merge them together
    pcl::PointCloud<PointType>::Ptr pc_keyframe = mypcl::append_clouds(src_pc);
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
}

void parallel_tail(LAYER& layer, int thread_id, LAYER& next_layer)
{
  int& part_length = layer.part_length;
  int& layer_num = layer.layer_num;
  int& left_gap_num = layer.left_gap_num;

  double load_t = 0, undis_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0,
    tran_t = 0, sol_t = 0, save_t = 0;
  
  if(layer.gap_num-(layer.thread_num-1)*part_length+1!=left_gap_num) printf("THIS IS WRONG!\n");

  for(uint i = thread_id*part_length; i < thread_id*part_length+left_gap_num; i++)
  {
    printf("parallel computing %d\n", i);
    double t0, t1;
    double t_begin = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE); raw_pc.resize(WIN_SIZE);
    
    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    for(int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }
    
    if(layer_num != 1)
    {
      t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
      load_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;
    }

    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      if(layer_num == 1)
      {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        {
          if(loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared();
        }
        load_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;
      }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

      for(size_t j = 0; j < WIN_SIZE; j++)
      {
        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        if(layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
        dsp_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;

        t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
        cut_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;
      }

      t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      recut_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;

      t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      VOX_HESS voxhess(WIN_SIZE);
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);
      tran_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;

      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
      sol_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;

      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
            
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        if(i < thread_id*part_length+left_gap_num)
          for(int j = 0; j < WIN_SIZE*(WIN_SIZE-1)/2; j++)
            layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }
    
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for(size_t j = 0; j < WIN_SIZE; j++)
    {
      t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
      save_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t1;
    }
    t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    downsample_voxel(*pc_keyframe, 0.05);
    dsp_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;

    t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    next_layer.pcds[i] = pc_keyframe;
    save_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0;
    
    total_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t_begin;
  }
  if(layer.tail > 0)
  {
    int i = thread_id*part_length+left_gap_num;

    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(layer.last_win_size); raw_pc.resize(layer.last_win_size);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(layer.last_win_size);
    for(int j = 0; j < layer.last_win_size; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }

    if(layer_num != 1)
    {
      for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
    }

    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      if(layer_num == 1)
        for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
        {
          if(loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared();          
        }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

      for(size_t j = 0; j < layer.last_win_size; j++)
      {
        if(layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, layer.last_win_size, layer.eigen_ratio);
      }
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      
      VOX_HESS voxhess(layer.last_win_size);
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);

      VOX_OPTIMIZER opt_lsv(layer.last_win_size);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
      
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        for(int j = 0; j < layer.last_win_size*(layer.last_win_size-1)/2; j++)
          layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];
        
        break;
      }
      residual_pre = residual_cur;
    }

    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for(size_t j = 0; j < layer.last_win_size; j++)
    {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
  printf("total time: %.2fs\n", total_t);
  printf("load pcd %.2fs %.2f%% | undistort pcd %.2fs %.2f%% | "
   "downsample %.2fs %.2f%% | cut voxel %.2fs %.2f%% | recut %.2fs %.2f%% | trans %.2fs %.2f%% | solve %.2fs %.2f%% | "
   "save pcd %.2fs %.2f%%\n",
    load_t, load_t/total_t*100, undis_t, undis_t/total_t*100,
    dsp_t, dsp_t/total_t*100, cut_t, cut_t/total_t*100, recut_t, recut_t/total_t*100, tran_t, tran_t/total_t*100,
    sol_t, sol_t/total_t*100, save_t, save_t/total_t*100);
}

void global_ba(LAYER& layer)
{
  int window_size = layer.pose_vec.size();
  vector<IMUST> x_buf(window_size);
  for(int i = 0; i < window_size; i++)
  {
    x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
    x_buf[i].p = layer.pose_vec[i].t;
  }

  vector<pcl::PointCloud<PointType>::Ptr> src_pc;
  src_pc.resize(window_size);
  for(int i = 0; i < window_size; i++)
    src_pc[i] = layer.pcds[i]; // this was a deep copy but since it's not used after it doesn't make sense

  double residual_cur = 0, residual_pre = 0;
  size_t mem_cost = 0, max_mem = 0;
  double dsp_t = 0, cut_t = 0, recut_t = 0, tran_t = 0, sol_t = 0, t0;
  for(int loop = 0; loop < layer.max_iter; loop++)
  {
    std::cout<<"---------------------"<<std::endl;
    std::cout<<"Iteration "<<loop<<std::endl;

    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

    for(int i = 0; i < window_size; i++)
    {
      t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      if(layer.downsample_size > 0) downsample_voxel(*src_pc[i], layer.downsample_size);
      dsp_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
      t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      cut_voxel(surf_map, *src_pc[i], Quaterniond(x_buf[i].R), x_buf[i].p, i,
                layer.voxel_size, window_size, layer.eigen_ratio*2);
      cut_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
    }
    t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut();
    recut_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
    
    t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    VOX_HESS voxhess(window_size);
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->tras_opt(voxhess);
    tran_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
    
    t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    VOX_OPTIMIZER opt_lsv(window_size);
    opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
    PLV(6) hess_vec;
    opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
    sol_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      delete iter->second;
    
    cout<<"Residual absolute: "<<abs(residual_pre-residual_cur)<<" | "
      <<"percentage: "<<abs(residual_pre-residual_cur)/abs(residual_cur)<<endl;
    
    if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
    {
      if(max_mem < mem_cost) max_mem = mem_cost;
      #ifdef FULL_HESS
      for(int i = 0; i < window_size*(window_size-1)/2; i++)
        layer.hessians[i] = hess_vec[i];
      #else
      for(int i = 0; i < window_size-1; i++)
      {
        Matrix6d hess = Hess_cur.block(6*i, 6*i+6, 6, 6);
        for(int row = 0; row < 6; row++)
          for(int col = 0; col < 6; col++)
            hessFile << hess(row, col) << ((row*col==25)?"":" ");
        if(i < window_size-2) hessFile << "\n";
      }
      #endif
      break;
    }
    residual_pre = residual_cur;
  }
  for(int i = 0; i < window_size; i++)
  {
    layer.pose_vec[i].q = Quaterniond(x_buf[i].R);
    layer.pose_vec[i].t = x_buf[i].p;
  }
  printf("Downsample: %f, Cut: %f, Recut: %f, Tras: %f, Sol: %f\n", dsp_t, cut_t, recut_t, tran_t, sol_t);
}

void distribute_thread(LAYER& layer, LAYER& next_layer)
{
  int& thread_num = layer.thread_num;
  double t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  for(int i = 0; i < thread_num-1; i++)
    layer.mthreads[i] = new thread(parallel_comp, ref(layer), i, ref(next_layer));
  layer.mthreads[thread_num-1] = new thread(parallel_tail, ref(layer), thread_num-1, ref(next_layer));
  // printf("Thread distribution time: %f\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0);

  t0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  for(int i = 0; i < thread_num; i++)
  {
    layer.mthreads[i]->join();
    delete layer.mthreads[i];
  }
  // printf("Thread join time: %f\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()-t0);
}

int main(int argc, char** argv)
{
//	ros::init(argc, argv, "hba");
//	ros::NodeHandle nh("~");

  int total_layer_num, thread_num;
  string data_path;

//  nh.getParam("total_layer_num", total_layer_num);
//  nh.getParam("pcd_name_fill_num", pcd_name_fill_num);
//  nh.getParam("data_path", data_path);
//  nh.getParam("thread_num", thread_num);

  total_layer_num = 3;
  pcd_name_fill_num = 5;
  data_path = "/home/samu/repos/HBA/kitti07/";
  thread_num = 16;

  HBA hba(total_layer_num, data_path, thread_num);
  for(int i = 0; i < total_layer_num-1; i++)
  {
    std::cout<<"---------------------"<<std::endl;
    distribute_thread(hba.curr_layer, hba.next_layer);
    hba.update_next_layer_state(i);
  }
  global_ba(hba.curr_layer);
  hba.pose_graph_optimization();
  printf("iteration complete\n");
}