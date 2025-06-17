#include <iostream>
#include <string>
#include <mutex>
#include <ros/ros.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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
  for(int i=0; i<feat_pt.points.size(); i++)
  {
    PointType p_c = feat_pt.points[i];
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = q * pvec_orig + t;

    for(int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if(loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->pointIndex[fnum].push_back(i);
      iter->second->vec_tran[fnum].push_back(pvec_tran);

      iter->second->sig_orig[fnum].push(pvec_orig);
      iter->second->sig_tran[fnum].push(pvec_tran);
    }
    else
    {
      OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
      ot->pointIndex[fnum].push_back(i);
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

void loadPCDs(LAYER& layer) {
  vector<std::thread> threads;
  atomic<int> index(0);

  auto worker = [&]() {
      while (true) {
        int i = index.fetch_add(1);
        if (i >= layer.pose_size) break;

        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, i, "pcd/");
        if (layer.downsample_size > 0)
          downsample_voxel(*pc, layer.downsample_size);
        layer.pcds[i] = pc; // TODO Make sure this is thread-safe
      }
  };

  for (int t = 0; t < layer.thread_num; ++t)
    threads.emplace_back(worker);

  for (auto& t : threads)
    t.join();
}

void compute_window(LAYER& layer, int part_id, LAYER& next_layer, int win_size = WIN_SIZE, bool print_info = false) {
  vector<pcl::PointCloud<PointType>::Ptr> src_pc(win_size);

  vector<set<int>> toRemove(win_size);

  double residual_cur = 0, residual_pre = 0;
  vector<IMUST> x_buf(win_size);
  for(int i = 0; i < win_size; i++)
  {
    x_buf[i].R = layer.pose_vec[part_id * GAP + i].q.toRotationMatrix();
    x_buf[i].p = layer.pose_vec[part_id * GAP + i].t;
  }

  for (int i = part_id * GAP; i < part_id * GAP + win_size; i++)
    src_pc[i - part_id * GAP] = (*layer.pcds[i]).makeShared(); // deep copy here (not necessary when is last layer btw)

  for(int loop = 0; loop < layer.max_iter; loop++) {
    if (print_info)
      cout << "---------------------" << endl
           << "Iteration " << loop << endl;

    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

    for(size_t i = 0; i < win_size; i++)
      cut_voxel(surf_map, *src_pc[i], Quaterniond(x_buf[i].R), x_buf[i].p, i, layer.voxel_size, win_size, layer.eigen_ratio);

    for(auto & iter : surf_map)
      iter.second->recut(toRemove);

    VOX_HESS voxhess(win_size);
    for(auto & iter : surf_map)
      iter.second->tras_opt(voxhess);

    VOX_OPTIMIZER opt_lsv(win_size);
    opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
    PLV(6) hess_vec;
    opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec);

    for(auto & iter : surf_map)
      delete iter.second;

    if (print_info)
      cout << "Residual absolute: " << abs(residual_pre-residual_cur) << " | "
           << "percentage: " << abs(residual_pre-residual_cur)/abs(residual_cur) << endl;

    if(loop > 0 && abs(residual_pre - residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1) {
      for(int i = 0; i < win_size * (win_size - 1) / 2; i++)
        layer.hessians[part_id*(WIN_SIZE-1)*WIN_SIZE/2 + i] = hess_vec[i];

      break;
    }
    residual_pre = residual_cur;
  }

  layer.computed_poses[part_id].resize(win_size);
  for(int i = 0; i < win_size; i++) {
    layer.computed_poses[part_id][i].q = Quaterniond(x_buf[i].R);
    layer.computed_poses[part_id][i].t = x_buf[i].p;
  }

  // transform all the clouds
  for(size_t i = 0; i < win_size; i++)
  {
    Eigen::Quaterniond q_tmp;
    Eigen::Vector3d t_tmp;
    assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[i].R),
              x_buf[0].R.inverse() * (x_buf[i].p - x_buf[0].p));

    // TODO make this directly in merging, to avoid copying clouds every time (there are 2/3 copies for every cloud)
    mypcl::transform_pointcloud(*src_pc[i], *src_pc[i], t_tmp, q_tmp);
  }

  // merge them together
  pcl::PointCloud<PointType>::Ptr pc_keyframe = mypcl::append_clouds(src_pc, toRemove);
  downsample_voxel(*pc_keyframe, next_layer.downsample_size);
  next_layer.pcds[part_id] = pc_keyframe;
}

void parallel_comp(LAYER& layer, int thread_id, LAYER& next_layer)
{
  int& part_length = layer.part_length;
  for(int i = thread_id*part_length; i < (thread_id+1)*part_length; i++)
    compute_window(layer, i, next_layer);
}

void parallel_tail(LAYER& layer, int thread_id, LAYER& next_layer)
{
  int& part_length = layer.part_length;
  int& left_gap_num = layer.left_gap_num;
  
  for(int i = thread_id*part_length; i < thread_id*part_length+left_gap_num; i++) {
    printf("parallel computing %d\n", i);
    compute_window(layer, i, next_layer);
  }
  if(layer.tail > 0) {
    int i = thread_id*part_length+left_gap_num;
    compute_window(layer, i, next_layer, layer.last_win_size);
  }
}

void global_ba(LAYER& layer, LAYER& next_layer)
{
  int win_size = layer.pose_vec.size();
  compute_window(layer, 0, next_layer, win_size, true);
}

void distribute_thread(LAYER& layer, LAYER& next_layer)
{
  int& thread_num = layer.thread_num;
  for(int i = 0; i < thread_num-1; i++)
    layer.mthreads[i] = new thread(parallel_comp, ref(layer), i, ref(next_layer));
  layer.mthreads[thread_num-1] = new thread(parallel_tail, ref(layer), thread_num-1, ref(next_layer));

  for(int i = 0; i < thread_num; i++)
  {
    layer.mthreads[i]->join();
    delete layer.mthreads[i];
  }
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

  pcd_name_fill_num = 5;
  data_path = "/home/samu/repos/HBA/kitti07/";
  thread_num = 20;

  HBA hba(data_path, thread_num);
  loadPCDs(hba.curr_layer);
  for(int i = 0; i < hba.total_layer_num-1; i++)
  {
    std::cout<<"---------------------"<<std::endl;
    distribute_thread(hba.curr_layer, hba.next_layer);
    hba.update_next_layer_state(i);
  }
  global_ba(hba.curr_layer, hba.next_layer);
  hba.pose_graph_optimization();

  // save hba.next_layer.pcds[0] to pcd file
  pcl::PointCloud<PointType>::Ptr final_pc = hba.next_layer.pcds[0];
  mypcl::savdPCD(hba.next_layer.data_path, pcd_name_fill_num, final_pc, 0);
  printf("iteration complete\n");
}