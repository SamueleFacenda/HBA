#ifndef HBA_HPP
#define HBA_HPP

#include <iomanip>
#include <fstream>
#include <string>

#include <mutex>
#include <assert.h>
#include <ros/ros.h>
#include <Eigen/StdVector>

#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

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

#include "mypcl.hpp"
#include "tools.hpp"
#include "ba.hpp"

class LAYER
{
public:
  int pose_size, layer_num, max_iter, part_length, last_thread_part_length, tail, thread_num,
    gap_num, last_win_size, left_gap_num;
  double downsample_size, voxel_size, eigen_ratio, reject_ratio;
  
  std::string data_path;
  vector<mypcl::pose> pose_vec;
  std::vector<thread*> mthreads;

  std::vector<VEC(6)> hessians;
  std::vector<pcl::PointCloud<PointType>::Ptr> pcds;

  LAYER()
  {
    pose_size = 0;
    layer_num = 1;
    max_iter = 10;
    downsample_size = 0.1;
    voxel_size = 4.0;
    eigen_ratio = 0.1;
    reject_ratio = 0.05;
    pose_vec.clear(); mthreads.clear(); pcds.clear();
    hessians.clear();
  }

  void init_storage(int total_layer_num_)
  {
    mthreads.resize(thread_num);
    pcds.resize(pose_size);
    pose_vec.resize(pose_size);

    #ifdef FULL_HESS
    if(layer_num < total_layer_num_)
    {
      int hessian_size = (thread_num-1)*(WIN_SIZE-1)*WIN_SIZE/2*part_length;
      hessian_size += (WIN_SIZE-1)*WIN_SIZE/2*left_gap_num;
      if(tail > 0) hessian_size += (last_win_size-1)*last_win_size/2;
      hessians.resize(hessian_size);
      printf("hessian_size: %d\n", hessian_size);
    }
    else
    {
      int hessian_size = pose_size*(pose_size-1)/2;
      hessians.resize(hessian_size);
      printf("hessian_size: %d\n", hessian_size);
    }
    #endif
  }

  void init_parameter(int pose_size_ = 0)
  {
    if(layer_num == 1)
      pose_size = pose_vec.size();
    else
      pose_size = pose_size_;

    tail = (pose_size - WIN_SIZE) % GAP;
    gap_num = (pose_size - WIN_SIZE) / GAP;
    last_win_size = pose_size - GAP * (gap_num+1);
    part_length = ceil((gap_num+1)/double(thread_num));
    
    if(gap_num-(thread_num-1)*part_length < 0)
      part_length = floor((gap_num+1)/double(thread_num));

    // reduce thread num if part_length is too small
    while(part_length == 0 || (gap_num-(thread_num-1)*part_length+1)/double(part_length) > 2)
    {
      thread_num -= 1;
      part_length = ceil((gap_num+1)/double(thread_num));
      if(gap_num-(thread_num-1)*part_length < 0)
        part_length = floor((gap_num+1)/double(thread_num));
    }
    left_gap_num = gap_num-(thread_num-1)*part_length+1;
    
    if(tail == 0)
      last_thread_part_length = gap_num - (thread_num - 1) * part_length + 1;
    else
      last_thread_part_length = gap_num - (thread_num - 1) * part_length + 2;

    printf("init parameter:\n");
    printf("layer_num %d | thread_num %d | pose_size %d | max_iter %d | part_length %d | gap_num %d | last_win_size %d | "
      "left_gap_num %d | tail  %d | last_thread_part_length %d | "
      "downsample_size %f | voxel_size %f | eigen_ratio %f | reject_ratio %f\n",
           layer_num, thread_num, pose_size, max_iter, part_length, gap_num, last_win_size,
           left_gap_num, tail, last_thread_part_length,
           downsample_size, voxel_size, eigen_ratio, reject_ratio);
  }
};

class HBA
{
public:
  int thread_num, total_layer_num;
  LAYER curr_layer, next_layer;
  string data_path;
  vector<vector<VEC(6)>> covariances;
  vector<vector<mypcl::pose>> poses;

  HBA(int total_layer_num_, std::string data_path_, int thread_num_)
  {
    total_layer_num = total_layer_num_;
    thread_num = thread_num_;
    data_path = data_path_;

    curr_layer.layer_num = 1;
    curr_layer.data_path = data_path;
    curr_layer.thread_num = thread_num;
    curr_layer.pose_vec = mypcl::read_pose(data_path + "pose.json");
    curr_layer.init_parameter();
    curr_layer.init_storage(total_layer_num);
    init_next_layer();

    printf("HBA init done!\n");
  }

  void init_next_layer() {
    next_layer = LAYER(); // clear old data
    if (next_layer.layer_num == total_layer_num)
      next_layer.eigen_ratio *= 2;
    next_layer.layer_num = curr_layer.layer_num + 1;
    next_layer.thread_num = curr_layer.thread_num;
    int pose_size_ = (curr_layer.thread_num-1)*curr_layer.part_length;
    pose_size_ += curr_layer.tail == 0 ? curr_layer.left_gap_num : (curr_layer.left_gap_num+1);
    next_layer.init_parameter(pose_size_);
    next_layer.init_storage(total_layer_num);
    next_layer.data_path = curr_layer.data_path;
  }

  void update_next_layer_state(int cur_layer_num)
  {
    poses.push_back(curr_layer.pose_vec);
    covariances.push_back(curr_layer.hessians);
    
    for(int i = 0; i < curr_layer.thread_num - 1; i++) {
      for (int j = 0; j < curr_layer.part_length; j++) {
        int index = (i * curr_layer.part_length + j) * GAP;
        next_layer.pose_vec[i * curr_layer.part_length + j] = curr_layer.pose_vec[index];
      }
    }
    for (int j = 0; j < curr_layer.last_thread_part_length; j++) {
      int index = ((curr_layer.thread_num-1) * curr_layer.part_length + j) * GAP;
      next_layer.pose_vec[(curr_layer.thread_num-1) * curr_layer.part_length + j] = curr_layer.pose_vec[index];
    }

    curr_layer = next_layer;
    init_next_layer();
  }

  void pose_graph_optimization()
  {
    poses.push_back(curr_layer.pose_vec); // TODO do it in the proper place
    covariances.push_back(curr_layer.hessians);

    int cnt;
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Vector vector6(6);
    vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
    gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(vector6);

    // build initials
    for (uint i = 0; i < poses[0].size(); i++)
      initial.insert(i, gtsam::Pose3(gtsam::Rot3(poses[0][i].q.toRotationMatrix()), gtsam::Point3(poses[0][i].t)));

    graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(poses[0][0].q.toRotationMatrix()),
                                                               gtsam::Point3(poses[0][0].t)), priorModel));

    for (int layer = 0; layer < total_layer_num; layer++) {
      // last layer has only one window with all poses
      int win_size = layer == total_layer_num - 1 ? poses[layer].size() : WIN_SIZE;
      int step = layer == total_layer_num - 1 ? poses[layer].size() : GAP;
      cnt = 0;
      for (int window_start = 0; window_start < poses[layer].size(); window_start += step) {
        // if there is the tail use the appropriate size
        int current_win_size = min(win_size, (int)poses[layer].size() - window_start);
        for (int i = 0; i < current_win_size - 1; i++) {
          for (int j = i + 1; j < current_win_size; j++, cnt++) {
            int first = window_start + i;
            int second = window_start + j;

            if (covariances[layer][cnt].norm() < 1e-20)
              continue;

            Eigen::Vector3d t_ab = poses[layer][first].t;
            Eigen::Matrix3d R_ab = poses[layer][first].q.toRotationMatrix();
            t_ab = R_ab.transpose() * (poses[layer][second].t - t_ab);
            R_ab = R_ab.transpose() * poses[layer][second].q.toRotationMatrix();
            gtsam::Rot3 R_sam(R_ab);
            gtsam::Point3 t_sam(t_ab);

            vector6 << fabs(1.0 / covariances[layer][cnt](0)), fabs(1.0 / covariances[layer][cnt](1)),
                    fabs(1.0 / covariances[layer][cnt](2)), fabs(1.0 / covariances[layer][cnt](3)),
                    fabs(1.0 / covariances[layer][cnt](4)), fabs(1.0 / covariances[layer][cnt](5));
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(vector6);

            int absoluteFirst = first * pow(GAP, layer);
            int absoluteSecond = second * pow(GAP, layer);

            gtsam::NonlinearFactor::shared_ptr factor(
                    new gtsam::BetweenFactor<gtsam::Pose3>(absoluteFirst, absoluteSecond, gtsam::Pose3(R_sam, t_sam), odometryNoise)
                    );
            graph.push_back(factor);
          }
        }
      }
    }

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    gtsam::ISAM2 isam(parameters);
    isam.update(graph, initial);
    isam.update();

    gtsam::Values results = isam.calculateEstimate();

    cout << "vertex size " << results.size() << endl;

    for(uint i = 0; i < results.size(); i++)
    {
      gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
      assign_qt(poses[0][i].q, poses[0][i].t, Eigen::Quaterniond(pose.rotation().matrix()), pose.translation());
    }
    mypcl::write_pose(poses[0], data_path);
    printf("pgo complete\n");
  }
};

#endif
