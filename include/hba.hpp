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

#define MAX_LAST_WIN_SIZE (4 * WIN_SIZE)
#define CONVERGENCE_THRESHOLD 0.05

class LAYER
{
public:
  int pose_size, layer_num, max_iter, part_length, last_thread_part_length, tail, thread_num,
    gap_num, last_win_size, left_gap_num;
  double downsample_size, voxel_size, eigen_ratio, reject_ratio;
  
  std::string data_path;
  vector<mypcl::pose> pose_vec;
  vector<vector<mypcl::pose>> computed_poses;
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

    if(layer_num < total_layer_num_)
    {
      int hessian_size = (thread_num-1)*(WIN_SIZE-1)*WIN_SIZE/2*part_length;
      hessian_size += (WIN_SIZE-1)*WIN_SIZE/2*left_gap_num;
      if(tail > 0) hessian_size += (last_win_size-1)*last_win_size/2;
      hessians.resize(hessian_size);
      printf("hessian_size: %d\n", hessian_size);

      computed_poses.resize((thread_num - 1) * part_length + last_thread_part_length);
      // for(int i = 0; i < thread_num * part_length + 1; i++)
      //   computed_poses[i].resize(WIN_SIZE);
    }
    else
    {
      int hessian_size = pose_size*(pose_size-1)/2;
      hessians.resize(hessian_size);
      printf("hessian_size: %d\n", hessian_size);

      computed_poses.resize(1);
      // computed_poses[0].resize(pose_size); // this is done by every thread
    }
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
      last_thread_part_length = left_gap_num;
    else
      last_thread_part_length = left_gap_num + 1;

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
  int thread_num, total_layer_num, iteration;
  LAYER curr_layer, next_layer;
  string data_path;
  vector<vector<VEC(6)>> covariances;
  vector<vector<vector<mypcl::pose>>> poses; // poses[layer_num][window_num]
  vector<mypcl::pose> initial_poses, final_poses;

  HBA(std::string data_path_, int thread_num_)
  {
    thread_num = thread_num_;
    data_path = data_path_;
    iteration = 0;

    // load poses, every iteration expects to have poses from previous iteration so we put them in final_poses
    final_poses = mypcl::read_pose(data_path + "pose.json");
    mypcl::write_pose(final_poses, data_path + "out" + std::to_string(iteration) + "_");

    // this is the resolution of the equation that finds the number of layers necessary to have a last window size <= MAX_LAST_WIN_SIZE
    // the minimum value for the last window size is MAX_LAST_WIN_SIZE / GAP
    // to use in the last layer at most the amount of resources of the first layer use MAX_LAST_WIN_SIZE = WIN_SIZE * sqrt(nthreads)
    // the plus one is because the last layer uses all the poses as a single window and doesn't divide them
    total_layer_num = ceil(log(final_poses.size() / MAX_LAST_WIN_SIZE) / log(GAP)) + 1;
  }

  void init_iteration() {
    iteration++;

    curr_layer.layer_num = 1;
    curr_layer.data_path = data_path;
    curr_layer.thread_num = thread_num;
    curr_layer.pose_vec = final_poses; // get the final poses from the previous iteration

    initial_poses = curr_layer.pose_vec;

    curr_layer.init_parameter();
    curr_layer.init_storage(total_layer_num);
    init_next_layer();
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
    poses.push_back(curr_layer.computed_poses);
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

  bool has_converged() {
    // compare initial and final poses
    double maxDistance = 0;
    double sumDistance = 0;
    for (size_t i = 0; i < initial_poses.size(); i++) {
      double distance = (initial_poses[i].t - final_poses[i].t).norm();
      if (distance > maxDistance)
        maxDistance = distance;
      sumDistance += distance;
    }
    cout << "Average distance between initial and final poses: " << sumDistance / initial_poses.size() << endl;
    cout << "Max distance between initial and final poses: " << maxDistance << endl;
    return maxDistance < CONVERGENCE_THRESHOLD;
  }

  void pose_graph_optimization()
  {
    poses.push_back(curr_layer.computed_poses); // TODO do it in the proper place
    covariances.push_back(curr_layer.hessians);

    int cnt;
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Vector vector6(6);
    vector6 << 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15;
    gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(vector6);

    // build initials
    for (int i = 0; i < initial_poses.size(); i++)
      initial.insert(i, gtsam::Pose3(gtsam::Rot3(initial_poses[i].q.toRotationMatrix()), gtsam::Point3(initial_poses[i].t)));

    graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(initial_poses[0].q.toRotationMatrix()),
                                                               gtsam::Point3(initial_poses[0].t)), priorModel));

    // to avoid disconnected graph, add factors between initial poses
    for (int i = 0; i < initial_poses.size() - 1; i++) {
      vector6 << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01; // TODO set proper values
      gtsam::NonlinearFactor::shared_ptr factor = getBetweenFactor(i, i+1, initial_poses[i], initial_poses[i+1], vector6);
      graph.push_back(factor);
    }

    for (int layer = 0; layer < total_layer_num; layer++) {
      cnt = 0;
      for (int window = 0; window < poses[layer].size(); window++) {
        // build the fully connected graph for the current window
        for (int i = 0; i < poses[layer][window].size() - 1; i++) {
          for (int j = i + 1; j < poses[layer][window].size(); j++, cnt++) {
            if (covariances[layer][cnt].norm() < 1e-20)
              continue;

            int absoluteFirst = (window * GAP + i) * pow(GAP, layer);
            int absoluteSecond = (window * GAP + j) * pow(GAP, layer);

            vector6 << fabs(1.0 / covariances[layer][cnt](0)), fabs(1.0 / covariances[layer][cnt](1)),
                    fabs(1.0 / covariances[layer][cnt](2)), fabs(1.0 / covariances[layer][cnt](3)),
                    fabs(1.0 / covariances[layer][cnt](4)), fabs(1.0 / covariances[layer][cnt](5));

            gtsam::NonlinearFactor::shared_ptr factor = getBetweenFactor(
                    absoluteFirst, absoluteSecond,
                    poses[layer][window][i], poses[layer][window][j], vector6);

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


    for(uint i = 0; i < results.size(); i++)
    {
      gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
      assign_qt(final_poses[i].q, final_poses[i].t, Eigen::Quaterniond(pose.rotation().matrix()), pose.translation());
    }
    mypcl::write_pose(final_poses, data_path);
    mypcl::write_pose(final_poses, data_path + "out" + std::to_string(iteration) + "_");
    cout << "PGO completed" << endl;
  }

private:
  static gtsam::NonlinearFactor::shared_ptr getBetweenFactor(int firstId, int secondId, const mypcl::pose& firstPose, const mypcl::pose& secondPose, const gtsam::Vector &covariance) {
    Eigen::Vector3d t_ab = firstPose.t;
    Eigen::Matrix3d R_ab = firstPose.q.toRotationMatrix();
    t_ab = R_ab.transpose() * (secondPose.t - t_ab);
    R_ab = R_ab.transpose() * secondPose.q.toRotationMatrix();
    gtsam::Rot3 R_sam(R_ab);
    gtsam::Point3 t_sam(t_ab);
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(covariance);
    gtsam::NonlinearFactor::shared_ptr factor(
            new gtsam::BetweenFactor<gtsam::Pose3>(firstId, secondId, gtsam::Pose3(R_sam, t_sam), odometryNoise));
    return factor;
  }
};

#endif
