#ifndef MYPCL_HPP
#define MYPCL_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vector_vec3d;
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > vector_quad;
// typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZ PointType;
// typedef pcl::PointXYZI PointType;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace mypcl
{
  struct pose
  {
    pose(Eigen::Quaterniond _q = Eigen::Quaterniond(1, 0, 0, 0),
         Eigen::Vector3d _t = Eigen::Vector3d(0, 0, 0)):q(_q), t(_t){}
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
  };

  void loadPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr& pc, int num,
               std::string prefix = "")
  {
    std::stringstream ss;
    if(pcd_fill_num > 0)
      ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
    else
      ss << num;
    pcl::io::loadPCDFile(filePath + prefix + ss.str() + ".pcd", *pc);
  }

  void savdPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr& pc, int num)
  {
    std::stringstream ss;
    ss << "out";
    if(pcd_fill_num > 0) // apply formatting if fill number is specified
      ss << std::setw(pcd_fill_num) << std::setfill('0');
    ss << num;

    cout << "Saving PCD to: " << filePath + ss.str() + ".pcd" << endl;
    pcl::io::savePCDFileBinary(filePath + ss.str() + ".pcd", *pc);
  }
  
  std::vector<pose> read_pose(std::string filename,
                              Eigen::Quaterniond qe = Eigen::Quaterniond(1, 0, 0, 0),
                              Eigen::Vector3d te = Eigen::Vector3d(0, 0, 0))
  {
    std::vector<pose> pose_vec;
    std::fstream file;
    file.open(filename);
    double tx, ty, tz, w, x, y, z;
    while(file >> tx >> ty >> tz >> w >> x >> y >> z)
    {
      Eigen::Quaterniond q(w, x, y, z);
      Eigen::Vector3d t(tx, ty, tz);
      pose_vec.push_back(pose(qe * q, qe * t + te));
    }
    file.close();
    return pose_vec;
  }

  void transform_pointcloud(pcl::PointCloud<PointType> const& pc_in,
                            pcl::PointCloud<PointType>& pt_out,
                            Eigen::Vector3d t,
                            Eigen::Quaterniond q)
  {
    size_t size = pc_in.points.size();
    pt_out.points.resize(size);
    for(size_t i = 0; i < size; i++)
    {
      Eigen::Vector3d pt_cur(pc_in.points[i].x, pc_in.points[i].y, pc_in.points[i].z);
      Eigen::Vector3d pt_to;
      // if(pt_cur.norm()<0.3) continue;
      pt_to = q * pt_cur + t;
      pt_out.points[i].x = pt_to.x();
      pt_out.points[i].y = pt_to.y();
      pt_out.points[i].z = pt_to.z();
      // pt_out.points[i].r = pc_in.points[i].r;
      // pt_out.points[i].g = pc_in.points[i].g;
      // pt_out.points[i].b = pc_in.points[i].b;
    }
  }

  // merge multiple clouds in the reference frame of the first cloud
  pcl::PointCloud<PointType>::Ptr append_clouds_with_poses(std::vector<pcl::PointCloud<PointType>::Ptr> pc_vec, const vector<IMUST>& poses, const vector<set<int>>& pointsToSkip) {
    pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
    int size = 0, curr_size = 0;
    for(int i = 0; i < pc_vec.size(); i++)
      size += pc_vec[i]->points.size() - pointsToSkip[i].size();

    pc->points.resize(size);
    for(int pc_id = 0; pc_id < pc_vec.size(); pc_id++) {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Eigen::Quaterniond(poses[0].R.inverse() * poses[pc_id].R),
                poses[0].R.inverse() * (poses[pc_id].p - poses[0].p));

      auto toSkipIter = pointsToSkip[pc_id].begin();
      for(int i=0; i<pc_vec[pc_id]->points.size(); i++) {
        if (toSkipIter != pointsToSkip[pc_id].end() && i == *toSkipIter) {
          toSkipIter++;
          continue;
        }

        Eigen::Vector3d pt_cur(pc_vec[pc_id]->points[i].x, pc_vec[pc_id]->points[i].y, pc_vec[pc_id]->points[i].z);
        Eigen::Vector3d pt_to = q_tmp * pt_cur + t_tmp;
        pc->points[curr_size].x = pt_to.x();
        pc->points[curr_size].y = pt_to.y();
        pc->points[curr_size].z = pt_to.z();
        curr_size++;
      }
    }

    pc->width = size;
    pc->height = 1; // set height to 1 for unorganized point cloud
    pc->is_dense = true; // set is_dense to true if there are no NaN values
    return pc;
  }

  pcl::PointCloud<PointType>::Ptr append_clouds_with_poses(std::vector<pcl::PointCloud<PointType>::Ptr> pc_vec, const vector<pose>& poses, const vector<set<int>>& pointsToSkip) {
    vector<IMUST> x_buf;
    x_buf.reserve(poses.size());
    for(const auto& pose : poses)
    {
      IMUST tmp;
      tmp.R = pose.q.toRotationMatrix();
      tmp.p = pose.t;
      x_buf.push_back(tmp);
    }
    return append_clouds_with_poses(pc_vec, x_buf, pointsToSkip);
  }

  double compute_inlier_ratio(std::vector<double> residuals, double ratio)
  {
    std::set<double> dis_vec;
    for(size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
      dis_vec.insert(fabs(residuals[3 * i + 0]) +
                     fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));

    return *(std::next(dis_vec.begin(), (int)((ratio) * dis_vec.size())));
  }

  void write_pose(std::vector<pose>& pose_vec, std::string path)
  {
    std::ofstream file;
    file.open(path + "pose.json", std::ofstream::trunc);
    file.close();
    Eigen::Quaterniond q0(pose_vec[0].q.w(), pose_vec[0].q.x(), pose_vec[0].q.y(), pose_vec[0].q.z());
    Eigen::Vector3d t0(pose_vec[0].t(0), pose_vec[0].t(1), pose_vec[0].t(2));
    file.open(path + "pose.json", std::ofstream::app);

    for(size_t i = 0; i < pose_vec.size(); i++)
    {
      //  not right here
      //pose_vec[i].t << q0.inverse()*(pose_vec[i].t-t0);
      //pose_vec[i].q.w() = (q0.inverse()*pose_vec[i].q).w();
      //pose_vec[i].q.x() = (q0.inverse()*pose_vec[i].q).x();
      //pose_vec[i].q.y() = (q0.inverse()*pose_vec[i].q).y();
      //pose_vec[i].q.z() = (q0.inverse()*pose_vec[i].q).z();
      file << pose_vec[i].t(0) << " "
           << pose_vec[i].t(1) << " "
           << pose_vec[i].t(2) << " "
           << pose_vec[i].q.w() << " " << pose_vec[i].q.x() << " "
           << pose_vec[i].q.y() << " " << pose_vec[i].q.z();
      if(i < pose_vec.size()-1) file << "\n";
    }
    file.close();
  }

  void writeEVOPose(std::vector<double>& lidar_times, std::vector<pose>& pose_vec, std::string path)
  {
    std::ofstream file;
    file.open(path + "evo_pose.txt", std::ofstream::trunc);
    for(size_t i = 0; i < pose_vec.size(); i++)
    {
      file << std::setprecision(18) << lidar_times[i] << " " << std::setprecision(6)
           << pose_vec[i].t(0) << " " << pose_vec[i].t(1) << " " << pose_vec[i].t(2) << " "
           << pose_vec[i].q.x() << " " << pose_vec[i].q.y() << " "
           << pose_vec[i].q.z() << " " << pose_vec[i].q.w();
      if(i < pose_vec.size()-1) file << "\n";
    }
    file.close();
  }
}

#endif
