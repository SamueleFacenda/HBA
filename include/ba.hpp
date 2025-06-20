#ifndef BA_HPP
#define BA_HPP

#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "tools.hpp"

#define WIN_SIZE 10
#define GAP 5
// #define ENABLE_RVIZ
// #define ENABLE_FILTER

const double ONE_THREE = (1.0 / 3.0);

int layer_limit = 2;
int MIN_PT = 15;
int thd_num = 16;

class VOX_HESS
{
public:
  vector<const vector<VOX_FACTOR>*> plvec_voxels;
  vector<PLV(3)> origin_points;
  int win_size;

  VOX_HESS(int _win_size = WIN_SIZE): win_size(_win_size){origin_points.resize(win_size);}

  ~VOX_HESS()
  {
    vector<const vector<VOX_FACTOR>*>().swap(plvec_voxels);
  }

  void get_center(const PLV(3)& vec_orig, PLV(3)& origin_points_)
  {
    size_t pt_size = vec_orig.size();
    for(size_t i = 0; i < pt_size; i++)
      origin_points_.emplace_back(vec_orig[i]);
  }

  void push_voxel(const vector<VOX_FACTOR>* sig_orig)
  {
    int process_size = 0;
    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        process_size++;

    #ifdef ENABLE_FILTER
    if(process_size < 1) return;

    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        get_center((*vec_orig)[i], origin_points[i]);
    #endif
    
    if(process_size < 2) return;
    
    plvec_voxels.push_back(sig_orig);
  }

  Eigen::Matrix<double, 6, 1> lam_f(Eigen::Vector3d *u, int m, int n)
  {
    Eigen::Matrix<double, 6, 1> jac;
    jac[0] = u[m][0] * u[n][0];
    jac[1] = u[m][0] * u[n][1] + u[m][1] * u[n][0];
    jac[2] = u[m][0] * u[n][2] + u[m][2] * u[n][0];
    jac[3] = u[m][1] * u[n][1];
    jac[4] = u[m][1] * u[n][2] + u[m][2] * u[n][1];
    jac[5] = u[m][2] * u[n][2];
    return jac;
  }

  void acc_evaluate2(const vector<IMUST>& xs, int head, int end,
                     Eigen::MatrixXd& Hess, Eigen::VectorXd& JacT, double& residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a = head; a < end; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];

      VOX_FACTOR sig;
      for(int i = 0; i < win_size; i++)
        if(sig_orig[i].N != 0)
        {
          sig_tran[i].transform(sig_orig[i], xs[i]);
          sig += sig_tran[i];
        }
      
      const Eigen::Vector3d& vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d& lmbd = saes.eigenvalues();
      const Eigen::Matrix3d& U = saes.eigenvectors();
      int NN = sig.N;
      
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d& uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i = 0; i < 3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i = 0; i < win_size; i++)
        if(sig_orig[i].N != 0)
        {
          Eigen::Matrix3d Pi = sig_orig[i].P;
          Eigen::Vector3d vi = sig_orig[i].v;
          Eigen::Matrix3d Ri = xs[i].R;
          double ni = sig_orig[i].N;

          Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
          Eigen::Vector3d RiTuk = Ri.transpose() * uk;
          Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();
          
          Eigen::Vector3d ti_v = xs[i].p - vBar;
          double ukTti_v = uk.dot(ti_v);

          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          JacT.block<6, 1>(6*i, 0) += jjt;

          const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          Hb.block<3, 3>(0, 0) +=
            2.0/NN*(combo1-RiTukhat*Pi)*RiTukhat - 2.0/NN/NN*viRiTuk[i]*viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
          Hb.block<3, 3>(0, 3) += HRt;
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

          Hess.block<6, 6>(6*i, 6*i) += Hb;
        }
      
      for(int i = 0; i < win_size-1; i++)
        if(sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          for(int j = i+1; j < win_size; j++)
            if(sig_orig[j].N != 0)
            {
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

              Hess.block<6, 6>(6*i, 6*j) += Hb;
            }
        }
      
      residual += lmbd[kk];
    }

    for(int i = 1; i < win_size; i++)
      for(int j = 0; j < i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  void evaluate_only_residual(const vector<IMUST>& xs, double& residual)
  {
    residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig;

      for(int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += lmbd[kk];
    }
  }

  std::vector<double> evaluate_residual(const vector<IMUST>& xs)
  {
    /* for outlier removal usage */
    std::vector<double> residuals;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig;

      for(int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residuals.push_back(lmbd[kk]);
    }

    return residuals;
  }

  void remove_residual(const vector<IMUST>& xs, double threshold, double reject_num)
  {
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int rej_cnt = 0;
    size_t i = 0;
    for(; i < plvec_voxels.size();)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[i];
      VOX_FACTOR sig;

      for(int j = 0; j < win_size; j++)
      {
        sig_tran[j].transform(sig_orig[j], xs[j]);
        sig += sig_tran[j];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      if(lmbd[kk] >= threshold)
      {
        plvec_voxels.erase(plvec_voxels.begin()+i);
        rej_cnt++;
        continue;
      }
      i++;
      if(rej_cnt == reject_num) break;
    }
  }
};

int BINGO_CNT = 0;
enum OCTO_STATE {UNKNOWN, MID_NODE, PLANE};
class OCTO_TREE_NODE
{
public:
  OCTO_STATE octo_state;
  int layer, win_size;
  vector<PLV(3)> vec_tran;
  vector<vector<int>> pointIndex;
  vector<VOX_FACTOR> sig_orig, sig_tran;

  OCTO_TREE_NODE* leaves[8];
  float voxel_center[3];
  float quater_length;
  float eigen_thr;

  Eigen::Vector3d center, direct, value_vector;
  double eigen_ratio;
  
  #ifdef ENABLE_RVIZ
  ros::NodeHandle nh;
  ros::Publisher pub_residual = nh.advertise<sensor_msgs::PointCloud2>("/residual", 1000);
  ros::Publisher pub_direct = nh.advertise<visualization_msgs::MarkerArray>("/direct", 1000);
  #endif

  OCTO_TREE_NODE(int _win_size = WIN_SIZE, float _eigen_thr = 1.0/10):
    win_size(_win_size), eigen_thr(_eigen_thr)
  {
    octo_state = UNKNOWN; layer = 0;
    vec_tran.resize(win_size);
    sig_orig.resize(win_size); sig_tran.resize(win_size);
    pointIndex.resize(win_size);
    for(int i = 0; i < 8; i++)
      leaves[i] = nullptr;
  }

  virtual ~OCTO_TREE_NODE()
  {
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  bool judge_eigen()
  {
    VOX_FACTOR covMat;
    for(int i = 0; i < win_size; i++)
      if(sig_tran[i].N > 0)
        covMat += sig_tran[i];
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    center = covMat.v / covMat.N;
    direct = saes.eigenvectors().col(0);

    eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // [0] is the smallest
    if(eigen_ratio > eigen_thr) return 0;

    double eva0 = saes.eigenvalues()[0];
    double sqr_eva0 = sqrt(eva0);
    Eigen::Vector3d center_turb = center + 5 * sqr_eva0 * direct;
    vector<VOX_FACTOR> covMats(8);
    for(int i = 0; i < win_size; i++)
    {
      for(Eigen::Vector3d ap: vec_tran[i])
      {
        int xyz[3] = {0, 0, 0};
        for(int k = 0; k < 3; k++)
          if(ap(k) > center_turb[k])
            xyz[k] = 1;

        Eigen::Vector3d pvec(ap(0), ap(1), ap(2));
        
        int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
        covMats[leafnum].push(pvec);
      }
    }

    double ratios[2] = {1.0/(3.0*3.0), 2.0*2.0};
    int num_all = 0, num_qua = 0;
    for(int i = 0; i < 8; i++)
      if(covMats[i].N > 10)
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov());
        double child_eva0 = (saes.eigenvalues()[0]);
        if(child_eva0 > ratios[0]*eva0 && child_eva0 < ratios[1]*eva0)
          num_qua++;
        num_all++;
      }

    double prop = 1.0 * num_qua / num_all;

    if(prop < 0.5) return 0;
    return 1;
  }

  void cut_func(int ci)
  {
    PLV(3)& pvec_tran = vec_tran[ci];

    uint a_size = pvec_tran.size();
    for(uint j = 0; j < a_size; j++)
    {
      int xyz[3] = {0, 0, 0};
      for(uint k = 0; k < 3; k++)
        if(pvec_tran[j][k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OCTO_TREE_NODE(win_size, eigen_thr);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2.0;
        leaves[leafnum]->layer = layer + 1;
      }

      leaves[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);
      
      leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
      leaves[leafnum]->pointIndex[ci].push_back(pointIndex[ci][j]);
    }

    PLV(3)().swap(pvec_tran);
    vector<int>().swap(pointIndex[ci]);
  }

  void recut(vector<set<int>>& toRemove)
  {
    if(octo_state == UNKNOWN)
    {
      int point_size = 0;
      for(int i = 0; i < win_size; i++)
        point_size += sig_orig[i].N;
      
      if(point_size < MIN_PT)
      {
        octo_state = MID_NODE;
        for(int i = 0; i < win_size; i++)
          toRemove[i].insert(pointIndex[i].begin(), pointIndex[i].end());

        vector<vector<int>>().swap(pointIndex);
        vector<PLV(3)>().swap(vec_tran);
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        return;
      }

      if(judge_eigen())
      {
        octo_state = PLANE;
        #ifndef ENABLE_FILTER
        #ifndef ENABLE_RVIZ
        vector<PLV(3)>().swap(vec_tran);
        #endif
        #endif
        return;
      }
      else
      {
        if(layer == layer_limit)
        {
          octo_state = MID_NODE;
          for(int i = 0; i < win_size; i++)
            toRemove[i].insert(pointIndex[i].begin(), pointIndex[i].end());

          vector<vector<int>>().swap(pointIndex);
          vector<PLV(3)>().swap(vec_tran);
          vector<VOX_FACTOR>().swap(sig_orig);
          vector<VOX_FACTOR>().swap(sig_tran);
          return;
        }
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        for(int i = 0; i < win_size; i++)
          cut_func(i);
      }
    }
    
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut(toRemove);
  }

  void tras_opt(VOX_HESS& vox_opt)
  {
    if(octo_state == PLANE)
      vox_opt.push_voxel(&sig_orig);
    else
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
  }

  void tras_display(int layer = 0)
  {
    float ref = 255.0*rand()/(RAND_MAX + 1.0f);
    pcl::PointXYZINormal ap;
    ap.intensity = ref;

    if(octo_state == PLANE)
    {
      // std::vector<unsigned int> colors;
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
      pcl::PointCloud<pcl::PointXYZINormal> color_cloud;

      for(int i = 0; i < win_size; i++)
      {
        for(size_t j = 0; j < vec_tran[i].size(); j++)
        {
          Eigen::Vector3d& pvec = vec_tran[i][j];
          ap.x = pvec.x();
          ap.y = pvec.y();
          ap.z = pvec.z();
          // ap.b = colors[0];
          // ap.g = colors[1];
          // ap.r = colors[2];
          ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
          ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
          ap.normal_z = sqrt(value_vector[0]);
          // ap.curvature = total;
          color_cloud.push_back(ap);
        }
      }

      #ifdef ENABLE_RVIZ
      sensor_msgs::PointCloud2 dbg_msg;
      pcl::toROSMsg(color_cloud, dbg_msg);
      dbg_msg.header.frame_id = "camera_init";
      pub_residual.publish(dbg_msg);

      visualization_msgs::Marker marker;
      visualization_msgs::MarkerArray marker_array;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = ros::Time::now();
      marker.ns = "basic_shapes";
      marker.id = BINGO_CNT; BINGO_CNT++;
      marker.action = visualization_msgs::Marker::ADD;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.color.a = 1;
      marker.color.r = layer==0?1:0;
      marker.color.g = layer==1?1:0;
      marker.color.b = layer==2?1:0;
      marker.scale.x = 0.01;
      marker.scale.y = 0.05;
      marker.scale.z = 0.05;
      marker.lifetime = ros::Duration();
      geometry_msgs::Point apoint;
      apoint.x = center(0); apoint.y = center(1); apoint.z = center(2);
      marker.points.push_back(apoint);
      apoint.x += 0.2*direct(0); apoint.y += 0.2*direct(1); apoint.z += 0.2*direct(2);
      marker.points.push_back(apoint);
      marker_array.markers.push_back(marker);
      pub_direct.publish(marker_array);
      #endif
    }
    else
    {
      if(layer == layer_limit) return;
      layer++;
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(layer);
    }
  }
};

class OCTO_TREE_ROOT: public OCTO_TREE_NODE
{
public:
  OCTO_TREE_ROOT(int _winsize, float _eigen_thr): OCTO_TREE_NODE(_winsize, _eigen_thr){}
};

class VOX_OPTIMIZER
{
public:
  int win_size, jac_leng, imu_leng;
  VOX_OPTIMIZER(int _win_size = WIN_SIZE): win_size(_win_size)
  {
    jac_leng = DVEL * win_size;
    imu_leng = DIM * win_size;
  }

  double divide_thread(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab,
                       Eigen::MatrixXd& Hess,Eigen::VectorXd& JacT)
  {
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);

    for(int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    for(int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1),
                               ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for(int i = 0; i < tthd_num; i++)
    {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }
    return residual/g_size;
  }

  double only_residual(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab)
  {
    double residual2 = 0;
    voxhess.evaluate_only_residual(x_stats, residual2);
    return residual2 / voxhess.plvec_voxels.size();
  }

  void remove_outlier(vector<IMUST>& x_stats, VOX_HESS& voxhess, double ratio)
  {
    std::vector<double> residuals = voxhess.evaluate_residual(x_stats);
    size_t nvx = voxhess.plvec_voxels.size();
    if (nvx == 0 || residuals.empty())
      return;
    std::sort(residuals.begin(), residuals.end()); // sort in ascending order

    int raw_index = static_cast<int>(std::floor((1.0 - ratio) * nvx) - 1);
    size_t threshold_index = std::clamp(raw_index, 0, static_cast<int>(nvx - 1));
    double threshold = residuals[threshold_index];

    int reject_num = std::floor(ratio * voxhess.plvec_voxels.size());
    voxhess.remove_residual(x_stats, threshold, reject_num);
  }

  void damping_iter(vector<IMUST>& x_stats, VOX_HESS& voxhess, double& residual, PLV(6)& hess_vec) {
    double u = 0.01, v = 2;
    Eigen::MatrixXd Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp;
    vector<IMUST> x_ab(win_size);

    x_ab[0] = x_stats[0];
    for (int i = 1; i < win_size; i++) {
      x_ab[i].p = x_stats[i - 1].R.transpose() * (x_stats[i].p - x_stats[i - 1].p);
      x_ab[i].R = x_stats[i - 1].R.transpose() * x_stats[i].R;
    }

    double hesstime = 0, solvtime = 0;
    size_t max_mem = 0;
    for (int i = 0; i < 10; i++) {
      if (is_calc_hess)
        residual1 = divide_thread(x_stats, voxhess, x_ab, Hess, JacT);


      vector<Eigen::Triplet<double>> triplets;
      for (int r = 0; r < jac_leng; ++r) {

        double d = Hess(r, r);
        triplets.emplace_back(r, r, Hess(r, r) + u * d);
        for (int c = r + 1; c < jac_leng; ++c) {
          double val = Hess(r, c);
          if (abs(val) > 1e-12) {
            triplets.emplace_back(r, c, val);
            triplets.emplace_back(c, r, val);
          }
        }
      }
      Eigen::SparseMatrix<double> HessuD_sparse(jac_leng, jac_leng);
      HessuD_sparse.setFromTriplets(triplets.begin(), triplets.end());
      HessuD_sparse.makeCompressed();

      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
      Solver_sparse.compute(HessuD_sparse);

      dxi = Solver_sparse.solve(-JacT);

      x_stats_temp = x_stats;
      for (int j = 0; j < win_size; j++) {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL * j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL * j + 3, 0);
      }

      double q1 = 0.5 * dxi.dot(u * Hess.diagonal().asDiagonal() * dxi - JacT);

      residual2 = only_residual(x_stats_temp, voxhess, x_ab);
      q1 /= voxhess.plvec_voxels.size();

      residual = residual2;
      q = residual1 - residual2;

      if (q > 0) {
        x_stats = x_stats_temp;
        q = q / q1;
        v = 2;
        q = 1 - pow(2 * q - 1, 3);
        u *= (q < ONE_THREE ? ONE_THREE : q);
        is_calc_hess = true;
      } else {
        u *= v;
        v *= 2;
        is_calc_hess = false;
      }

      if ((fabs(residual1 - residual2) / residual1) < 0.05 || i == 9) {
          for (int j = 0; j < win_size - 1; j++)
              for (int k = j + 1; k < win_size; k++)
                  hess_vec.push_back(Hess.block<DVEL, DVEL>(DVEL * j, DVEL * k).diagonal().segment<DVEL>(0));

          break;
      }
    }
  }
};

#endif