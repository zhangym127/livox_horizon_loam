#include "imu_processor/data_process.h"
#include <nav_msgs/Odometry.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <cmath>

#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>

using Sophus::SE3d;
using Sophus::SO3d;

pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudtmp(
    new pcl::PointCloud<pcl::PointXYZINormal>());

ImuProcess::ImuProcess()
    : b_first_frame_(true), last_lidar_(nullptr), last_imu_(nullptr) {
  Eigen::Quaterniond q(1, 0, 0, 0);
  Eigen::Vector3d t(0, 0, 0);
  T_i_l = Sophus::SE3d(q, t);
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
  ROS_WARN("Reset ImuProcess");

  b_first_frame_ = true;
  last_lidar_ = nullptr; /* 上一帧点云 */
  last_imu_ = nullptr; /* 上一帧IMU数据 */

  gyr_int_.Reset(-1, nullptr);

  cur_pcl_in_.reset(new PointCloudXYZI());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

/**
 * 对IMU数据进行积分。
 * @param v_imu: 完整的IMU数据列表
 * @return 无
 */
void ImuProcess::IntegrateGyr(
    const std::vector<sensor_msgs::Imu::ConstPtr> &v_imu) {
  /* 
   * 对IMU积分器进行复位，清空上一次的积分结果，用上一帧点云时间戳和
   * 上一帧IMU重新初始化积分器，last_lidar在时间上必须晚于last_imu_
   */
  /// Reset gyr integrator  
  gyr_int_.Reset(last_lidar_->header.stamp.toSec(), last_imu_);

  /* 对当前同步帧内的所有IMU数据进行积分，获得位姿终态 */
  /// And then integrate all the imu measurements
  for (const auto &imu : v_imu) {
    gyr_int_.Integrate(imu);
  }
  ROS_INFO("integrate rotation angle [x, y, z]: [%.2f, %.2f, %.2f]",
           gyr_int_.GetRot().angleX() * 180.0 / M_PI,
           gyr_int_.GetRot().angleY() * 180.0 / M_PI,
           gyr_int_.GetRot().angleZ() * 180.0 / M_PI);
}

/**
 * 对点云进行帧内校正。
 * @param pcl_in_out: 当前点云帧
 * @param dt_be: 当前点云帧与前一点云帧的时差，即当前同步帧的总时长
 * @param Tbe: 当前同步帧的IMU位姿终态，是一个SE3转换矩阵
 * @return 无
 */
void ImuProcess::UndistortPcl(const PointCloudXYZI::Ptr &pcl_in_out,
                              double dt_be, const Sophus::SE3d &Tbe) {

  /** 获得位姿的平移分量 */
  const Eigen::Vector3d &tbe = Tbe.translation();
  /** 将位姿的旋转矩阵分量转成李代数 */
  Eigen::Vector3d rso3_be = Tbe.so3().log();
  
  for (auto &pt : pcl_in_out->points) {
	
	/**根据点的时间戳准备好时间比例*/
	/* 获得当前点到同步帧起点的时差dt_bi，dt_bi中的b是begin的意思，i是当前点，
	 * 即从帧起点到该点的时差 */
    int ring = int(pt.intensity);//intensity的整数部分是行号，小数部分是时间戳
    float dt_bi = pt.intensity - ring;
	
	/* 时差为0的第一个点 */
    if (dt_bi == 0) laserCloudtmp->push_back(pt);
	
	/*获得当前点到同步帧起点时差占同步帧总时长的比值*/
    double ratio_bi = dt_bi / dt_be;
    /// Rotation from i-e
	/* 用于补偿的旋转矩阵是同步帧的终态，而ratio_bi是同步帧起点到当前点的时差
	 * 占比，因此需要用1减去比值，获得从当前点到同步帧终点的时差占比。
     * ratio_ie中的i是当前点，e是end的意思，即当前到终点的时差比值。 */
    double ratio_ie = 1 - ratio_bi;
	
	/**准备好旋转的补偿量*/
	/*李代数形式的三维向量乘以该点对应的时间比例。李群形式的旋转矩阵是非线性的，
	 *是无法用时间比例去计算对应的旋转量，转成李代数之后，就变成了线性的，可以
	 *乘以比例，再还原成李群，就获得了对应比例的旋转矩阵 */
    Eigen::Vector3d rso3_ie = ratio_ie * rso3_be;
	/*李代数转李群，即向量转成旋转矩阵*/
    SO3d Rie = SO3d::exp(rso3_ie);

    /// Transform to the 'end' frame, using only the rotation
    /// Note: Compensation direction is INVERSE of Frame's moving direction
    /// So if we want to compensate a point at timestamp-i to the frame-e
    /// P_compensate = R_ei * Pi + t_ei
	
	/**准备好平移的补偿量*/
	/*根据与结束帧的时间差，获得当前点对应的平移 */
    Eigen::Vector3d tie = ratio_ie * tbe;
	
	/**进行旋转和平移的补偿*/
    // Eigen::Vector3d tei = Eigen::Vector3d::Zero();
	/*构造该点的三维向量 */
    Eigen::Vector3d v_pt_i(pt.x, pt.y, pt.z);
	/*对该点进行补偿，先平移，后旋转 */
    Eigen::Vector3d v_pt_comp_e = Rie.inverse() * (v_pt_i - tie);

    /// Undistorted point
    pt.x = v_pt_comp_e.x();
    pt.y = v_pt_comp_e.y();
    pt.z = v_pt_comp_e.z();
  }
}

void ImuProcess::Process(const MeasureGroup &meas) {
  ROS_ASSERT(!meas.imu.empty());
  ROS_ASSERT(meas.lidar != nullptr);
  ROS_DEBUG("Process lidar at time: %.4f, %lu imu msgs from %.4f to %.4f",
            meas.lidar->header.stamp.toSec(), meas.imu.size(),
            meas.imu.front()->header.stamp.toSec(),
            meas.imu.back()->header.stamp.toSec());
  /* 
   * 每个同步帧meas中只有一帧点云，但是有多帧匹配的IMU数据，IMU数据是4.8ms一帧，
   * 同一个同步帧meas中，所有IMU数据帧的时间戳都小于点云的时间戳。
   */

  auto pcl_in_msg = meas.lidar;

  /* 记录第一帧点云和IMU，不做任何处理 */
  if (b_first_frame_) {
    /// The very first lidar frame

    /// Reset
    Reset();

    /// Record first lidar, and first useful imu
	/* 更新上一帧点云和上一帧IMU数据 */
	/* last_lidar_在时间上必须晚于last_imu_ */
    last_lidar_ = pcl_in_msg;
    last_imu_ = meas.imu.back();

    ROS_WARN("The very first lidar frame");

    /// Do nothing more, return
    b_first_frame_ = false;
    return;
  }

  /* 对IMU数据进行积分，获得最新的姿态 */
  /// Integrate all input imu message
  IntegrateGyr(meas.imu);

  /// Compensate lidar points with IMU rotation
  //// Initial pose from IMU (with only rotation)
  /* 获得IMU积分后的位姿终态，gyr_int_.GetRot()返回IMU位姿积分的最后状态 */
  SE3d T_l_c(gyr_int_.GetRot(), Eigen::Vector3d::Zero());
  
  /* 获得当前帧与上一帧点云的时差 */
  dt_l_c_ =
      pcl_in_msg->header.stamp.toSec() - last_lidar_->header.stamp.toSec();
	  
  //// Get input pcl 
  /* 获得输入点云 */
  pcl::fromROSMsg(*pcl_in_msg, *cur_pcl_in_);

  /// Undistort points
  /* 对IMU积分后的位姿终态前后分别乘以T_i_l，FIXME：这一步的含义还不清楚 */
  Sophus::SE3d T_l_be = T_i_l.inverse() * T_l_c * T_i_l;
  pcl::copyPointCloud(*cur_pcl_in_, *cur_pcl_un_);
  
  /* 对点云进行帧内校正 */
  UndistortPcl(cur_pcl_un_, dt_l_c_, T_l_be);

  
  { /* 发布第一个点给其他节点 */
    static ros::Publisher pub_UndistortPcl =
        nh.advertise<sensor_msgs::PointCloud2>("/livox_first_point", 100);
    sensor_msgs::PointCloud2 pcl_out_msg;
    pcl::toROSMsg(*laserCloudtmp, pcl_out_msg);
    pcl_out_msg.header = pcl_in_msg->header;
    pcl_out_msg.header.frame_id = "/camera_init";
    pub_UndistortPcl.publish(pcl_out_msg);
    laserCloudtmp->clear();
  }

  
  { /* 发布校正后的点云给其他节点 */
    static ros::Publisher pub_UndistortPcl =
        nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
    sensor_msgs::PointCloud2 pcl_out_msg;
    pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
    pcl_out_msg.header = pcl_in_msg->header;
    pcl_out_msg.header.frame_id = "/camera_init";
    pub_UndistortPcl.publish(pcl_out_msg);
  }

  { /* 发布校正前的点云给其他节点 */
    static ros::Publisher pub_UndistortPcl =
        nh.advertise<sensor_msgs::PointCloud2>("/livox_distort", 100);
    sensor_msgs::PointCloud2 pcl_out_msg;
    pcl::toROSMsg(*cur_pcl_in_, pcl_out_msg);
    pcl_out_msg.header = pcl_in_msg->header;
    pcl_out_msg.header.frame_id = "/camera_init";
    pub_UndistortPcl.publish(pcl_out_msg);
  }

  /* 更新上一帧点云和IMU数据指针 */
  /// Record last measurements
  last_lidar_ = pcl_in_msg;
  last_imu_ = meas.imu.back();
  
  /* 更新校正前后的点云指针 */
  cur_pcl_in_.reset(new PointCloudXYZI());
  cur_pcl_un_.reset(new PointCloudXYZI());
}
