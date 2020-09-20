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
  last_lidar_ = nullptr; /* ��һ֡���� */
  last_imu_ = nullptr; /* ��һ֡IMU���� */

  gyr_int_.Reset(-1, nullptr);

  cur_pcl_in_.reset(new PointCloudXYZI());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

/**
 * ��IMU���ݽ��л��֡�
 * @param v_imu: ������IMU�����б�
 * @return ��
 */
void ImuProcess::IntegrateGyr(
    const std::vector<sensor_msgs::Imu::ConstPtr> &v_imu) {
  /* 
   * ��IMU���������и�λ�������һ�εĻ��ֽ��������һ֡����ʱ�����
   * ��һ֡IMU���³�ʼ����������last_lidar��ʱ���ϱ�������last_imu_
   */
  /// Reset gyr integrator  
  gyr_int_.Reset(last_lidar_->header.stamp.toSec(), last_imu_);

  /* �Ե�ǰͬ��֡�ڵ�����IMU���ݽ��л��֣����λ����̬ */
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
 * �Ե��ƽ���֡��У����
 * @param pcl_in_out: ��ǰ����֡
 * @param dt_be: ��ǰ����֡��ǰһ����֡��ʱ�����ǰͬ��֡����ʱ��
 * @param Tbe: ��ǰͬ��֡��IMUλ����̬����һ��SE3ת������
 * @return ��
 */
void ImuProcess::UndistortPcl(const PointCloudXYZI::Ptr &pcl_in_out,
                              double dt_be, const Sophus::SE3d &Tbe) {

  /** ���λ�˵�ƽ�Ʒ��� */
  const Eigen::Vector3d &tbe = Tbe.translation();
  /** ��λ�˵���ת�������ת������� */
  Eigen::Vector3d rso3_be = Tbe.so3().log();
  
  for (auto &pt : pcl_in_out->points) {
	
	/**���ݵ��ʱ���׼����ʱ�����*/
	/* ��õ�ǰ�㵽ͬ��֡����ʱ��dt_bi��dt_bi�е�b��begin����˼��i�ǵ�ǰ�㣬
	 * ����֡��㵽�õ��ʱ�� */
    int ring = int(pt.intensity);//intensity�������������кţ�С��������ʱ���
    float dt_bi = pt.intensity - ring;
	
	/* ʱ��Ϊ0�ĵ�һ���� */
    if (dt_bi == 0) laserCloudtmp->push_back(pt);
	
	/*��õ�ǰ�㵽ͬ��֡���ʱ��ռͬ��֡��ʱ���ı�ֵ*/
    double ratio_bi = dt_bi / dt_be;
    /// Rotation from i-e
	/* ���ڲ�������ת������ͬ��֡����̬����ratio_bi��ͬ��֡��㵽��ǰ���ʱ��
	 * ռ�ȣ������Ҫ��1��ȥ��ֵ����ôӵ�ǰ�㵽ͬ��֡�յ��ʱ��ռ�ȡ�
     * ratio_ie�е�i�ǵ�ǰ�㣬e��end����˼������ǰ���յ��ʱ���ֵ�� */
    double ratio_ie = 1 - ratio_bi;
	
	/**׼������ת�Ĳ�����*/
	/*�������ʽ����ά�������Ըõ��Ӧ��ʱ���������Ⱥ��ʽ����ת�����Ƿ����Եģ�
	 *���޷���ʱ�����ȥ�����Ӧ����ת����ת�������֮�󣬾ͱ�������Եģ�����
	 *���Ա������ٻ�ԭ����Ⱥ���ͻ���˶�Ӧ��������ת���� */
    Eigen::Vector3d rso3_ie = ratio_ie * rso3_be;
	/*�����ת��Ⱥ��������ת����ת����*/
    SO3d Rie = SO3d::exp(rso3_ie);

    /// Transform to the 'end' frame, using only the rotation
    /// Note: Compensation direction is INVERSE of Frame's moving direction
    /// So if we want to compensate a point at timestamp-i to the frame-e
    /// P_compensate = R_ei * Pi + t_ei
	
	/**׼����ƽ�ƵĲ�����*/
	/*���������֡��ʱ����õ�ǰ���Ӧ��ƽ�� */
    Eigen::Vector3d tie = ratio_ie * tbe;
	
	/**������ת��ƽ�ƵĲ���*/
    // Eigen::Vector3d tei = Eigen::Vector3d::Zero();
	/*����õ����ά���� */
    Eigen::Vector3d v_pt_i(pt.x, pt.y, pt.z);
	/*�Ըõ���в�������ƽ�ƣ�����ת */
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
   * ÿ��ͬ��֡meas��ֻ��һ֡���ƣ������ж�֡ƥ���IMU���ݣ�IMU������4.8msһ֡��
   * ͬһ��ͬ��֡meas�У�����IMU����֡��ʱ�����С�ڵ��Ƶ�ʱ�����
   */

  auto pcl_in_msg = meas.lidar;

  /* ��¼��һ֡���ƺ�IMU�������κδ��� */
  if (b_first_frame_) {
    /// The very first lidar frame

    /// Reset
    Reset();

    /// Record first lidar, and first useful imu
	/* ������һ֡���ƺ���һ֡IMU���� */
	/* last_lidar_��ʱ���ϱ�������last_imu_ */
    last_lidar_ = pcl_in_msg;
    last_imu_ = meas.imu.back();

    ROS_WARN("The very first lidar frame");

    /// Do nothing more, return
    b_first_frame_ = false;
    return;
  }

  /* ��IMU���ݽ��л��֣�������µ���̬ */
  /// Integrate all input imu message
  IntegrateGyr(meas.imu);

  /// Compensate lidar points with IMU rotation
  //// Initial pose from IMU (with only rotation)
  /* ���IMU���ֺ��λ����̬��gyr_int_.GetRot()����IMUλ�˻��ֵ����״̬ */
  SE3d T_l_c(gyr_int_.GetRot(), Eigen::Vector3d::Zero());
  
  /* ��õ�ǰ֡����һ֡���Ƶ�ʱ�� */
  dt_l_c_ =
      pcl_in_msg->header.stamp.toSec() - last_lidar_->header.stamp.toSec();
	  
  //// Get input pcl 
  /* ���������� */
  pcl::fromROSMsg(*pcl_in_msg, *cur_pcl_in_);

  /// Undistort points
  /* ��IMU���ֺ��λ����̬ǰ��ֱ����T_i_l��FIXME����һ���ĺ��廹����� */
  Sophus::SE3d T_l_be = T_i_l.inverse() * T_l_c * T_i_l;
  pcl::copyPointCloud(*cur_pcl_in_, *cur_pcl_un_);
  
  /* �Ե��ƽ���֡��У�� */
  UndistortPcl(cur_pcl_un_, dt_l_c_, T_l_be);

  
  { /* ������һ����������ڵ� */
    static ros::Publisher pub_UndistortPcl =
        nh.advertise<sensor_msgs::PointCloud2>("/livox_first_point", 100);
    sensor_msgs::PointCloud2 pcl_out_msg;
    pcl::toROSMsg(*laserCloudtmp, pcl_out_msg);
    pcl_out_msg.header = pcl_in_msg->header;
    pcl_out_msg.header.frame_id = "/camera_init";
    pub_UndistortPcl.publish(pcl_out_msg);
    laserCloudtmp->clear();
  }

  
  { /* ����У����ĵ��Ƹ������ڵ� */
    static ros::Publisher pub_UndistortPcl =
        nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
    sensor_msgs::PointCloud2 pcl_out_msg;
    pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
    pcl_out_msg.header = pcl_in_msg->header;
    pcl_out_msg.header.frame_id = "/camera_init";
    pub_UndistortPcl.publish(pcl_out_msg);
  }

  { /* ����У��ǰ�ĵ��Ƹ������ڵ� */
    static ros::Publisher pub_UndistortPcl =
        nh.advertise<sensor_msgs::PointCloud2>("/livox_distort", 100);
    sensor_msgs::PointCloud2 pcl_out_msg;
    pcl::toROSMsg(*cur_pcl_in_, pcl_out_msg);
    pcl_out_msg.header = pcl_in_msg->header;
    pcl_out_msg.header.frame_id = "/camera_init";
    pub_UndistortPcl.publish(pcl_out_msg);
  }

  /* ������һ֡���ƺ�IMU����ָ�� */
  /// Record last measurements
  last_lidar_ = pcl_in_msg;
  last_imu_ = meas.imu.back();
  
  /* ����У��ǰ��ĵ���ָ�� */
  cur_pcl_in_.reset(new PointCloudXYZI());
  cur_pcl_un_.reset(new PointCloudXYZI());
}
