#include "imu_processor/gyr_int.h"
#include <ros/ros.h>

using Sophus::SO3d;

GyrInt::GyrInt() : start_timestamp_(-1), last_imu_(nullptr) {}

/**
 *复位IMU位姿积分器，准备对新的一个同步帧进行位姿积分
 * @param start_timestamp: 起始时间戳，上一个同步帧中的点云时间戳
 * @param lastimu：上一个同步帧中的最后一帧IMU数据
 * @return 当前最新的IMU位姿
 */
void GyrInt::Reset(double start_timestamp,
                   const sensor_msgs::ImuConstPtr &lastimu) {
  start_timestamp_ = start_timestamp;
  last_imu_ = lastimu;

  v_rot_.clear();
  v_imu_.clear();
}

/**
 *获得当前最新的IMU位姿
 * @param 无
 * @return 当前最新的IMU位姿
 */
const Sophus::SO3d GyrInt::GetRot() const {
  if (v_rot_.empty()) {
    return SO3d();
  } else {
    return v_rot_.back();
  }
}

/**
 *对IMU数据进行积分
 * @param imu: 待积分的IMU当前帧数据
 * @return 无
 */
void GyrInt::Integrate(const sensor_msgs::ImuConstPtr &imu) {
  
  /*初始化本地旋转矩阵和imu数据*/
  if (v_rot_.empty()) {
    ROS_ASSERT(start_timestamp_ > 0);
    ROS_ASSERT(last_imu_ != nullptr);

    /// Identity rotation
	/*用单位矩阵初始化旋转矩阵*/
    v_rot_.push_back(SO3d());

    /// Interpolate imu in
    sensor_msgs::ImuPtr imu_inter(new sensor_msgs::Imu());
	
	/* dt1是上一个同步帧meas中，点云时间戳与最后一帧IMU数据的时间差
     * dt2是当前同步帧meas中的第一帧IMU与上一个同步帧中点云时间戳的时间差
     * dt1 + dt2 = 当前meas的第一个IMU与上个meas的最后一个IMU的时间差，也
	 * 是一个标准的IMU帧间隔：4.8ms
	 */
    double dt1 = start_timestamp_ - last_imu_->header.stamp.toSec();
    double dt2 = imu->header.stamp.toSec() - start_timestamp_;
    ROS_ASSERT_MSG(dt1 >= 0 && dt2 >= 0, "%f - %f - %f",
                   last_imu_->header.stamp.toSec(), start_timestamp_,
                   imu->header.stamp.toSec());
	
	/* 对于跨同步帧meas的相邻两帧IMU数据，取两帧IMU角速度和加速度的均值 */
    double w1 = dt2 / (dt1 + dt2 + 1e-9);
    double w2 = dt1 / (dt1 + dt2 + 1e-9);
    const auto &gyr1 = last_imu_->angular_velocity;
    const auto &acc1 = last_imu_->linear_acceleration;
    const auto &gyr2 = imu->angular_velocity;
    const auto &acc2 = imu->linear_acceleration;
    imu_inter->header.stamp.fromSec(start_timestamp_);
    imu_inter->angular_velocity.x = w1 * gyr1.x + w2 * gyr2.x;
    imu_inter->angular_velocity.y = w1 * gyr1.y + w2 * gyr2.y;
    imu_inter->angular_velocity.z = w1 * gyr1.z + w2 * gyr2.z;
    imu_inter->linear_acceleration.x = w1 * acc1.x + w2 * acc2.x;
    imu_inter->linear_acceleration.y = w1 * acc1.y + w2 * acc2.y;
    imu_inter->linear_acceleration.z = w1 * acc1.z + w2 * acc2.z;

    v_imu_.push_back(imu_inter);
  }

  /* 获得前一次的旋转矩阵 */
  const SO3d &rot_last = v_rot_.back();
  
  /* 获得前一帧IMU的时间戳和角速度 */
  const auto &imumsg_last = v_imu_.back();
  const double &time_last = imumsg_last->header.stamp.toSec();
  Eigen::Vector3d gyr_last(imumsg_last->angular_velocity.x,
                           imumsg_last->angular_velocity.y,
                           imumsg_last->angular_velocity.z);
						   
  /* 获得当前帧IMU的时间戳和角速度 */
  double time = imu->header.stamp.toSec();
  Eigen::Vector3d gyr(imu->angular_velocity.x, imu->angular_velocity.y,
                      imu->angular_velocity.z);

  /* 获得两帧IMU的时差dt */
  assert(time >= 0);
  double dt = time - time_last;
  /* 旋转角增量delta_angle等于最近两帧IMU角速度的均值乘以dt */
  auto delta_angle = dt * 0.5 * (gyr + gyr_last);
  /* 旋转角增量delta_angle对应的李群就是旋转矩阵的增量delta_R */
  auto delta_r = SO3d::exp(delta_angle);

  /* 在之前的旋转矩阵上叠加当前的旋转矩阵增量 */
  SO3d rot = rot_last * delta_r;

  /* 保存当前帧imu数据和旋转矩阵 */
  v_imu_.push_back(imu);
  v_rot_.push_back(rot);
}
