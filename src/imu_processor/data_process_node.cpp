#include <ros/ros.h>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <mutex>
#include <thread>

#include "imu_processor/data_process.h"

/// *************Config data
std::string topic_pcl = "/livox_pcl0";
std::string topic_imu = "/imu";
/// *************

/// To notify new data
std::mutex mtx_buffer;
std::condition_variable sig_buffer;
bool b_exit = false;
bool b_reset = false;

/// Buffers for measurements
double last_timestamp_lidar = -1;
std::deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
double last_timestamp_imu = -1;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

void SigHandle(int sig) {
  b_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}

/**
 * 将当前点云帧添加到缓冲
 * @param msg: 当前点云帧
 * @return 无
 */
void pointcloud_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  const double timestamp = msg->header.stamp.toSec();
  // ROS_DEBUG("get point cloud at time: %.6f", timestamp);

  mtx_buffer.lock();

  if (timestamp < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  last_timestamp_lidar = timestamp;

  lidar_buffer.push_back(msg);

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

/**
 * 将当前IMU数据帧添加到缓冲
 * @param msg_in: 当前IMU帧
 * @return 无
 */
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  double timestamp = msg->header.stamp.toSec();
  // ROS_DEBUG("get imu at time: %.6f", timestamp);

  mtx_buffer.lock();

  if (timestamp < last_timestamp_imu) {
    ROS_ERROR("imu loop back, clear buffer");
    imu_buffer.clear();
    b_reset = true;
  }
  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

/**
 * 对点云和IMU数据进行同步。
 * @param measgroup: 存放同步数据的缓冲
 * @return 无
 */
bool SyncMeasure(MeasureGroup &measgroup) {
	
  if (lidar_buffer.empty() || imu_buffer.empty()) {
    /// Note: this will happen
    return false;
  }

  /* 所有的IMU数据时间戳都比点云更大，点云数据已经过时，直接清空点云数据 */
  if (imu_buffer.front()->header.stamp.toSec() >
      lidar_buffer.back()->header.stamp.toSec()) {
    lidar_buffer.clear();
    ROS_ERROR("clear lidar buffer, only happen at the beginning");
    return false;
  }

  /* 所有的IMU数据时间戳都比点云更小，IMU数据有可能不完整，直接返回 */
  if (imu_buffer.back()->header.stamp.toSec() <
      lidar_buffer.front()->header.stamp.toSec()) {
    return false;
  }

  /* 以点云的时间戳为准，小于该时间戳的所有IMU数据都转移到同步缓冲 */
  
  /* 取最旧的一帧点云放入同步缓冲 */
  /// Add lidar data, and pop from buffer
  measgroup.lidar = lidar_buffer.front();
  lidar_buffer.pop_front();
  double lidar_time = measgroup.lidar->header.stamp.toSec();

  /* 将时间戳小于该点云的所有IMU数据转移到同步缓冲，
   * 这就确保同步缓冲中的所有IMU数据时间戳都小于点云 */
  /// Add imu data, and pop from buffer
  measgroup.imu.clear();
  int imu_cnt = 0;
  for (const auto &imu : imu_buffer) {
    double imu_time = imu->header.stamp.toSec();
    if (imu_time <= lidar_time) {
      measgroup.imu.push_back(imu);
      imu_cnt++;
    }
  }
  for (int i = 0; i < imu_cnt; ++i) {
    imu_buffer.pop_front();
  }
  // ROS_DEBUG("add %d imu msg", imu_cnt);

  return true;
}

/**
 * 对缓冲中的点云和IMU数据进行同步，匹配后进行处理。
 * @param measgroup: 存放同步数据的缓冲
 * @return 无
 */
void ProcessLoop(std::shared_ptr<ImuProcess> p_imu) {
  ROS_INFO("Start ProcessLoop");

  ros::Rate r(1000);
  while (ros::ok()) {
    MeasureGroup meas;
    std::unique_lock<std::mutex> lk(mtx_buffer);
    sig_buffer.wait(lk,
                    [&meas]() -> bool { return SyncMeasure(meas) || b_exit; });
    lk.unlock();

    if (b_exit) {
      ROS_INFO("b_exit=true, exit");
      break;
    }

    if (b_reset) {
      ROS_WARN("reset when rosbag play back");
      p_imu->Reset();
      b_reset = false;
      continue;
    }
    p_imu->Process(meas);

    r.sleep();
  }
}

/**
 * 传感器数据处理节点。
 * @param argc: 
 * @param argv: 
 * @return 无
 */
int main(int argc, char **argv) {
  ros::init(argc, argv, "data_process");
  ros::NodeHandle nh;
  signal(SIGINT, SigHandle);

  /* 订阅点云和IMU数据，并注册对应的回调函数 */
  ros::Subscriber sub_pcl = nh.subscribe(topic_pcl, 100, pointcloud_cbk);
  ros::Subscriber sub_imu = nh.subscribe(topic_imu, 1000, imu_cbk);

  std::shared_ptr<ImuProcess> p_imu(new ImuProcess());

  std::vector<double> vec;
  if( nh.getParam("/ExtIL", vec) ){
    Eigen::Quaternion<double> q_il;
    Eigen::Vector3d t_il;
    q_il.w() = vec[0];
    q_il.x() = vec[1];
    q_il.y() = vec[2];
    q_il.z() = vec[3];
    t_il << vec[4], vec[5], vec[6];
    p_imu->set_T_i_l(q_il, t_il);
    ROS_INFO("Extrinsic Parameter RESET ... ");
  }

  /// for debug
  p_imu->nh = nh;

  std::thread th_proc(ProcessLoop, p_imu);

  // ros::spin();
  ros::Rate r(1000);
  while (ros::ok()) {
    if (b_exit) break;

    ros::spinOnce();
    r.sleep();
  }

  ROS_INFO("Wait for process loop exit");
  if (th_proc.joinable()) th_proc.join();

  return 0;
}
