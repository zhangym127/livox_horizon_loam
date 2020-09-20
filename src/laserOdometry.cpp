// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               Livox@gmail.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "lidarFactor.hpp"
#include "loam_horizon/common.h"
#include "loam_horizon/tic_toc.h"

#define DISTORTION 0 // Low-speed scene, without distortion correction

int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast(
    new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast(
    new pcl::KdTreeFLANN<PointType>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(
    new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(
    new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

/**
 * 对点云进行帧内校正，向第一个点看齐
 * @param pi 输入点
 * @param po 输出点
 * @return 无
 */
// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po) {
  // interpolation ratio
  double s;
  
  /* intensity的整数部分是行号，小数部分是时间戳 */
  if (DISTORTION)
    s = (pi->intensity - int(pi->intensity))*10;
  else
    s = 1.0;
  // s = 1;
  
  /* 在表示旋转的四元数Identity(0,0,0,0)和q_last_curr之间进行插值，比例系数是s，s必须在0到1之间。
   * q_last_curr应该是当前点云帧中最后一个点的旋转四元数，而表示第一个点的旋转四元数是Identity。
   * 通过插值，找到帧内每个点对应的旋转向量q_point_last。 */
  Eigen::Quaterniond q_point_last =
      Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
	  
  /* t_last_curr应该是当前点云帧中最后一个点的平移向量，乘以当前点的时间比例系数 */
  Eigen::Vector3d t_point_last = s * t_last_curr;
  
  /* 对当前点进行旋转和平移，获得帧内校正后的点 */
  Eigen::Vector3d point(pi->x, pi->y, pi->z);
  Eigen::Vector3d un_point = q_point_last * point + t_point_last;

  po->x = un_point.x();
  po->y = un_point.y();
  po->z = un_point.z();
  po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame

void TransformToEnd(PointType const *const pi, PointType *const po) {
  // undistort point first
  PointType un_point_tmp;
  TransformToStart(pi, &un_point_tmp);

  Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
  Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

  po->x = point_end.x();
  po->y = point_end.y();
  po->z = point_end.z();

  // Remove distortion time info
  po->intensity = pi->intensity;
}

void laserCloudSharpHandler(
    const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2) {
  mBuf.lock();
  cornerSharpBuf.push(cornerPointsSharp2);
  mBuf.unlock();
}

void laserCloudLessSharpHandler(
    const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2) {
  mBuf.lock();
  cornerLessSharpBuf.push(cornerPointsLessSharp2);
  mBuf.unlock();
}

void laserCloudFlatHandler(
    const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2) {
  mBuf.lock();
  surfFlatBuf.push(surfPointsFlat2);
  mBuf.unlock();
}

void laserCloudLessFlatHandler(
    const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2) {
  mBuf.lock();
  surfLessFlatBuf.push(surfPointsLessFlat2);
  mBuf.unlock();
}

// receive all point cloud
void laserCloudFullResHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2) {
  mBuf.lock();
  fullPointsBuf.push(laserCloudFullRes2);
  mBuf.unlock();
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "laserOdometry");
  ros::NodeHandle nh;

  nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

  printf("Mapping %d Hz \n", 10 / skipFrameNum);

  /* 订阅Sharp、LessSharp、Flat、LessFlat四种点云，并调用回调函数将点云添加到对应的缓冲中 */
  ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>(
      "/laser_cloud_sharp", 100, laserCloudSharpHandler);

  ros::Subscriber subCornerPointsLessSharp =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100,
                                             laserCloudLessSharpHandler);

  ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>(
      "/laser_cloud_flat", 100, laserCloudFlatHandler);

  ros::Subscriber subSurfPointsLessFlat =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100,
                                             laserCloudLessFlatHandler);

  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>(
      "/velodyne_cloud_2", 100, laserCloudFullResHandler);

  /* 创建publisher对象，准备进行发布 */
  ros::Publisher pubLaserCloudCornerLast =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

  ros::Publisher pubLaserCloudSurfLast =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

  ros::Publisher pubLaserCloudFullRes =
      nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

  ros::Publisher pubLaserOdometry =
      nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

  ros::Publisher pubLaserPath =
      nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

  nav_msgs::Path laserPath;

  int frameCount = 0;
  ros::Rate rate(100);

  while (ros::ok()) {
    ros::spinOnce();


    if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
        !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
        !fullPointsBuf.empty()) {
	  /* 获得Sharp等四种点云的起始时间戳 */
      timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
      timeCornerPointsLessSharp =
          cornerLessSharpBuf.front()->header.stamp.toSec();
      timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
      timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
      timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
	
	  /* 如果四种点云的时间戳与整个点云的起始时间戳不一致，则break */
      if (timeCornerPointsSharp != timeLaserCloudFullRes ||
          timeCornerPointsLessSharp != timeLaserCloudFullRes ||
          timeSurfPointsFlat != timeLaserCloudFullRes ||
          timeSurfPointsLessFlat != timeLaserCloudFullRes) {
        printf("unsync messeage!");
        ROS_BREAK();
      }

	  /* 从订阅的ROS消息中取出对应的四组点云，以及完整点云 */
      mBuf.lock();
      cornerPointsSharp->clear();
      pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
      cornerSharpBuf.pop();

      cornerPointsLessSharp->clear();
      pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
      cornerLessSharpBuf.pop();

      surfPointsFlat->clear();
      pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
      surfFlatBuf.pop();

      surfPointsLessFlat->clear();
      pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
      surfLessFlatBuf.pop();

      laserCloudFullRes->clear();
      pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
      fullPointsBuf.pop();
      mBuf.unlock();

      TicToc t_whole;
      // initializing
      if (!systemInited) {
        systemInited = true;
        std::cout << "Initialization finished \n";
      } else {
        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        int surfPointsFlatNum = surfPointsFlat->points.size();

        TicToc t_opt;
        for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter) {
          corner_correspondence = 0;
          plane_correspondence = 0;

		  /* 下面使用Ceres库对前后两帧点云进行非线性优化，获得里程计 */
          // ceres::LossFunction *loss_function = NULL;
          ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
          ceres::LocalParameterization *q_parameterization =
              new ceres::EigenQuaternionParameterization();
          ceres::Problem::Options problem_options;

          ceres::Problem problem(problem_options);
          problem.AddParameterBlock(para_q, 4, q_parameterization);
          problem.AddParameterBlock(para_t, 3);

          PointType pointSel;
          std::vector<int> pointSearchInd;
          std::vector<float> pointSearchSqDis;

          TicToc t_data;
		  /* 遍历曲度为Sharp的点云，与前一帧的LessSharp点云匹配 */
          for (int i = 0; i < cornerPointsSharpNum; ++i) {
			/* 对当前点进行帧内校正，由于是慢速场景，实际上没有做 */
            TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
			/* 在上一帧LessSharp点云中寻找与当前点最近的5个点，pointSearchInd是找到的5个最近点，
			 * pointSearchSqDis是五个最近点的距离 */
            kdtreeCornerLast->nearestKSearch(pointSel, 5, pointSearchInd,
                                             pointSearchSqDis);

			/* 如果5个最近点的距离都小于给定阈值，将这5个点加入nearCorners，并计算这五个点的质心center，即均值 */
            if (pointSearchSqDis[4] < DISTANCE_SQ_THRESHOLD) {
              std::vector<Eigen::Vector3d> nearCorners;
              Eigen::Vector3d center(0, 0, 0);
              for (int j = 0; j < 5; j++) {
                Eigen::Vector3d tmp(
                    laserCloudCornerLast->points[pointSearchInd[j]].x,
                    laserCloudCornerLast->points[pointSearchInd[j]].y,
                    laserCloudCornerLast->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);
              }
              center = center / 5.0;

			  /* 求五个点的协方差矩阵 */
              Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
              for (int j = 0; j < 5; j++) {
                Eigen::Matrix<double, 3, 1> tmpZeroMean =
                    nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
              }

			  /* 计算自伴随矩阵covMat的特征值和特征向量 */
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

			  /* 确认这五个点是成一条线分布，也就某个物体边缘的一部分 */
              // if is indeed line feature
			  
			  /* Eigen计算出来的特征值以增序排列，col(2)就是取三个特征值中最大的一个 */
              // note Eigen library sort eigenvalues in increasing order
              Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
              // Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
			  
			  /* 确认最大的特征值远远大于（3倍以上）次大的特征值，也就是五个点基本上成一条线分布 */
              if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                Eigen::Vector3d point_on_line = center;
				/* 沿最大特征向量方向，以质心为中心，在质心两侧构造a、b两个点，a、b两个点距离质心0.1米 */
                Eigen::Vector3d last_point_a, last_point_b;
                last_point_a = 0.1 * unit_direction + point_on_line;
                last_point_b = -0.1 * unit_direction + point_on_line;

                Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                           cornerPointsSharp->points[i].y,
                                           cornerPointsSharp->points[i].z);

                double s;
                if (DISTORTION)
                  s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity))*10;
                else
                  s = 1.0;

				/* 将当前点p和a、b两个点加入优化序列，最优的结果是a、p、b三点一线，p在正中间 */
                // printf(" Edge s------ %f  \n", s);
                ceres::CostFunction *cost_function = LidarEdgeFactor::Create(
                    curr_point, last_point_a, last_point_b, s);
                problem.AddResidualBlock(cost_function, loss_function, para_q,
                                         para_t);
                corner_correspondence++;
              }
            }
          }

		  /* 遍历曲度为Flat的点云，与前一帧的LessFlat点云匹配 */
          // find correspondence for plane features
          for (int i = 0; i < surfPointsFlatNum; ++i) {
			/* 对该点进行帧内校正，校正到帧内起点的状态 */
            TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
			
			/* 在上一帧LessFlat点云中寻找与当前点最近的5个点，pointSearchInd是找到的5个最近点，
			 * pointSearchSqDis是五个最近点的距离 */
            kdtreeSurfLast->nearestKSearch(pointSel, 5, pointSearchInd,
                                           pointSearchSqDis);

			/* 定义矩阵A[5,3]和B[5,1]，其中B是全-1列向量 */
            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 =
                -1 * Eigen::Matrix<double, 5, 1>::Ones();
			
			/* 如果5个最近点的距离都小于给定阈值，用五个点的坐标初始化矩阵A */
            if (pointSearchSqDis[4] < DISTANCE_SQ_THRESHOLD) {
              for (int j = 0; j < 5; j++) {
                matA0(j, 0) = laserCloudSurfLast->points[pointSearchInd[j]].x;
                matA0(j, 1) = laserCloudSurfLast->points[pointSearchInd[j]].y;
                matA0(j, 2) = laserCloudSurfLast->points[pointSearchInd[j]].z;
                // printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j,
                // 2));
              }
			  /* 求解三元一次方程组Ax=b，进行平面的法向量估计
			   * 三元一次方程Ax+By+Cz+D=0对应于空间平面，向量n=(A,B,C)是其法向量，
			   * 这里已知五个点在同一平面，且设定D为1，则一定可以找到唯一的一组法
			   * 向量n=(A,B,C)与该平面对应，调用A.colPivHouseholderQr().solve(b)
			   * 求解三元一次方程组，获得法向量n=(A,B,C)。*/
              // find the norm of plane
              Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
			  
			  /* 下面用法向量检查这五个点是否构成一个严格的平面
               * 在平面上的点p(x,y,z)一定满足方程Ax+By+Cz+1=0,如果法向量(A,B,C)的范数
               * 是n，则方程的两边同时除以n，等式仍然成立：(A/n)x+(B/n)y+(C/n)z+1/n=0。
               * 不满足这个等式的点不在该平面上。*/
              double negative_OA_dot_norm = 1 / norm.norm(); //求1/n
              norm.normalize(); //求(A/n,B/n,C/n)

              // Here n(pa, pb, pc) is unit norm of plane
              bool planeValid = true;
              for (int j = 0; j < 5; j++) {
				/* 计算等式(A/n)x+(B/n)y+(C/n)z+1/n的值，并检查是否接近于0 */
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) *
                             laserCloudSurfLast->points[pointSearchInd[j]].x +
                         norm(1) *
                             laserCloudSurfLast->points[pointSearchInd[j]].y +
                         norm(2) *
                             laserCloudSurfLast->points[pointSearchInd[j]].z +
                         negative_OA_dot_norm) > 0.02) {
				  /* 该点距离平面太远 */
                  planeValid = false;
                  break;
                }
              }

			  /* 仅当五个点都在一个平面上时，抽取第1、3、5号点与当前点进行空间变换的非线性优化 */
              if (planeValid) {
                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                           surfPointsFlat->points[i].y,
                                           surfPointsFlat->points[i].z);
                Eigen::Vector3d last_point_a(
                    laserCloudSurfLast->points[pointSearchInd[0]].x,
                    laserCloudSurfLast->points[pointSearchInd[0]].y,
                    laserCloudSurfLast->points[pointSearchInd[0]].z);
                Eigen::Vector3d last_point_b(
                    laserCloudSurfLast->points[pointSearchInd[2]].x,
                    laserCloudSurfLast->points[pointSearchInd[2]].y,
                    laserCloudSurfLast->points[pointSearchInd[2]].z);
                Eigen::Vector3d last_point_c(
                    laserCloudSurfLast->points[pointSearchInd[4]].x,
                    laserCloudSurfLast->points[pointSearchInd[4]].y,
                    laserCloudSurfLast->points[pointSearchInd[4]].z);

                double s;
                if (DISTORTION)
                  s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity))*10;
                else
                  s = 1.0;
                // printf(" Plane s------ %f  \n", s);
                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(
                    curr_point, last_point_a, last_point_b, last_point_c, s);
                problem.AddResidualBlock(cost_function, loss_function, para_q,
                                         para_t);
                plane_correspondence++;
              }
            }
            //}
          }
          // printf("coner_correspondance %d, plane_correspondence %d \n",
          // corner_correspondence, plane_correspondence);
          printf("data association time %f ms \n", t_data.toc());

          if ((corner_correspondence + plane_correspondence) < 10) {
            printf(
                "less correspondence! "
                "*************************************************\n");
          }

		  /* 开始优化 */
          TicToc t_solver;
          ceres::Solver::Options options;
          options.linear_solver_type = ceres::DENSE_QR;
          options.max_num_iterations = 20;
          options.minimizer_progress_to_stdout = false;
          ceres::Solver::Summary summary;
          ceres::Solve(options, &problem, &summary);
          printf("solver time %f ms \n", t_solver.toc());
        }
        printf("optimization twice time %f \n", t_opt.toc());

		/* 优化后的结果保存在q_last_curr和t_last_curr中 */
        t_w_curr = t_w_curr + q_w_curr * t_last_curr;
        q_w_curr = q_w_curr * q_last_curr;
        std::cout<<"t_w_curr: "<<t_w_curr.transpose()<<std::endl;
      }

      TicToc t_pub;

	  /* 发布里程计，即当前最新的位姿，包括旋转和平移 */
      // publish odometry
      nav_msgs::Odometry laserOdometry;
      laserOdometry.header.frame_id = "/camera_init";
      laserOdometry.child_frame_id = "/laser_odom";
      laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometry.pose.pose.orientation.x = q_w_curr.x();
      laserOdometry.pose.pose.orientation.y = q_w_curr.y();
      laserOdometry.pose.pose.orientation.z = q_w_curr.z();
      laserOdometry.pose.pose.orientation.w = q_w_curr.w();
      laserOdometry.pose.pose.position.x = t_w_curr.x();
      laserOdometry.pose.pose.position.y = t_w_curr.y();
      laserOdometry.pose.pose.position.z = t_w_curr.z();
      pubLaserOdometry.publish(laserOdometry);

	  /* 发布位姿的轨迹 */
      geometry_msgs::PoseStamped laserPose;
      laserPose.header = laserOdometry.header;
      laserPose.pose = laserOdometry.pose.pose;
      laserPath.header.stamp = laserOdometry.header.stamp;
      laserPath.poses.push_back(laserPose);
      laserPath.header.frame_id = "/camera_init";
      pubLaserPath.publish(laserPath);

	  /* 将Sharp、Flat和完整点云校正为向最后一个点看齐 */
      // transform corner features and plane features to the scan end point
      if (DISTORTION) {
        int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
        for (int i = 0; i < cornerPointsLessSharpNum; i++) {
          TransformToEnd(&cornerPointsLessSharp->points[i],
                         &cornerPointsLessSharp->points[i]);
        }

        int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
        for (int i = 0; i < surfPointsLessFlatNum; i++) {
          TransformToEnd(&surfPointsLessFlat->points[i],
                         &surfPointsLessFlat->points[i]);
        }

        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          TransformToEnd(&laserCloudFullRes->points[i],
                         &laserCloudFullRes->points[i]);
        }
      }

      pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
      cornerPointsLessSharp = laserCloudCornerLast;
      laserCloudCornerLast = laserCloudTemp;

      laserCloudTemp = surfPointsLessFlat;
      surfPointsLessFlat = laserCloudSurfLast;
      laserCloudSurfLast = laserCloudTemp;

      laserCloudCornerLastNum = laserCloudCornerLast->points.size();
      laserCloudSurfLastNum = laserCloudSurfLast->points.size();

      // std::cout << "the size of corner last is " << laserCloudCornerLastNum
      // << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

      kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
      kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

      if (frameCount % skipFrameNum == 0) {
        frameCount = 0;

		/* 发布Sharp类型的点云，尽管Sharp点云没有任何变化，再次发布出去 */
        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*cornerPointsSharp, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp =
            ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/aft_mapped";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

		/* 发布Flat类型的点云，尽管Flat点云也没有任何变化，再次发布出去 */
        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*surfPointsFlat, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp =
            ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/aft_mapped";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

		/* 发布完整点云，尽管完整点云也没有任何变化，再次发布出去 */
        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp =
            ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudFullRes3.header.frame_id = "/aft_mapped";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);
      }
      printf("publication time %f ms \n", t_pub.toc());
      printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
      if (t_whole.toc() > 100) ROS_WARN("odometry process over 100ms");

      frameCount++;
    }
    rate.sleep();
  }
  return 0;
}
