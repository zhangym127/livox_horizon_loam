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

#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/MarkerArray.h>
#include <cmath>
#include <string>
#include <vector>

#include "loam_horizon/common.h"
#include "loam_horizon/tic_toc.h"

using std::atan2;
using std::cos;
using std::sin;

constexpr bool dbg_show_id = false;
constexpr bool b_only_1_scan = false;
constexpr bool b_viz_curv = false;

constexpr bool b_normalize_curv = true;

const double scanPeriod = 0.1;

const int systemDelay = 0;
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;

float cloudCurvature[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp(int i, int j) { return (cloudCurvature[i] < cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

ros::Publisher pub_curvature;

bool PUB_EACH_LINE = true;

double MINIMUM_RANGE = 0.1;
double THRESHOLD_FLAT = 0.01;
double THRESHOLD_SHARP = 0.01;

/**
 * 删除到坐标原点的距离小于阈值thres的点
 * @param cloud_in 输入点云
 * @param cloud_out 输出点云
 * @param thres 阈值
 * @return 无
 */
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                            pcl::PointCloud<PointT> &cloud_out, float thres) {
  if (&cloud_in != &cloud_out) {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());
  }

  size_t j = 0;

  for (size_t i = 0; i < cloud_in.points.size(); ++i) {
    if (cloud_in.points[i].x * cloud_in.points[i].x +
            cloud_in.points[i].y * cloud_in.points[i].y +
            cloud_in.points[i].z * cloud_in.points[i].z <
        thres * thres)
      continue;
    cloud_out.points[j] = cloud_in.points[i];
    j++;
  }
  if (j != cloud_in.points.size()) {
    cloud_out.points.resize(j);
  }

  cloud_out.height = 1;
  cloud_out.width = static_cast<uint32_t>(j);
  cloud_out.is_dense = true;
}

/**
 * 显示点云的曲度
 * @param 
 * @return 无
 */
template <typename PointT>
void VisualizeCurvature(float *v_curv, int *v_label,
                        const pcl::PointCloud<PointT> &pcl_in,
                        const std_msgs::Header &hdr) {
  ROS_ASSERT(pcl_in.size() < 400000);

  /// Same marker attributes
  visualization_msgs::Marker txt_mk;
  txt_mk.header = hdr;
  txt_mk.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  txt_mk.ns = "default";
  txt_mk.id = 0;
  txt_mk.action = visualization_msgs::Marker::ADD;
  txt_mk.pose.orientation.x = 0;
  txt_mk.pose.orientation.y = 0;
  txt_mk.pose.orientation.z = 0;
  txt_mk.pose.orientation.w = 1;
  txt_mk.scale.z = 0.05;
  txt_mk.color.a = 1;
  txt_mk.color.r = 0;
  txt_mk.color.g = 1;
  txt_mk.color.b = 0;

  static visualization_msgs::MarkerArray curv_txt_msg;
  // for (size_t i = 0; i < curv_txt_msg.markers.size(); ++i) {
  //   auto &mk_i = curv_txt_msg.markers[i];
  //   mk_i = txt_mk;
  //   mk_i.header.stamp = txt_mk.header.stamp - ros::Duration(0.001);
  //   mk_i.action = visualization_msgs::Marker::DELETE;
  //   mk_i.text = "";
  //   mk_i.ns = "old";
  //   mk_i.color.a = 0;
  // }
  // pub_curvature.publish(curv_txt_msg);
  // ros::Rate r(200);
  // r.sleep();

  /// Marger array message
  static size_t pre_pt_num = 0;
  size_t pt_num = pcl_in.size();

  if (pre_pt_num == 0) {
    curv_txt_msg.markers.reserve(400000);
  }
  if (pre_pt_num > pt_num) {
    curv_txt_msg.markers.resize(pre_pt_num);
  } else {
    curv_txt_msg.markers.resize(pt_num);
  }

  int edge_num = 0, edgeless_num = 0, flat_num = 0, flatless_num = 0, nn = 0;

  /// Add marker and namespace
  for (size_t i = 0; i < pcl_in.size(); ++i) {
    auto curv = v_curv[i];
    auto label = v_label[i];  /// -1: flat, 0: less-flat, 1:less-edge, 2:edge
    const auto &pt = pcl_in[i];

    switch (label) {
      case 2: {
        /// edge
        auto &mk_i = curv_txt_msg.markers[i];
        mk_i = txt_mk;
        mk_i.ns = "edge";
        mk_i.id = i;
        mk_i.pose.position.x = pt.x;
        mk_i.pose.position.y = pt.y;
        mk_i.pose.position.z = pt.z;
        mk_i.color.a = 1;
        mk_i.color.r = 1;
        mk_i.color.g = 0;
        mk_i.color.b = 0;
        char cstr[10];
        snprintf(cstr, 9, "%.2f", curv);
        mk_i.text = std::string(cstr);
        /// debug
        if (dbg_show_id) {
          mk_i.text = std::to_string(i);
        }

        edge_num++;
        break;
      }
      case 1: {
        /// less edge
        auto &mk_i = curv_txt_msg.markers[i];
        mk_i = txt_mk;
        mk_i.ns = "edgeless";
        mk_i.id = i;
        mk_i.pose.position.x = pt.x;
        mk_i.pose.position.y = pt.y;
        mk_i.pose.position.z = pt.z;
        mk_i.color.a = 0.5;
        mk_i.color.r = 0.5;
        mk_i.color.g = 0;
        mk_i.color.b = 0.8;
        char cstr[10];
        snprintf(cstr, 9, "%.2f", curv);
        mk_i.text = std::string(cstr);
        /// debug
        if (dbg_show_id) {
          mk_i.text = std::to_string(i);
        }

        edgeless_num++;
        break;
      }
      case 0: {
        /// less flat
        auto &mk_i = curv_txt_msg.markers[i];
        mk_i = txt_mk;
        mk_i.ns = "flatless";
        mk_i.id = i;
        mk_i.pose.position.x = pt.x;
        mk_i.pose.position.y = pt.y;
        mk_i.pose.position.z = pt.z;
        mk_i.color.a = 0.5;
        mk_i.color.r = 0;
        mk_i.color.g = 0.5;
        mk_i.color.b = 0.8;
        char cstr[10];
        snprintf(cstr, 9, "%.2f", curv);
        mk_i.text = std::string(cstr);
        /// debug
        if (dbg_show_id) {
          mk_i.text = std::to_string(i);
        }

        flatless_num++;
        break;
      }
      case -1: {
        /// flat
        auto &mk_i = curv_txt_msg.markers[i];
        mk_i = txt_mk;
        mk_i.ns = "flat";
        mk_i.id = i;
        mk_i.pose.position.x = pt.x;
        mk_i.pose.position.y = pt.y;
        mk_i.pose.position.z = pt.z;
        mk_i.color.a = 1;
        mk_i.color.r = 0;
        mk_i.color.g = 1;
        mk_i.color.b = 0;
        char cstr[10];
        snprintf(cstr, 9, "%.2f", curv);
        mk_i.text = std::string(cstr);
        /// debug
        if (dbg_show_id) {
          mk_i.text = std::to_string(i);
        }

        flat_num++;
        break;
      }
      default: {
        /// Un-reliable
        /// Do nothing for label=99
        // ROS_ASSERT_MSG(false, "%d", label);
        auto &mk_i = curv_txt_msg.markers[i];
        mk_i = txt_mk;
        mk_i.ns = "unreliable";
        mk_i.id = i;
        mk_i.pose.position.x = pt.x;
        mk_i.pose.position.y = pt.y;
        mk_i.pose.position.z = pt.z;
        mk_i.color.a = 0;
        mk_i.color.r = 0;
        mk_i.color.g = 0;
        mk_i.color.b = 0;
        char cstr[10];
        snprintf(cstr, 9, "%.2f", curv);
        mk_i.text = std::string(cstr);

        nn++;
        break;
      }
    }
  }
  ROS_INFO("edge/edgeless/flatless/flat/nn num: [%d / %d / %d / %d / %d] - %lu",
           edge_num, edgeless_num, flatless_num, flat_num, nn, pt_num);

  /// Delete old points
  if (pre_pt_num > pt_num) {
    ROS_WARN("%lu > %lu", pre_pt_num, pt_num);
    // curv_txt_msg.markers.resize(pre_pt_num);
    for (size_t i = pt_num; i < pre_pt_num; ++i) {
      auto &mk_i = curv_txt_msg.markers[i];
      mk_i.action = visualization_msgs::Marker::DELETE;
      mk_i.color.a = 0;
      mk_i.color.r = 0;
      mk_i.color.g = 0;
      mk_i.color.b = 0;
      mk_i.ns = "old";
      mk_i.text = "";
    }
  }
  pre_pt_num = pt_num;

  pub_curvature.publish(curv_txt_msg);
}

/**
 * 处理帧内校正后的点云
 * 首先将点云按照扫描线顺序进行重排序，然后遍历每条扫描线，求每个点的曲度。
 * 然后按照曲度的不同，以及是否是邻点，分为Sharp、Lesssharp、Flat、Lessflat四类。
 * @param laserCloudMsg 完成帧内校正的点云，但是只校正了旋转，未校正平移
 * @return 无
 */
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {
  if (!systemInited) {
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    } else
      return;
  }

  TicToc t_whole;
  TicToc t_prepare;
  /* 扫描线指针，分别指向每个激光器对应扫描线的起点和终点 */
  std::vector<int> scanStartInd(N_SCANS, 0);
  std::vector<int> scanEndInd(N_SCANS, 0);

  /* 从ROSMsg取得校正后的点云 */
  pcl::PointCloud<PointType> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;

  /* 删除无效点，特别是到坐标原点距离小于MINIMUM_RANGE的点 */
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
  removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

  /* Horizon雷达有6条扫描线，根据扫描线的不同，把点云分成六份 */
  int cloudSize = laserCloudIn.points.size();
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn.points[i].x;
    point.y = laserCloudIn.points[i].y;
    point.z = laserCloudIn.points[i].z;
    point.intensity = laserCloudIn.points[i].intensity;
    point.curvature = laserCloudIn.points[i].curvature;
    int scanID = 0;
    if (N_SCANS == 6) {
      scanID = (int)point.intensity;
    }
    laserCloudScans[scanID].push_back(point);
  }

  cloudSize = count;
  printf("points size %d \n", cloudSize);

  /* 将六个扫描线的点云依次汇总在一起，并且初始化各扫描线的指针，分别指向各扫描线的起始和结束 */
  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++) {
    scanStartInd[i] = laserCloud->size() + 5;
    *laserCloud += laserCloudScans[i];
    scanEndInd[i] = laserCloud->size() - 6;
    // ROS_INFO("scan %d start-end [%d, %d]", i, scanStartInd[i],
    // scanEndInd[i]);
  }

  printf("prepare time %f \n", t_prepare.toc());

  int kNumCurvSize = 5;
  constexpr int kNumRegion = 50;       // 6
  constexpr int kNumEdge = 2;          // 2
  constexpr int kNumFlat = 4;          // 4
  constexpr int kNumEdgeNeighbor = 5;  // 5;
  constexpr int kNumFlatNeighbor = 5;  // 5;
  float kThresholdSharp = 50;          // 0.1;
  float kThresholdFlat = 30;           // 0.1;
  constexpr float kThresholdLessflat = 0.1;

  /* 遍历按照扫描线重新排序的点云，求每个点的曲度 */
  constexpr float kDistanceFaraway = 25;
  for (int i = 5; i < cloudSize - 5; i++) {
	/* 如果当前点的距离>25米，取两侧的4个点，否则取两侧的10个点 */
    float dis = sqrt(laserCloud->points[i].x * laserCloud->points[i].x +
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);
    if (dis > kDistanceFaraway) {
      kNumCurvSize = 2;
    }
	/* 分别求当前点两侧的4个或10个点的x,y,z之和	*/
    float diffX = 0, diffY = 0, diffZ = 0;
    for (int j = 1; j <= kNumCurvSize; ++j) {
      diffX += laserCloud->points[i - j].x + laserCloud->points[i + j].x;
      diffY += laserCloud->points[i - j].y + laserCloud->points[i + j].y;
      diffZ += laserCloud->points[i - j].z + laserCloud->points[i + j].z;
    }
	/* 再分别减去同等数量的当前点的x,y,z之和 */
    diffX -= 2 * kNumCurvSize * laserCloud->points[i].x;
    diffY -= 2 * kNumCurvSize * laserCloud->points[i].y;
    diffZ -= 2 * kNumCurvSize * laserCloud->points[i].z;
	
	/* 如果当前点是一个角点，即位于边缘的点，则diff的绝对值比较大
     * 如果当前点是平面中间的点，则diff的绝对值较小，趋近于0 */
    /*
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x +
                  laserCloud->points[i - 3].x + laserCloud->points[i - 2].x +
                  laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x +
                  laserCloud->points[i + 1].x + laserCloud->points[i + 2].x +
                  laserCloud->points[i + 3].x + laserCloud->points[i + 4].x +
                  laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y +
                  laserCloud->points[i - 3].y + laserCloud->points[i - 2].y +
                  laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y +
                  laserCloud->points[i + 1].y + laserCloud->points[i + 2].y +
                  laserCloud->points[i + 3].y + laserCloud->points[i + 4].y +
                  laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z +
                  laserCloud->points[i - 3].z + laserCloud->points[i - 2].z +
                  laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z +
                  laserCloud->points[i + 1].z + laserCloud->points[i + 2].z +
                  laserCloud->points[i + 3].z + laserCloud->points[i + 4].z +
                  laserCloud->points[i + 5].z;
                  */

	/* 求diff的平方和及其平方根，获得当前点的曲度，曲度越大，说明当前点是角点 */
    float tmp2 = diffX * diffX + diffY * diffY + diffZ * diffZ;
    float tmp = sqrt(tmp2);

	/* 记录每个点的曲度，可以选择平方和，或者规格化两种 */
    cloudCurvature[i] = tmp2;
    if (b_normalize_curv) {
      /// use normalized curvature
      cloudCurvature[i] = tmp / (2 * kNumCurvSize * dis + 1e-3);
    }
    cloudSortInd[i] = i;
    cloudNeighborPicked[i] = 0;
    cloudLabel[i] = 0;

	/* 对无效点进行标识，例如距离过大或者过小 */
    /// Mark un-reliable points
    constexpr float kMaxFeatureDis = 1e4;
    if (fabs(dis) > kMaxFeatureDis || fabs(dis) < 1e-4 || !std::isfinite(dis)) {
      cloudLabel[i] = 99; /* 99表示无效点 */
      cloudNeighborPicked[i] = 1;
    }
  }

  /* 再次遍历按照扫描线重新排序的点云，将离点标识出来 */
  for (int i = 5; i < cloudSize - 6; i++) {
	/* 求后面一个点与当前点的欧式距离 */
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;
	/* 求当前点与前面一个点的欧式距离 */
    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;
	/* 求当前点到原点的欧式距离 */
    float dis = laserCloud->points[i].x * laserCloud->points[i].x +
                laserCloud->points[i].y * laserCloud->points[i].y +
                laserCloud->points[i].z * laserCloud->points[i].z;
	/* 如果前后的点同时距离当前点足够远，则进行标识 */
    if (diff > 0.00015 * dis && diff2 > 0.00015 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }

  TicToc t_pts;

  /* 将点云根据曲度分成四类 */
  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  if (b_normalize_curv) {
    kThresholdFlat = THRESHOLD_FLAT;
    kThresholdSharp = THRESHOLD_SHARP;
  }

  /* 逐一遍历每个扫描线 */
  float t_q_sort = 0;
  for (int i = 0; i < N_SCANS; i++) {
    if (scanEndInd[i] - scanStartInd[i] < kNumCurvSize) continue;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(
        new pcl::PointCloud<PointType>);

    /// debug
    if (b_only_1_scan && i > 0) {
      break;
    }

	/* 将每个扫描线依次划分成若干(50)个区，遍历每个区 */
    for (int j = 0; j < kNumRegion; j++) {
	  /* sp指向每个区的起点，ep指向每个区的终点， */
      int sp =
          scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / kNumRegion;
      int ep = scanStartInd[i] +
               (scanEndInd[i] - scanStartInd[i]) * (j + 1) / kNumRegion - 1;
      //      ROS_INFO("scan [%d], id from-to [%d-%d] in [%d-%d]", i, sp, ep,
      //               scanStartInd[i], scanEndInd[i]);

      TicToc t_tmp;
	  /* 对区内点云按照曲度大小进行排序，从小到大，顺序记录在cloudSortInd中 */
      //      std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);
      // sort the curvatures from small to large
      for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
          if (cloudCurvature[cloudSortInd[l]] <
              cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }

	  /* 求每个区最大曲度，以及区内曲度累加和 */
      float SumCurRegion = 0.0;
      float MaxCurRegion = cloudCurvature[cloudSortInd[ep]];  //the largest curvature in sp ~ ep
      for (int k = ep - 1; k >= sp; k--) {
        SumCurRegion += cloudCurvature[cloudSortInd[k]];
      }

	  /* 如果最大曲度大于区内曲度累加和的三倍，则标识该最大点 */
      if (MaxCurRegion > 3 * SumCurRegion)
        cloudNeighborPicked[cloudSortInd[ep]] = 1;

      t_q_sort += t_tmp.toc();

	  /* 确认区内的点按照cloudSortInd记录的顺序，曲度从小到大排列 */
      if (true) {
        for (int tt = sp; tt < ep - 1; ++tt) {
          ROS_ASSERT(cloudCurvature[cloudSortInd[tt]] <=
                     cloudCurvature[cloudSortInd[tt + 1]]);
        }
      }

	  /* 从曲度高到底的顺序反向遍历区内的点 */
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];

		/* 之前已经被标识的点排除在外 */
        if (cloudNeighborPicked[ind] != 0) continue;

		/* 选择曲度大于Sharp阈值的点 */
        if (cloudCurvature[ind] > kThresholdSharp) {
          largestPickedNum++;
		  /* 每个区选曲度最高的2个点，归类为Sharp，最高的20个点归类为LessSharp */
          if (largestPickedNum <= kNumEdge) {
            cloudLabel[ind] = 2; /* 2表示曲度为Sharp的点 */
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
            // ROS_INFO("pick sharp at sort_id [%d], primary id[%d]", k, ind);
            // if (ind == 211 || ind == 212 || ind == 213 || ind == 214) {
            //   const auto &pt = laserCloud->points[ind];
            //   printf("%d-[%f, %f, %f]\n", ind, pt.x, pt.y, pt.z);
            // }
          } else if (largestPickedNum <= 20) {
            cloudLabel[ind] = 1; /* 2表示曲度为LessSharp的点 */
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break; /* 只处理曲度最高的20个点，超出则跳出循环 */
          }

		  /* 如果曲度大于Sharp阈值，则标识 */
          cloudNeighborPicked[ind] = 1;

		  /* 检查当前点左右的10个邻点，如果欧式距离小于0.02的阈值则标识 */
          for (int l = 1; l <= kNumEdgeNeighbor; l++) {
            float diffX = laserCloud->points[ind + l].x -
                          laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y -
                          laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z -
                          laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -kNumEdgeNeighbor; l--) {
            float diffX = laserCloud->points[ind + l].x -
                          laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y -
                          laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z -
                          laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

	  /* 从曲度低到高的顺序正向遍历区内的点 */
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];

		/* 之前已经被标识的点排除在外 */
        if (cloudNeighborPicked[ind] != 0) continue;

		/* 选择曲度小于Flat阈值的点，归类为Flat */
        if (cloudCurvature[ind] < kThresholdFlat) {
          cloudLabel[ind] = -1; /* -1表示曲度为Flat的点 */
          surfPointsFlat.push_back(laserCloud->points[ind]);
          cloudNeighborPicked[ind] = 1;

		  /* 选择曲度最小的4个点，超出则跳出循环 */
          smallestPickedNum++;
          if (smallestPickedNum >= kNumFlat) {
            break;
          }

		  /* 检查当前点左右的10个邻点，如果欧式距离小于0.02的阈值则标识 */
          for (int l = 1; l <= kNumFlatNeighbor; l++) {
            float diffX = laserCloud->points[ind + l].x -
                          laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y -
                          laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z -
                          laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -kNumFlatNeighbor; l--) {
            float diffX = laserCloud->points[ind + l].x -
                          laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y -
                          laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z -
                          laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

	  /* 选择曲度小于Lessflat阈值的点，归类为LessFlatScan */
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0 && cloudCurvature[k] < kThresholdLessflat) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

    surfPointsLessFlat += surfPointsFlat;
    cornerPointsLessSharp += cornerPointsSharp;
    /// Whether downsample less-flat points
    if (false) {
      pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
      pcl::VoxelGrid<PointType> downSizeFilter;
      downSizeFilter.setInputCloud(surfPointsLessFlatScan);
      downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
      downSizeFilter.filter(surfPointsLessFlatScanDS);
      surfPointsLessFlat += surfPointsLessFlatScanDS;
    } else {
      surfPointsLessFlat += *surfPointsLessFlatScan;
    }
  }
  printf("sort q time %f \n", t_q_sort);
  printf("seperate points time %f \n", t_pts.toc());

  if (false) {
    removeClosedPointCloud(*laserCloud, *laserCloud, MINIMUM_RANGE);
    removeClosedPointCloud(cornerPointsLessSharp, cornerPointsLessSharp,
                           MINIMUM_RANGE);
    removeClosedPointCloud(cornerPointsSharp, cornerPointsSharp, MINIMUM_RANGE);
    removeClosedPointCloud(surfPointsFlat, surfPointsFlat, MINIMUM_RANGE);
    removeClosedPointCloud(surfPointsLessFlat, surfPointsLessFlat,
                           MINIMUM_RANGE);
  }

  /// Visualize curvature
  /* 显示点云的曲度 */
  if (b_viz_curv) {
    std_msgs::Header ros_hdr = laserCloudMsg->header;
    ros_hdr.frame_id = "/aft_mapped";
    VisualizeCurvature(cloudCurvature, cloudLabel, *laserCloud, ros_hdr);
  }

  /* 发布按照扫描线重排序的点云 */
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/aft_mapped";
  pubLaserCloud.publish(laserCloudOutMsg);

  /* 发布曲度为Sharp的点云 */
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/aft_mapped";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  /* 发布曲度为LessSharp的点云 */
  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/aft_mapped";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  /* 发布曲度为Flat的点云 */
  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/aft_mapped";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  /* 发布曲度为LessFlat的点云 */
  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/aft_mapped";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  /* 发布每个扫描线的点云 */
  // pub each scam
  if (PUB_EACH_LINE) {
    for (int i = 0; i < N_SCANS; i++) {
      sensor_msgs::PointCloud2 scanMsg;
      pcl::toROSMsg(laserCloudScans[i], scanMsg);
      scanMsg.header.stamp = laserCloudMsg->header.stamp;
      scanMsg.header.frame_id = "/aft_mapped";
      pubEachScan[i].publish(scanMsg);
    }
  }

  printf("scan registration time %f ms *************\n", t_whole.toc());
  if (t_whole.toc() > 100) ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  nh.param<int>("scan_line", N_SCANS, 6); // Horizon has 6 scan lines
  nh.param<double>("threshold_flat", THRESHOLD_FLAT, 0.01);
  nh.param<double>("threshold_sharp", THRESHOLD_SHARP, 0.01);

  printf("scan line number %d \n", N_SCANS);

  if (N_SCANS != 6) {
    printf("only support livox horizon lidar!");
    return 0;
  }

  /* 订阅帧内校正后的点云，注册回调函数laserCloudHandler进行点云的处理 */
  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
      "/livox_undistort", 100, laserCloudHandler);

  /* 获得publisher对象，准备在完成点云的处理后进行发布 */
  pubLaserCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

  pubCornerPointsSharp =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

  pubCornerPointsLessSharp =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

  pubSurfPointsFlat =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

  pubSurfPointsLessFlat =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

  pubRemovePoints =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

  pub_curvature =
      nh.advertise<visualization_msgs::MarkerArray>("/curvature", 100);

  if (PUB_EACH_LINE) {
    for (int i = 0; i < N_SCANS; i++) {
      ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>(
          "/laser_scanid_" + std::to_string(i), 100);
      pubEachScan.push_back(tmp);
    }
  }
  ros::spin();

  return 0;
}
