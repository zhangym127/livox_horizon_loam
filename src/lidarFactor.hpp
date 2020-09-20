// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen3/Eigen/Dense>

/* 定义基于仿函数的代价函数，总共有两个，下面是第一个，对边缘区域点进行拟合 */
struct LidarEdgeFactor {
  LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
                  Eigen::Vector3d last_point_b_, double s_)
      : curr_point(curr_point_),
        last_point_a(last_point_a_),
        last_point_b(last_point_b_),
        s(s_) {}

  template <typename T>

  /* q和t分别是要优化的旋转和平移，找到最优的q和t，使得两个点云的匹配最优。
   * residual是残差  */
  bool operator()(const T *q, const T *t, T *residual) const {
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()),
                               T(last_point_a.z())};
    Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()),
                               T(last_point_b.z())};

	/* 构造旋转四元数和平移向量 */
    // Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) *
    // q[2]};
    Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
    Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
	/* 下面根据时间戳进行帧内校正 */
	/* 根据时间戳的比例系数对旋转四元数进行插值 */
    q_last_curr = q_identity.slerp(T(s), q_last_curr);
	/* 根据时间戳的比例系数确定平移的量 */
    Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

	/* 对当前点进行旋转和平移，使其接近于last_point */
    Eigen::Matrix<T, 3, 1> lp;
    lp = q_last_curr * cp + t_last_curr;

	/* 以向量叉乘的方式求pa和pb所围成的平行四边形面积 */
    Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
	/* 求线段ab的长 */
    Eigen::Matrix<T, 3, 1> de = lpa - lpb;

	/* 获得残差，即点p到直线ab的距离 */
	/* 这里使用的是面积法，pa叉乘pb得到平行四边形的面积，除以对角线长度ab就得
	 * 到点p到直线ab的距离。 */
    residual[0] = nu.x() / de.norm();
    residual[1] = nu.y() / de.norm();
    residual[2] = nu.z() / de.norm();

    return true;
  }

  /* 创建代价函数 */
  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d last_point_a_,
                                     const Eigen::Vector3d last_point_b_,
                                     const double s_) {
	/* 创建代价函数的实例，第一个3是输出维度，即残差的维度，后续的4和3分别表示
	 * 待优化的旋转四元数q和平移t的维度。
	 * 注意：这里的new LidarEdgeFactor()调用的是构造函数，而不是仿函数 */
    return (new ceres::AutoDiffCostFunction<LidarEdgeFactor, 3, 4, 3>(
        new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
  }

  Eigen::Vector3d curr_point, last_point_a, last_point_b;
  double s;
};

/* 下面是第二个代价函数，对平面区域的点进行拟合 */
struct LidarPlaneFactor {
  LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
                   Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_,
                   double s_)
      : curr_point(curr_point_),
        last_point_j(last_point_j_),
        last_point_l(last_point_l_),
        last_point_m(last_point_m_),
        s(s_) {
	/* 如果j,l,m三个点都在一个平面上，则ljm_norm就是该平面的法向量	*/
    ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
    ljm_norm.normalize();
  }

  /* q和t分别是要优化的旋转和平移，找到最优的q和t，使得两个点云的匹配最优。
   * residual是残差  */
  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const {
	
	/* 分别构造当前点p的向量和上一帧点j的向量 */
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()),
                               T(last_point_j.z())};
    // Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()),
    // T(last_point_l.z())};
    // Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()),
    // T(last_point_m.z())};
	/* 构造j,l,m三个点所在平面的法向量 */
    Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()),
                               T(ljm_norm.z())};

    // Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) *
    // q[2]};
    Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
    Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
	/* 下面根据时间戳进行帧内校正 */
	/* 根据时间戳的比例系数对旋转四元数进行插值 */
    q_last_curr = q_identity.slerp(T(s), q_last_curr);
	/* 根据时间戳的比例系数确定平移的量 */
    Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

	/* 对当前点进行旋转和平移，使其接近于last_point */
    Eigen::Matrix<T, 3, 1> lp;
    lp = q_last_curr * cp + t_last_curr;

	/* 如果当前点p旋转平移之后的p'位于jlm的平面上，则向量p'j与该平面的法向量垂直，点积等于0。
     * 点p离jlm平面越近，则下面的点积的值越小，也就起到了优化的作用。	*/
    residual[0] = (lp - lpj).dot(ljm);

    return true;
  }

  /* 创建代价函数 */
  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d last_point_j_,
                                     const Eigen::Vector3d last_point_l_,
                                     const Eigen::Vector3d last_point_m_,
                                     const double s_) {
	/* 创建代价函数的实例，第一个3是输出维度，即残差的维度，后续的4和3分别表示
	 * 待优化的旋转四元数q和平移t的维度。
	 * 注意：这里的new LidarEdgeFactor()调用的是构造函数，而不是仿函数 */
    return (new ceres::AutoDiffCostFunction<LidarPlaneFactor, 1, 4, 3>(
        new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_,
                             last_point_m_, s_)));
  }

  Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
  Eigen::Vector3d ljm_norm;
  double s;
};

struct LidarPlaneNormFactor {
  LidarPlaneNormFactor(Eigen::Vector3d curr_point_,
                       Eigen::Vector3d plane_unit_norm_,
                       double negative_OA_dot_norm_)
      : curr_point(curr_point_),
        plane_unit_norm(plane_unit_norm_),
        negative_OA_dot_norm(negative_OA_dot_norm_) {}

  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const {
    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> point_w;
    point_w = q_w_curr * cp + t_w_curr;

    Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()),
                                T(plane_unit_norm.z()));
    residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d plane_unit_norm_,
                                     const double negative_OA_dot_norm_) {
    return (new ceres::AutoDiffCostFunction<LidarPlaneNormFactor, 1, 4, 3>(
        new LidarPlaneNormFactor(curr_point_, plane_unit_norm_,
                                 negative_OA_dot_norm_)));
  }

  Eigen::Vector3d curr_point;
  Eigen::Vector3d plane_unit_norm;
  double negative_OA_dot_norm;
};

struct LidarDistanceFactor {
  LidarDistanceFactor(Eigen::Vector3d curr_point_,
                      Eigen::Vector3d closed_point_)
      : curr_point(curr_point_), closed_point(closed_point_) {}

  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const {
    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> point_w;
    point_w = q_w_curr * cp + t_w_curr;

    residual[0] = point_w.x() - T(closed_point.x());
    residual[1] = point_w.y() - T(closed_point.y());
    residual[2] = point_w.z() - T(closed_point.z());
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d closed_point_) {
    return (new ceres::AutoDiffCostFunction<LidarDistanceFactor, 3, 4, 3>(
        new LidarDistanceFactor(curr_point_, closed_point_)));
  }

  Eigen::Vector3d curr_point;
  Eigen::Vector3d closed_point;
};