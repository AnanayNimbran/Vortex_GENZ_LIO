// ===========================================================================
//  genz_lio.hpp
//
//  Independent Clean-room C++ reimplementation of:
//    D. Lee, H. Lim, S. Kim, et al.,
//    "GenZ-LIO: Generalizable LiDAR-Inertial Odometry Beyond Confined-Open
//     Boundaries", arXiv:2603.16273v2 (15 Jun 2026).
//
//  This header declares the full estimator. Equation / algorithm numbers in the
//  comments refer to that paper. Three novel contributions are implemented:
//
//    (IV)  Scale-aware adaptive voxelization     -> AdaptiveVoxelizer   (Eq.1-8,
//                                                    Algorithm 1)
//    (V-B) Voxel-pruned correspondence search    -> VoxelMap            (Fig.5/6,
//                                                    Algorithm 2)
//    (V)   Hybrid-metric ESIKF state update       -> Esikf              (Eq.9-36,
//                                                    Algorithm 3)
//
//  Pieces the paper explicitly delegates to prior work are implemented from the
//  cited reference and labelled accordingly:
//    * per-point / plane uncertainty  -> Yuan et al. "VoxelMap" [11]
//    * ESIKF prediction & reparam. J  -> Xu et al. "FAST-LIO2"  [6]
//
//  The geometry-only parts (SO(3), boxplus/boxminus, the PD controller, and the
//  voxel-pruning geometry) are header-inline so they can be unit tested without
//  ROS or PCL. Define GENZ_LIO_HEADER_ONLY before including this header to drop
//  the ROS node declaration (used by test/test_genz_math.cpp).
// ===========================================================================
#ifndef GENZ_LIO__GENZ_LIO_HPP_
#define GENZ_LIO__GENZ_LIO_HPP_

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace genz_lio
{

// ---------------------------------------------------------------------------
//  Convenience aliases
// ---------------------------------------------------------------------------
using Vec3   = Eigen::Vector3d;
using Mat3   = Eigen::Matrix3d;
using Vec18  = Eigen::Matrix<double, 18, 1>;
using Mat18  = Eigen::Matrix<double, 18, 18>;
using Mat12  = Eigen::Matrix<double, 12, 12>;
using Mat18x12 = Eigen::Matrix<double, 18, 12>;

// Error-state index layout (matches state ordering of Eq.10 / Eq.12):
//   [ 0: 3) -> d(W t_I)   position
//   [ 3: 6) -> d(W R_I)   rotation (SO(3) tangent, right perturbation)
//   [ 6: 9) -> d(W v_I)   velocity
//   [ 9:12) -> d b_g      gyro bias
//   [12:15) -> d b_a      accel bias
//   [15:18) -> d(W g)     gravity
enum : int { IDX_P = 0, IDX_R = 3, IDX_V = 6, IDX_BG = 9, IDX_BA = 12, IDX_G = 15 };
static constexpr int STATE_DIM = 18;
static constexpr int NOISE_DIM = 12;  // [ n_g, n_a, n_bg, n_ba ]

// ===========================================================================
//  SO(3) utilities  (right-perturbation convention: R_true = R_hat * Exp(d))
// ===========================================================================
namespace so3
{

inline Mat3 hat(const Vec3 & w)
{
  Mat3 S;
  S <<      0, -w.z(),  w.y(),
        w.z(),      0, -w.x(),
       -w.y(),  w.x(),      0;
  return S;
}

// Exponential map: rotation vector -> rotation matrix (Rodrigues).
inline Mat3 Exp(const Vec3 & w)
{
  const double th = w.norm();
  if (th < 1e-9) {
    // 2nd-order series, numerically stable near 0.
    return Mat3::Identity() + hat(w) + 0.5 * hat(w) * hat(w);
  }
  const Vec3 a = w / th;
  const Mat3 K = hat(a);
  return Mat3::Identity() + std::sin(th) * K + (1.0 - std::cos(th)) * K * K;
}

// Logarithm map: rotation matrix -> rotation vector.
inline Vec3 Log(const Mat3 & R)
{
  const double cos_th = std::min(1.0, std::max(-1.0, 0.5 * (R.trace() - 1.0)));
  const double th = std::acos(cos_th);
  Vec3 w(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  if (th < 1e-9) {
    return 0.5 * w;  // sin(th)/th -> 1/2 * (R - R^T)^vee
  }
  if (std::abs(M_PI - th) < 1e-6) {
    // Near pi: extract axis from (R + I)/2 diagonal to keep sign/magnitude.
    const Mat3 M = 0.5 * (R + Mat3::Identity());
    Vec3 axis(std::sqrt(std::max(0.0, M(0, 0))),
              std::sqrt(std::max(0.0, M(1, 1))),
              std::sqrt(std::max(0.0, M(2, 2))));
    if (w.x() < 0) axis.x() = -axis.x();
    if (w.y() < 0) axis.y() = -axis.y();
    if (w.z() < 0) axis.z() = -axis.z();
    return th * axis.normalized();
  }
  return (th / (2.0 * std::sin(th))) * w;
}

// Right Jacobian of SO(3).
inline Mat3 rightJacobian(const Vec3 & w)
{
  const double th = w.norm();
  const Mat3 K = hat(w);
  if (th < 1e-7) {
    return Mat3::Identity() - 0.5 * K + (1.0 / 6.0) * K * K;
  }
  const double th2 = th * th, th3 = th2 * th;
  return Mat3::Identity()
       - ((1.0 - std::cos(th)) / th2) * K
       + ((th - std::sin(th)) / th3) * K * K;
}

// Inverse right Jacobian of SO(3).
inline Mat3 rightJacobianInv(const Vec3 & w)
{
  const double th = w.norm();
  const Mat3 K = hat(w);
  if (th < 1e-7) {
    return Mat3::Identity() + 0.5 * K + (1.0 / 12.0) * K * K;
  }
  const double th2 = th * th;
  const double half = 0.5;
  const double c = (1.0 / th2) - (1.0 + std::cos(th)) / (2.0 * th * std::sin(th));
  return Mat3::Identity() + half * K + c * K * K;
}

}  // namespace so3

// ===========================================================================
//  State  (manifold M = SO(3) x R^15, dim = 18; Eq.10)
//    x = [ W t_I , W R_I , W v_I , b_g , b_a , W g ]
// ===========================================================================
struct State
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vec3 pos  = Vec3::Zero();          // W t_I
  Mat3 rot  = Mat3::Identity();      // W R_I
  Vec3 vel  = Vec3::Zero();          // W v_I
  Vec3 bg   = Vec3::Zero();          // gyro bias
  Vec3 ba   = Vec3::Zero();          // accel bias
  Vec3 grav = Vec3(0, 0, -9.81);     // W g

  // x [+] dx  (boxplus on the manifold; SO(3) part is a RIGHT perturbation).
  State boxplus(const Vec18 & dx) const
  {
    State s;
    s.pos  = pos  + dx.segment<3>(IDX_P);
    s.rot  = rot * so3::Exp(dx.segment<3>(IDX_R));
    s.vel  = vel  + dx.segment<3>(IDX_V);
    s.bg   = bg   + dx.segment<3>(IDX_BG);
    s.ba   = ba   + dx.segment<3>(IDX_BA);
    s.grav = grav + dx.segment<3>(IDX_G);
    return s;
  }

  // x [-] y  (boxminus; inverse of boxplus, consistent with right perturbation).
  Vec18 boxminus(const State & y) const
  {
    Vec18 d;
    d.segment<3>(IDX_P)  = pos  - y.pos;
    d.segment<3>(IDX_R)  = so3::Log(y.rot.transpose() * rot);
    d.segment<3>(IDX_V)  = vel  - y.vel;
    d.segment<3>(IDX_BG) = bg   - y.bg;
    d.segment<3>(IDX_BA) = ba   - y.ba;
    d.segment<3>(IDX_G)  = grav - y.grav;
    return d;
  }
};

// Reparametrization matrix J^l : partial of ((x_hat^l [+] dx) [-] x_hat) wrt dx,
// evaluated at dx = 0 (FAST-LIO2 [6], used in Eq.33-35). It is identity except
// on the SO(3) block, which is the inverse right Jacobian of the rotation error
// between the iterate and the propagated prior.
inline Mat18 reparamJacobian(const State & x_iter, const State & x_prior)
{
  Mat18 J = Mat18::Identity();
  const Vec3 dtheta = so3::Log(x_prior.rot.transpose() * x_iter.rot);
  J.block<3, 3>(IDX_R, IDX_R) = so3::rightJacobianInv(dtheta);
  return J;
}

// ===========================================================================
//  Raw measurement containers
// ===========================================================================
struct ImuData
{
  double t = 0.0;        // [s]
  Vec3 acc = Vec3::Zero(); // specific force [m/s^2]
  Vec3 gyr = Vec3::Zero(); // angular rate  [rad/s]
};

// One LiDAR return. `time` is the per-point timestamp relative to the cloud
// header stamp [s] (used for deskewing); 0 if the driver does not provide it.
struct PointXYZIT
{
  double x = 0, y = 0, z = 0;
  float intensity = 0.f;
  double time = 0.0;
  Vec3 vec() const { return Vec3(x, y, z); }
};
using PointCloud = std::vector<PointXYZIT>;

// ===========================================================================
//  Parameters (defaults = "System Configurations", Sec.VI-A-3 of the paper)
// ===========================================================================
struct Params
{
  // ---- Scale-aware adaptive voxelization (Sec.IV) ----
  double d0        = 0.5;     // initial voxel size d_0 [m]
  double d_min     = 0.02;    // [d_min, d_max] = [0.02, 1.0] m
  double d_max     = 1.0;
  int    N_min     = 1000;    // [N_min, N_max] = [1000, 4000]
  int    N_max     = 4000;
  double tau_m     = 30.0;    // spatial-scale threshold [m]
  double p_exp     = 2.0;     // setpoint exponent p (>1)
  int    N_w       = 5;       // median sliding-window size
  double lambda_p  = 0.1;     // tracking-error normalizer
  double lambda_d  = 1.0;     // error-derivative normalizer
  double Kp_min    = 1e-6;    // proportional-gain bounds
  double Kp_max    = 1e-4;
  double Kd_min    = 1e-9;    // derivative-gain bounds
  double Kd_max    = 1e-7;

  // ---- Voxel-pruned correspondence search / map (Sec.V-B) ----
  double d_root        = 0.5;   // fixed root voxel size [m]
  double tau_closest   = 1.0;   // point-to-point outlier-rejection distance [m]
  int    plane_min_pts = 6;     // min points to attempt a plane fit
  double plane_ratio   = 0.10;  // planarity: lambda_3 <= plane_ratio * lambda_2
  int    max_pts_voxel = 50;    // N_stored cap per root voxel
  double gate_sigma    = 3.0;   // 3-sigma plane gating

  // ---- Hybrid-metric state update (Sec.V) ----
  double eps_po        = 1e-6;  // near-zero point-to-point residual tolerance [m]
  double lambda_po     = 0.1;   // point-to-point covariance scaling (Eq.30)
  int    max_iter      = 10;    // ESIKF max iterations
  double tau_converge  = 1e-3;  // convergence threshold on ||dx||

  // ---- Sensor noise (Intel RealSense D455 IMU; tune to the unit) ----
  double acc_noise   = 1.0e-2;  // accelerometer noise density
  double gyr_noise   = 1.0e-3;  // gyroscope noise density
  double acc_bias_rw = 1.0e-4;  // accel bias random walk
  double gyr_bias_rw = 1.0e-5;  // gyro bias random walk
  double lidar_range_std   = 0.02;  // sigma_d  [m]
  double lidar_bearing_std = 0.0017;// sigma_omega [rad] (~0.1 deg)

  // ---- Extrinsics  L (LiDAR) -> I (IMU/body)  : I_T_L = (R_IL, t_IL) ----
  Mat3 R_IL = Mat3::Identity();
  Vec3 t_IL = Vec3::Zero();

  // ---- LiDAR geometry / range gating ----
  double lidar_min_range = 0.1;    // discard returns closer than this [m]
  double lidar_max_range = 100.0;  // discard returns farther than this [m]
  double lidar_fov_down  = -15.0;  // vertical FoV lower bound [deg]
  double lidar_fov_up    = 15.0;   // vertical FoV upper bound [deg]
  int    lidar_scan_lines = 64;    // number of vertical channels / scan lines

  // ---- Init ----
  int    imu_init_count = 100;  // static IMU samples for gravity/bias init
};

// ===========================================================================
//  Voxel hashing
// ===========================================================================
struct VoxelKey
{
  int64_t x = 0, y = 0, z = 0;
  bool operator==(const VoxelKey & o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & k) const
  {
    // Large primes, a la VoxelMap / Faster-LIO spatial hashing.
    return (static_cast<std::size_t>(k.x) * 73856093) ^
           (static_cast<std::size_t>(k.y) * 19349663) ^
           (static_cast<std::size_t>(k.z) * 83492791);
  }
};

inline VoxelKey keyOf(const Vec3 & p, double leaf)
{
  return VoxelKey{
    static_cast<int64_t>(std::floor(p.x() / leaf)),
    static_cast<int64_t>(std::floor(p.y() / leaf)),
    static_cast<int64_t>(std::floor(p.z() / leaf))};
}

// Voxel-keyed hash map whose mapped value carries fixed-size Eigen members
// (e.g. VoxelBlock -> PlaneData's 6x6 covariance). The Eigen::aligned_allocator
// guarantees correct 16/32-byte alignment of the hash nodes on x86 (required
// once -march=native enables SSE/AVX vectorization).
template <typename V>
using VoxelHashMap = std::unordered_map<
  VoxelKey, V, VoxelKeyHash, std::equal_to<VoxelKey>,
  Eigen::aligned_allocator<std::pair<const VoxelKey, V>>>;

// Uniform voxel-grid downsample: returns one centroid per occupied voxel.
// This is the "Voxelize(., d)" operator used throughout Algorithm 1.
PointCloud voxelDownsample(const PointCloud & in, double leaf);

// ===========================================================================
//  (IV) Scale-aware adaptive voxelization
//      PD controller w/ sensitivity-informed gain scheduling (Eq.1-8, Alg.1)
// ===========================================================================
class AdaptiveVoxelizer
{
public:
  explicit AdaptiveVoxelizer(const Params & p) : prm_(p), d_prev_(p.d0) {}

  struct Result
  {
    PointCloud V_t;        // re-voxelized at d_t          -> state update
    PointCloud V_merge_t;  // voxelized at d_t/2           -> map integration
    double d_t = 0.0;      // chosen voxel size
    double m_bar = 0.0;    // smoothed scale indicator
    int    N_desired = 0;  // scale-informed setpoint
    int    N_temp = 0;     // temporary voxelized point count
  };

  // Algorithm 1. `deskewed` is S_t (LiDAR frame, motion-compensated).
  Result process(const PointCloud & deskewed, double dt_scan);

  double currentVoxelSize() const { return d_prev_; }

  // --- Pure PD-control math (Eq.1-8). Header-inline so it is unit-testable. ---
  // Given the smoothed scale m_bar, current temporary count N_temp and the
  // scan period dt_scan, returns the next voxel size d_t and fills diagnostics.
  struct Control { double d_t; int N_desired; double Kp, Kd, e, de; };
  Control computeControl(double m_bar, int N_temp, double dt_scan)
  {
    const Params & q = prm_;

    // -- Scale-informed setpoint N_desired,t  (Eq.1-2) --
    int N_desired;
    if (m_bar >= q.tau_m) {
      N_desired = q.N_max;
    } else {
      const double rho = 1.0 - std::pow(1.0 - m_bar / q.tau_m, q.p_exp);  // Eq.2
      N_desired = static_cast<int>(
        std::round(q.N_min + (q.N_max - q.N_min) * rho));               // Eq.1
    }

    // -- Tracking error and its derivative (Eq.3) --
    const double e  = static_cast<double>(N_desired - N_temp);
    const double de = (e - e_prev_) / std::max(dt_scan, 1e-6);

    // -- Sensitivity-informed gain scheduling (Eq.4-6) --
    const double phi = std::min(m_bar, q.tau_m) / q.tau_m;               // phi_t
    const double dN  = static_cast<double>(N_desired);
    const double psi_p = std::min(std::abs(e), q.lambda_p * dN) /
                         std::max(q.lambda_p * dN, 1e-9);                // psi_p,t
    const double de_cap = q.lambda_d * dN / std::max(dt_scan, 1e-6);
    const double psi_d = std::min(std::abs(de), de_cap) /
                         std::max(de_cap, 1e-9);                         // psi_d,t

    const double Gamma_p = std::sqrt(phi * psi_p);                       // Eq.5
    const double Gamma_d = std::sqrt(phi * psi_d);
    const double Kp = q.Kp_min + (q.Kp_max - q.Kp_min) * Gamma_p;        // Eq.6
    const double Kd = q.Kd_min + (q.Kd_max - q.Kd_min) * Gamma_d;

    // -- PD control law + voxel-size update (Eq.7-8) --
    const double dd = -Kp * e - Kd * de;                                 // Eq.7
    double d_t = d_prev_ + dd;
    d_t = std::min(q.d_max, std::max(q.d_min, d_t));                     // Eq.8 clamp

    return Control{d_t, N_desired, Kp, Kd, e, de};
  }

  // Push a fresh median range into the sliding window and return m_bar
  // (moving average, Sec.IV-A). Header-inline for testability.
  double pushMedian(double m_t)
  {
    window_.push_back(m_t);
    if (static_cast<int>(window_.size()) > prm_.N_w) window_.pop_front();
    double s = 0.0;
    for (double v : window_) s += v;
    return s / static_cast<double>(window_.size());
  }

  // Reset internal controller memory (e_{t-1}, d_{t-1}, window).
  void reset()
  {
    d_prev_ = prm_.d0;
    e_prev_ = 0.0;
    window_.clear();
  }

private:
  Params prm_;
  std::deque<double> window_;  // sliding window W of median ranges
  double d_prev_;              // d_{t-1}
  double e_prev_ = 0.0;        // e_{t-1}
};

// ===========================================================================
//  Voxel-pruning geometry  (Fig.5: candidate voxels, Fig.6: distance-to-voxel)
//  Header-inline so the geometry can be unit tested directly.
// ===========================================================================
namespace voxutil
{

// Classify which third of the root voxel (along each axis) the query point sits
// in:  -1 = low third, 0 = middle third, +1 = high third (the "27 regions").
inline Eigen::Vector3i regionSign(const Vec3 & p, const VoxelKey & root, double d)
{
  Eigen::Vector3i s;
  for (int i = 0; i < 3; ++i) {
    const double base = static_cast<double>(i == 0 ? root.x : (i == 1 ? root.y : root.z)) * d;
    const double local = p[i] - base;          // in [0, d)
    if (local < d / 3.0)        s[i] = -1;
    else if (local < 2.0 * d / 3.0) s[i] = 0;
    else                        s[i] = +1;
  }
  return s;
}

// Candidate neighbour offsets selected by the sharing relation (Fig.5):
//   center (0 non-mid axes)  -> {}              (1 voxel  total: root only)
//   surface (1 non-mid axis) -> 1 neighbour     (2 voxels total)
//   edge    (2 non-mid axes) -> 3 neighbours    (4 voxels total)
//   corner  (3 non-mid axes) -> 7 neighbours    (8 voxels total)
// The root offset (0,0,0) is intentionally excluded from the returned list.
inline std::vector<Eigen::Vector3i> candidateOffsets(const Eigen::Vector3i & s)
{
  std::vector<Eigen::Vector3i> out;
  // Per axis the offset is either 0 or s[i] (when s[i] != 0).
  for (int ox : {0, s[0]}) {
    for (int oy : {0, s[1]}) {
      for (int oz : {0, s[2]}) {
        const Eigen::Vector3i o(ox, oy, oz);
        if (o.isZero()) continue;            // skip root
        // Guard against duplicates when some s[i] == 0 (then {0, 0}).
        bool dup = false;
        for (const auto & e : out) if (e == o) { dup = true; break; }
        if (!dup) out.push_back(o);
      }
    }
  }
  return out;
}

// Minimum distance d_nbr from the query point to the candidate neighbour voxel
// (Fig.6). For axes shared with the neighbour (offset != 0) the distance is to
// the corresponding face plane; axes that are not shared contribute 0. This
// yields point-to-surface / point-to-edge / point-to-corner distances for the
// surface / edge / corner cases respectively.
inline double computeDistanceToVoxel(const Vec3 & p, const VoxelKey & root,
                                     const Eigen::Vector3i & offset, double d)
{
  double sq = 0.0;
  for (int i = 0; i < 3; ++i) {
    if (offset[i] == 0) continue;
    const int64_t base_idx = (i == 0 ? root.x : (i == 1 ? root.y : root.z));
    // Far boundary of the root voxel in the offset direction.
    const double boundary = static_cast<double>(base_idx + (offset[i] > 0 ? 1 : 0)) * d;
    const double diff = p[i] - boundary;
    sq += diff * diff;
  }
  return std::sqrt(sq);
}

inline VoxelKey shift(const VoxelKey & k, const Eigen::Vector3i & o)
{
  return VoxelKey{k.x + o.x(), k.y + o.y(), k.z + o.z()};
}

}  // namespace voxutil

// ===========================================================================
//  Voxel map  (root voxels of fixed size d_root; Sec.V-B)
// ===========================================================================
struct PlaneData
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool   valid = false;
  Vec3   normal   = Vec3::Zero();           // W n
  Vec3   centroid = Vec3::Zero();           // W q
  Eigen::Matrix<double, 6, 6> cov_nq        // Sigma_{w_n, w_q}  (Eq.19 block)
    = Eigen::Matrix<double, 6, 6>::Zero();
};

struct StoredPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vec3 p = Vec3::Zero();                     // W p'   (world frame)
  Eigen::Matrix3f cov = Eigen::Matrix3f::Identity();  // Sigma_{W p'} (Eq.28 block)
};

// Aligned vector of stored observations (StoredPoint carries fixed-size Eigen
// members; the aligned_allocator keeps each element correctly aligned).
using StoredPointVector = std::vector<StoredPoint, Eigen::aligned_allocator<StoredPoint>>;

struct VoxelBlock
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StoredPointVector pts;                     // accumulated observations P
  PlaneData plane;
  bool plane_dirty = true;                   // needs (re)fit
};

// Correspondence record passed from search to the residual builders.
struct PlaneCorr
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vec3 p_lidar;          // L p   (query point, LiDAR frame)
  Vec3 n;                // W n
  Vec3 q;                // W q
  Eigen::Matrix<double, 6, 6> cov_nq;
};

struct PointCorr
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vec3 p_lidar;          // L p
  Vec3 p_target;         // W p'
  Eigen::Matrix3f cov_target;
  int  N_vox_acc = 0;    // voxels accessed during the pruned search (Eq.29)
  int  N_pt_eval = 0;    // points evaluated during the pruned search (Eq.29)
};

class VoxelMap
{
public:
  explicit VoxelMap(const Params & p) : prm_(p) {}

  // Insert motion-compensated, world-frame points (Eq.36 output) with their
  // world covariances; (re)fits planes lazily.
  void insert(const StoredPointVector & world_pts);

  // Point-to-plane search over the candidate voxels with 3-sigma gating
  // (Sec.V-B-2). Returns true and fills `out` if a plane correspondence is
  // accepted.
  bool searchPlane(const Vec3 & p_world, const Vec3 & p_lidar, PlaneCorr & out) const;

  // Algorithm 2: point-to-point nearest neighbour with two-stage voxel pruning.
  // Returns true and fills `out` (incl. the Eq.29 discretization-error counters)
  // if a target point within tau_closest is found.
  bool searchPoint(const Vec3 & p_world, const Vec3 & p_lidar, PointCorr & out) const;

  std::size_t numVoxels() const { return map_.size(); }

  // Flatten every stored world-frame point into `out` (x/y/z populated, used to
  // dump the accumulated map to a .pcd file on shutdown).
  void collectWorldPoints(PointCloud & out) const;

private:
  // Nearest stored point to `p_world` inside one voxel; increments eval counter.
  // Returns squared distance (inf if voxel missing/empty).
  double nnInVoxel(const VoxelKey & k, const Vec3 & p_world,
                   Vec3 & best, Eigen::Matrix3f & best_cov, int & n_eval) const;

  void fitPlane(VoxelBlock & b) const;

  Params prm_;
  VoxelHashMap<VoxelBlock> map_;
};

// Range-bearing point covariance in the LiDAR frame (VoxelMap [11] model).
// Header-inline (used by both the map and the residual builders).
inline Mat3 pointCovLidar(const Vec3 & p, double sigma_r, double sigma_b)
{
  const double r = p.norm();
  if (r < 1e-6) return (sigma_r * sigma_r) * Mat3::Identity();
  const Vec3 w = p / r;                                   // bearing direction
  // Build an orthonormal basis {w, b1, b2}.
  Vec3 ref = (std::abs(w.x()) < 0.9) ? Vec3::UnitX() : Vec3::UnitY();
  const Vec3 b1 = (ref - ref.dot(w) * w).normalized();
  const Vec3 b2 = w.cross(b1);
  const double tang = r * r * sigma_b * sigma_b;          // lateral (range*angle)
  return (sigma_r * sigma_r) * (w * w.transpose())
       + tang * (b1 * b1.transpose() + b2 * b2.transpose());
}

// ===========================================================================
//  (V) Hybrid-metric ESIKF
// ===========================================================================
class Esikf
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit Esikf(const Params & p) : prm_(p)
  {
    P_.setIdentity();
    P_ *= 1e-3;
    P_.block<3, 3>(IDX_G, IDX_G) = 1e-4 * Mat3::Identity();  // gravity weakly known
    buildQ();
  }

  const State & state() const { return x_; }
  State & mutableState() { return x_; }
  const Mat18 & cov() const { return P_; }
  void setState(const State & s) { x_ = s; }
  void setCov(const Mat18 & P) { P_ = P; }

  // --- Prediction (forward propagation; FAST-LIO2 [6], state model Eq.9-10) ---
  // Propagates the nominal state and covariance across one IMU interval.
  void predict(const ImuData & imu, double dt);

  // --- Hybrid-metric iterated update (Algorithm 3; Eq.31-35) ---
  // `V_t` are the (downsampled) scan points in the LiDAR frame; `map` is G.
  // Returns the number of iterations actually run.
  int update(const PointCloud & V_t, const VoxelMap & map);

  // Per-correspondence residual / Jacobian / covariance builders.
  // Point-to-plane (Eq.13-19): scalar residual.
  void planeTerms(const PlaneCorr & c, const State & xl,
                  double & z, Eigen::Matrix<double, 1, STATE_DIM> & H, double & R) const;
  // Point-to-point (Eq.20-30): L2-norm scalar residual; returns false if the
  // residual is below eps_po and must be skipped.
  bool pointTerms(const PointCorr & c, const State & xl,
                  double & z, Eigen::Matrix<double, 1, STATE_DIM> & H, double & R) const;

private:
  void buildQ();

  Params prm_;
  State  x_;
  Mat18  P_ = Mat18::Identity();
  Mat12  Q_ = Mat12::Identity();
};

}  // namespace genz_lio

#ifndef GENZ_LIO_HEADER_ONLY

// ===========================================================================
//  ROS 2 node
// ===========================================================================
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>

namespace genz_lio
{

// GenZ-LIO node.
//   Subscribes:  ~/points  (sensor_msgs/PointCloud2)  -- LiDAR
//                ~/imu     (sensor_msgs/Imu)          -- IMU
//   Publishes :  ~/odometry, ~/path, ~/cloud_registered  and the odom->base TF.
class GenZLioNode : public rclcpp::Node
{
public:
  GenZLioNode();
  // Flushes the accumulated voxel map to `pcd_save_path` when pcd_save_en=true.
  ~GenZLioNode() override;

private:
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // Pull one LiDAR sweep plus all IMU samples spanning it, then run the
  // predict/deskew/update pipeline. Returns false if not enough data yet.
  bool tryProcess();

  // Convert a PointCloud2 (auto-detecting common time/intensity field names)
  // into the internal cloud; `scan_t0` is the header stamp [s].
  bool parseCloud(const sensor_msgs::msg::PointCloud2 & msg, PointCloud & out,
                  double & scan_t0, double & scan_dur);

  // One integrated IMU/body pose in the world frame at absolute time `t`.
  struct PoseStamp { double t; Mat3 R; Vec3 p; };

  // FAST-LIO2-style backward propagation: undistort `raw` (LiDAR frame, per
  // point `time` relative to scan start [s]) to the scan-end pose, using the
  // world-frame body trajectory `traj` recorded during forward propagation.
  // Output stays in the LiDAR frame (referenced to scan end).
  PointCloud deskew(const PointCloud & raw, const std::vector<PoseStamp> & traj,
                    double scan_t0, double scan_dur); //double scan_t0); //CRITICAL FIX: ADDED scan_dur HERE 

  void initializeImu(const ImuData & imu);
  void publish(double stamp);

  // Publish the current sweep as a registered PointCloud2: in the world frame
  // (~/cloud_registered, if scan_publish_en) and/or in the body frame
  // (~/cloud_registered_body, if scan_bodyframe_pub_en). `lidar_pts` are the
  // deskewed points in the LiDAR frame; `s` is the pose used to register them.
  void publishScan(double stamp, const PointCloud & lidar_pts, const State & s);

  // Pack an internal cloud into a sensor_msgs/PointCloud2 (x,y,z,intensity).
  sensor_msgs::msg::PointCloud2 toCloud2(
    const PointCloud & pts, const std::string & frame, const rclcpp::Time & stamp) const;

  // -- params / sub-modules --
  Params params_;
  std::unique_ptr<AdaptiveVoxelizer> voxelizer_;
  std::unique_ptr<VoxelMap> map_;
  std::unique_ptr<Esikf> ekf_;

  // -- I/O --
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_body_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // -- buffers (guarded by mutex) --
  std::mutex buf_mutex_;
  std::deque<ImuData> imu_buf_;
  std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_buf_;

  // -- bookkeeping --
  std::string world_frame_ = "odom";
  std::string body_frame_  = "base_link";
  std::string imu_topic_;                 // raw IMU topic (selects axis-swap)
  std::string points_topic_;              // raw Point Cloud topic (selects axis-swap)
  bool   imu_inited_ = false;
  bool   first_scan_ = true;
  double point_time_scale_ = 1.0;
  double lidar_time_offset_ = 0.0; // NEW ADDITION
  double last_imu_t_ = -1.0;
  double last_scan_end_ = -1.0;
  Vec3   last_gyr_ = Vec3::Zero();         // most recent debiasable body rate
  std::vector<ImuData> init_samples_;
  nav_msgs::msg::Path path_;

  // -- output options (from YAML) --
  int    deskew_mode_;
  bool   publish_tf_ = true;
  bool   scan_publish_en_ = true;
  bool   scan_bodyframe_pub_en_ = false;
  bool   pcd_save_en_ = false;
  std::string pcd_save_path_ = "/tmp/genz_lio_map.pcd";
  double lidar_topic_hz_ = 10.0;
  double imu_topic_hz_ = 200.0;
};

}  // namespace genz_lio
#endif  // GENZ_LIO_HEADER_ONLY

#endif  // GENZ_LIO__GENZ_LIO_HPP_
