// ===========================================================================
//  genz_lio.cpp  -- full implementation of the GenZ-LIO estimator + ROS 2 node.
//  Equation / algorithm numbers refer to arXiv:2603.16273v2.
// ===========================================================================
#include "genz_lio/genz_lio.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>

#ifndef GENZ_LIO_HEADER_ONLY
#include <fstream>
#include <string>

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#endif  // GENZ_LIO_HEADER_ONLY

namespace genz_lio
{

// ===========================================================================
//  Voxelize(., d)  -- uniform voxel-grid downsample to one centroid per voxel.
// ===========================================================================
PointCloud voxelDownsample(const PointCloud & in, double leaf)
{
  if (leaf <= 0.0 || in.empty()) return in;

  struct Accum {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 sum = Vec3::Zero(); double inten = 0; double time = 0; int n = 0;
  };
  VoxelHashMap<Accum> grid;
  grid.reserve(in.size());

  for (const auto & p : in) {
    const VoxelKey k = keyOf(p.vec(), leaf);
    Accum & a = grid[k];
    a.sum += p.vec();
    a.inten += p.intensity;
    a.time += p.time;
    ++a.n;
  }

  PointCloud out;
  out.reserve(grid.size());
  for (const auto & kv : grid) {
    const Accum & a = kv.second;
    const double inv = 1.0 / static_cast<double>(a.n);
    PointXYZIT q;
    q.x = a.sum.x() * inv;
    q.y = a.sum.y() * inv;
    q.z = a.sum.z() * inv;
    q.intensity = static_cast<float>(a.inten * inv);
    q.time = a.time * inv;
    out.push_back(q);
  }
  return out;
}

// ===========================================================================
//  (IV) AdaptiveVoxelizer::process  -- Algorithm 1
// ===========================================================================
AdaptiveVoxelizer::Result
AdaptiveVoxelizer::process(const PointCloud & deskewed, double dt_scan)
{
  Result r;

  // Alg.1 (2-3): temporary voxelization at the previous voxel size d_{t-1}.
  const PointCloud V_temp = voxelDownsample(deskewed, d_prev_);
  r.N_temp = static_cast<int>(V_temp.size());

  if (V_temp.empty()) {
    // Nothing to control on; keep previous size, emit empty clouds.
    r.d_t = d_prev_;
    return r;
  }

  // Alg.1 (4-8): scale indicator m_t = median range, smoothed to m_bar.
  std::vector<double> ranges;
  ranges.reserve(V_temp.size());
  for (const auto & p : V_temp) ranges.push_back(p.vec().norm());
  const std::size_t mid = ranges.size() / 2;
  std::nth_element(ranges.begin(), ranges.begin() + mid, ranges.end());
  const double m_t = ranges[mid];
  r.m_bar = pushMedian(m_t);

  // Alg.1 (9-28): setpoint, tracking error, gain scheduling, PD voxel update.
  const Control c = computeControl(r.m_bar, r.N_temp, dt_scan);
  r.d_t = c.d_t;
  r.N_desired = c.N_desired;

  // Commit controller memory (d_{t-1} <- d_t, e_{t-1} <- e_t; Alg.1 lines 28-29).
  d_prev_ = c.d_t;
  e_prev_ = c.e;

  // Alg.1 (30-32): bi-resolution voxelization.
  r.V_merge_t = voxelDownsample(deskewed, c.d_t * 0.5);  // d_t/2  -> map
  r.V_t       = voxelDownsample(r.V_merge_t, c.d_t);      // d_t    -> state update
  return r;
}

// ===========================================================================
//  (V-B) VoxelMap
// ===========================================================================
void VoxelMap::fitPlane(VoxelBlock & b) const
{
  b.plane.valid = false;
  const int N = static_cast<int>(b.pts.size());
  if (N < prm_.plane_min_pts) return;

  // Single-pass Welford accumulation of the centroid q and the (population)
  // covariance A: for each new point p, delta = p - mean; mean += delta/k;
  // M2 += delta * (p - mean_updated)^T. This computes the mean and the second
  // moment in one pass (numerically stabler than the naive two-pass form).
  Vec3 q = Vec3::Zero();          // running mean
  Mat3 M2 = Mat3::Zero();         // running sum of squared deviations (outer prod)
  int k = 0;
  for (const auto & sp : b.pts) {
    ++k;
    const Vec3 delta = sp.p - q;
    q += delta / static_cast<double>(k);
    const Vec3 delta2 = sp.p - q;   // deviation from the updated mean
    M2.noalias() += delta * delta2.transpose();
  }
  const Mat3 A = M2 / static_cast<double>(N);   // population covariance (matches /N)

  Eigen::SelfAdjointEigenSolver<Mat3> es(A);
  const Vec3 ev = es.eigenvalues();            // ascending: ev[0] <= ev[1] <= ev[2]
  const Mat3 U  = es.eigenvectors();
  // Planarity: smallest eigenvalue much smaller than the middle one.
  if (ev[0] > prm_.plane_ratio * ev[1]) return;

  const Vec3 n = U.col(0);                      // normal = smallest-eigenvalue dir
  b.plane.valid    = true;
  b.plane.normal   = n;
  b.plane.centroid = q;

  // Plane-parameter covariance Sigma_{n,q} (VoxelMap [11]; Eq.19 block).
  // dn/dp_i = (1/N) sum_{m!=0} u_m/(l_0 - l_m) [ (u_m.(p_i-q)) u_0^T
  //                                            + ((p_i-q).u_0) u_m^T ]
  Mat3 Sn = Mat3::Zero();      // cov(n)
  Mat3 Snq = Mat3::Zero();     // cov(n, q) (before 1/N already folded below)
  Mat3 Sq = Mat3::Zero();      // cov(q)
  const double invN = 1.0 / static_cast<double>(N);
  for (const auto & sp : b.pts) {
    const Vec3 dpi = sp.p - q;
    Mat3 F = Mat3::Zero();     // dn/dp_i
    for (int m = 1; m < 3; ++m) {
      const Vec3 um = U.col(m);
      const double denom = ev[0] - ev[m];       // negative (l_0 smallest)
      if (std::abs(denom) < 1e-12) continue;
      const double coeff = 1.0 / denom;
      // u_m * [ (u_m . dpi) u_0^T + (dpi . u_0) u_m^T ]   (3x1 * 1x3 -> 3x3)
      const Eigen::RowVector3d row =
        um.dot(dpi) * n.transpose() + dpi.dot(n) * um.transpose();
      F += coeff * (um * row);
    }
    F *= invN;
    const Mat3 Cp = sp.cov.cast<double>();      // Sigma_{p_i} (world)
    Sn  += F * Cp * F.transpose();
    Snq += F * Cp;                              // (.)*(dq/dp_i)^T, dq/dp_i = (1/N) I
    Sq  += Cp;
  }
  Snq *= invN;          // cov(n,q) = sum F Cp (1/N)
  Sq  *= invN * invN;   // cov(q)   = (1/N^2) sum Cp

  b.plane.cov_nq.setZero();
  b.plane.cov_nq.block<3, 3>(0, 0) = Sn;
  b.plane.cov_nq.block<3, 3>(0, 3) = Snq;
  b.plane.cov_nq.block<3, 3>(3, 0) = Snq.transpose();
  b.plane.cov_nq.block<3, 3>(3, 3) = Sq;
  b.plane_dirty = false;
}

void VoxelMap::insert(const StoredPointVector & world_pts)
{
  for (const auto & sp : world_pts) {
    const VoxelKey k = keyOf(sp.p, prm_.d_root);
    VoxelBlock & b = map_[k];
    if (static_cast<int>(b.pts.size()) < prm_.max_pts_voxel) {
      b.pts.push_back(sp);
      b.plane_dirty = true;
    }
  }
  // (Re)fit planes for voxels that changed.
  for (auto & kv : map_) {
    if (kv.second.plane_dirty) fitPlane(kv.second);
  }
}

void VoxelMap::collectWorldPoints(PointCloud & out) const
{
  std::size_t total = 0;
  for (const auto & kv : map_) total += kv.second.pts.size();
  out.clear();
  out.reserve(total);
  for (const auto & kv : map_) {
    for (const auto & sp : kv.second.pts) {
      PointXYZIT p;
      p.x = sp.p.x();
      p.y = sp.p.y();
      p.z = sp.p.z();
      out.push_back(p);
    }
  }
}

double VoxelMap::nnInVoxel(const VoxelKey & k, const Vec3 & p_world,
                           Vec3 & best, Eigen::Matrix3f & best_cov, int & n_eval) const
{
  auto it = map_.find(k);
  if (it == map_.end()) return std::numeric_limits<double>::infinity();
  double best_d2 = std::numeric_limits<double>::infinity();
  for (const auto & sp : it->second.pts) {
    const double d2 = (sp.p - p_world).squaredNorm();
    ++n_eval;
    if (d2 < best_d2) { best_d2 = d2; best = sp.p; best_cov = sp.cov; }
  }
  return best_d2;
}

bool VoxelMap::searchPlane(const Vec3 & p_world, const Vec3 & p_lidar,
                           PlaneCorr & out) const
{
  const VoxelKey root = keyOf(p_world, prm_.d_root);
  const Eigen::Vector3i s = voxutil::regionSign(p_world, root, prm_.d_root);
  std::vector<Eigen::Vector3i> offs = voxutil::candidateOffsets(s);

  // Candidate voxels = root + selected neighbours (Fig.5).
  std::vector<VoxelKey> cand;
  cand.reserve(offs.size() + 1);
  cand.push_back(root);
  for (const auto & o : offs) cand.push_back(voxutil::shift(root, o));

  bool found = false;
  double best_score = std::numeric_limits<double>::infinity();  // |r|^2 / var
  const double sigma_r2 = prm_.lidar_range_std * prm_.lidar_range_std;
  const double g2 = prm_.gate_sigma * prm_.gate_sigma;

  for (const auto & k : cand) {
    auto it = map_.find(k);
    if (it == map_.end() || !it->second.plane.valid) continue;
    const PlaneData & pl = it->second.plane;

    const double r = pl.normal.dot(p_world - pl.centroid);  // point-to-plane dist
    // Gating variance from the plane covariance (Eq.18 structure) + a point-noise
    // floor (the exact per-state R_pl is recomputed in Esikf::planeTerms).
    Eigen::Matrix<double, 6, 1> gvec;
    gvec << (p_world - pl.centroid), -pl.normal;
    double var = (gvec.transpose() * pl.cov_nq * gvec)(0, 0) + sigma_r2;
    var = std::max(var, 1e-9);

    if (r * r <= g2 * var) {              // 3-sigma gate (Sec.V-B-2)
      const double score = r * r / var;   // smaller -> higher matching prob.
      if (score < best_score) {
        best_score = score;
        out.p_lidar = p_lidar;
        out.n       = pl.normal;
        out.q       = pl.centroid;
        out.cov_nq  = pl.cov_nq;
        found = true;
      }
    }
  }
  return found;
}

bool VoxelMap::searchPoint(const Vec3 & p_world, const Vec3 & p_lidar,
                           PointCorr & out) const
{
  // ---- Algorithm 2: point-to-point NN with two-stage voxel pruning ----
  const VoxelKey root = keyOf(p_world, prm_.d_root);
  const Eigen::Vector3i s = voxutil::regionSign(p_world, root, prm_.d_root);
  const std::vector<Eigen::Vector3i> offs = voxutil::candidateOffsets(s);

  double d_closest = prm_.tau_closest;          // line 2: init with outlier thresh
  Vec3 best = Vec3::Zero();
  Eigen::Matrix3f best_cov = Eigen::Matrix3f::Identity();
  int N_vox_acc = 0, N_pt_eval = 0;

  // line 3-5: search the root voxel.
  {
    int ne = 0;
    const double d2 = nnInVoxel(root, p_world, best, best_cov, ne);
    N_vox_acc += 1;                              // root is always accessed
    N_pt_eval += ne;
    if (std::isfinite(d2) && std::sqrt(d2) < d_closest) d_closest = std::sqrt(d2);
  }

  // line 6-11: coarse-to-fine pruned traversal of neighbour voxels.
  for (const auto & o : offs) {
    const double d_nbr = voxutil::computeDistanceToVoxel(p_world, root, o, prm_.d_root);
    if (d_nbr < d_closest) {                      // line 8 prune
      const VoxelKey nk = voxutil::shift(root, o);
      Vec3 tb; Eigen::Matrix3f tcov; int ne = 0;
      const double d2 = nnInVoxel(nk, p_world, tb, tcov, ne);
      N_vox_acc += 1;
      N_pt_eval += ne;
      const double d = std::sqrt(d2);
      if (std::isfinite(d2) && d < d_closest) {   // line 10
        d_closest = d; best = tb; best_cov = tcov;
      }
    }
  }

  // line 12-15: accept only if within tau_closest.
  if (d_closest < prm_.tau_closest) {
    out.p_lidar    = p_lidar;
    out.p_target   = best;
    out.cov_target = best_cov;
    out.N_vox_acc  = std::max(N_vox_acc, 1);
    out.N_pt_eval  = std::max(N_pt_eval, 1);
    return true;
  }
  return false;
}

// ===========================================================================
//  (V) Esikf
// ===========================================================================
void Esikf::buildQ()
{
  Q_.setZero();
  Q_.block<3, 3>(0, 0) = (prm_.gyr_noise   * prm_.gyr_noise)   * Mat3::Identity(); // n_g
  Q_.block<3, 3>(3, 3) = (prm_.acc_noise   * prm_.acc_noise)   * Mat3::Identity(); // n_a
  Q_.block<3, 3>(6, 6) = (prm_.gyr_bias_rw * prm_.gyr_bias_rw) * Mat3::Identity(); // n_bg
  Q_.block<3, 3>(9, 9) = (prm_.acc_bias_rw * prm_.acc_bias_rw) * Mat3::Identity(); // n_ba
}

void Esikf::predict(const ImuData & imu, double dt)
{
  if (dt <= 0.0) return;

  // Jacobians are evaluated at the pre-propagation state.
  const Mat3 R = x_.rot;
  const Vec3 a = imu.acc - x_.ba;               // body specific force, debiased
  const Vec3 w = imu.gyr - x_.bg;               // body angular rate, debiased
  const Vec3 acc_world = R * a + x_.grav;       // Eq.10 row 3

  // ---- Nominal propagation (Eq.9-10, discretized over dt) ----
  x_.pos += x_.vel * dt + 0.5 * acc_world * dt * dt;  // Eq.10 row 1
  x_.vel += acc_world * dt;                            // Eq.10 row 3
  x_.rot  = R * so3::Exp(w * dt);                       // Eq.10 row 2
  // biases (rows 4-5) and gravity (row 6) are constant under zero process noise.

////////////////////////////////////////////
// CRITICAL FIX: Re-orthogonalize to prevent floating point explosion
  Eigen::Quaterniond q_pred(x_.rot);
  q_pred.normalize();
  x_.rot = q_pred.toRotationMatrix();
//////////////////////////////////////////////////

  // ---- Error-state covariance propagation (FAST-LIO2 [6]) ----
  Mat18 Fx = Mat18::Identity();
  Fx.block<3, 3>(IDX_P, IDX_V)  = Mat3::Identity() * dt;
  Fx.block<3, 3>(IDX_P, IDX_R)  = -0.5 * R * so3::hat(a) * dt * dt;
  Fx.block<3, 3>(IDX_P, IDX_BA) = -0.5 * R * dt * dt;
  Fx.block<3, 3>(IDX_P, IDX_G)  = 0.5 * Mat3::Identity() * dt * dt;
  Fx.block<3, 3>(IDX_R, IDX_R)  = so3::Exp(w * dt).transpose();
  Fx.block<3, 3>(IDX_R, IDX_BG) = -so3::rightJacobian(w * dt) * dt;
  Fx.block<3, 3>(IDX_V, IDX_R)  = -R * so3::hat(a) * dt;
  Fx.block<3, 3>(IDX_V, IDX_BA) = -R * dt;
  Fx.block<3, 3>(IDX_V, IDX_G)  = Mat3::Identity() * dt;

  // Continuous noise-input matrix G_c (no dt); discretized via *dt below.
  Mat18x12 Fw = Mat18x12::Zero();
  Fw.block<3, 3>(IDX_R, 0)  = -so3::rightJacobian(w * dt);  // n_g -> rotation
  Fw.block<3, 3>(IDX_V, 3)  = -R;                            // n_a -> velocity
  Fw.block<3, 3>(IDX_BG, 6) = Mat3::Identity();              // n_bg -> gyro bias
  Fw.block<3, 3>(IDX_BA, 9) = Mat3::Identity();              // n_ba -> accel bias

  P_ = Fx * P_ * Fx.transpose() + (Fw * Q_ * Fw.transpose()) * dt;
  P_ = 0.5 * (P_ + P_.transpose());  // keep symmetric
}

void Esikf::planeTerms(const PlaneCorr & c, const State & xl,
                       double & z, Eigen::Matrix<double, 1, STATE_DIM> & H,
                       double & R) const
{
  const Mat3 R_wl = xl.rot * prm_.R_IL;                       // W R_I * I R_L
  const Vec3 p_I  = prm_.R_IL * c.p_lidar + prm_.t_IL;        // point in IMU frame
  const Vec3 p_w  = xl.rot * p_I + xl.pos;                    // W T_I I T_L L p

  // Residual z_pl = W n^T ( W T_I I T_L L p - W q )   (Eq.16)
  z = c.n.dot(p_w - c.q);

  // Jacobian wrt error state (Eq.16). The paper writes the rotation block as
  //   -n^T R_I R_IL [L p]_x ; for a non-zero extrinsic translation the exact
  // body-frame form is -n^T R_I [p_I]_x , which reduces to the paper's
  // expression when I t_L = 0.
  H.setZero();
  H.block<1, 3>(0, IDX_P) = c.n.transpose();
  H.block<1, 3>(0, IDX_R) = -(c.n.transpose() * xl.rot * so3::hat(p_I));

  // Measurement-noise covariance R_pl = H_v Sigma H_v^T   (Eq.18-19).
  // H_v (1x9) = [ (p_w - q)^T , -n^T , n^T R_wl ]   over v_pl = [dw_n, dw_q, d_Lp]
  Eigen::Matrix<double, 1, 9> Hv;
  Hv.block<1, 3>(0, 0) = (p_w - c.q).transpose();
  Hv.block<1, 3>(0, 3) = -c.n.transpose();
  Hv.block<1, 3>(0, 6) = c.n.transpose() * R_wl;

  Eigen::Matrix<double, 9, 9> Sig = Eigen::Matrix<double, 9, 9>::Zero();
  Sig.block<6, 6>(0, 0) = c.cov_nq;                           // Sigma_{w_n, w_q}
  Sig.block<3, 3>(6, 6) =
    pointCovLidar(c.p_lidar, prm_.lidar_range_std, prm_.lidar_bearing_std); // Sigma_Lp

  R = (Hv * Sig * Hv.transpose())(0, 0);
  R = std::max(R, 1e-9);
}

bool Esikf::pointTerms(const PointCorr & c, const State & xl,
                       double & z, Eigen::Matrix<double, 1, STATE_DIM> & H,
                       double & R) const
{
  const Mat3 R_wl = xl.rot * prm_.R_IL;
  const Vec3 p_I  = prm_.R_IL * c.p_lidar + prm_.t_IL;
  const Vec3 p_w  = xl.rot * p_I + xl.pos;

  const Vec3 z_vec = p_w - c.p_target;                        // Eq.24 (3x1)
  const double norm = z_vec.norm();
  if (norm < prm_.eps_po) return false;                       // skip near-zero (Sec.V-D)
  z = norm;                                                   // z_po^norm = ||z_po||

  // Full point-to-point Jacobian (Eq.24): H_x,po = [ I3 , -R_I [p_I]_x , 0 ]
  Eigen::Matrix<double, 3, STATE_DIM> Hx = Eigen::Matrix<double, 3, STATE_DIM>::Zero();
  Hx.block<3, 3>(0, IDX_P) = Mat3::Identity();
  Hx.block<3, 3>(0, IDX_R) = -(xl.rot * so3::hat(p_I));

  const Vec3 u = z_vec / norm;                                // unit residual dir
  H = u.transpose() * Hx;                                     // normalized (Eq.26)

  // R_po = H_v Sigma H_v^T (Eq.27); H_v,po = [ R_wl , -I3 ] (Eq.24);
  // Sigma = blockdiag(Sigma_Lp, Sigma_{W p'}) (Eq.28).
  Eigen::Matrix<double, 3, 6> Hv;
  Hv.block<3, 3>(0, 0) = R_wl;
  Hv.block<3, 3>(0, 3) = -Mat3::Identity();
  Eigen::Matrix<double, 6, 6> Sig = Eigen::Matrix<double, 6, 6>::Zero();
  Sig.block<3, 3>(0, 0) =
    pointCovLidar(c.p_lidar, prm_.lidar_range_std, prm_.lidar_bearing_std);
  Sig.block<3, 3>(3, 3) = c.cov_target.cast<double>();
  const Mat3 R_po = Hv * Sig * Hv.transpose();                // Eq.27

  const double R_norm = (u.transpose() * R_po * u);           // Eq.26
  // Discretization variance from the pruned search (Eq.29).
  const double R_disc =
    (static_cast<double>(c.N_vox_acc) * prm_.d_root * prm_.d_root) /
    static_cast<double>(std::max(c.N_pt_eval, 1));
  R = prm_.lambda_po * (R_norm + R_disc);                     // Eq.30
  R = std::max(R, 1e-9);
  return true;
}

int Esikf::update(const PointCloud & V_t, const VoxelMap & map)
{
  const State x_prior = x_;                 // x_hat  (propagated prior)
  State x_iter = x_;                         // x_hat^0
  const Mat18 P_prior_inv = P_.inverse();    // P_hat^{-1}

  Mat18 S = Mat18::Identity();
  int iter = 0;
  for (; iter < prm_.max_iter; ++iter) {
    Mat18 A = Mat18::Zero();                 // sum H^T R^{-1} H
    Vec18 b = Vec18::Zero();                 // sum H^T R^{-1} z

    // -- Algorithm 3 line 4-6: correspondences + residual/Jacobian/cov terms --
    for (const auto & pt : V_t) {
      const Vec3 p_lidar = pt.vec();
      const Vec3 p_I = prm_.R_IL * p_lidar + prm_.t_IL;
      const Vec3 p_w = x_iter.rot * p_I + x_iter.pos;

      Eigen::Matrix<double, 1, STATE_DIM> H;
      double z = 0.0, R = 1.0;

      PlaneCorr pc;
      if (map.searchPlane(p_w, p_lidar, pc)) {          // point-to-plane (C_pl)
        planeTerms(pc, x_iter, z, H, R);
      } else {
        PointCorr ptc;
        if (map.searchPoint(p_w, p_lidar, ptc)) {       // fallback point-to-point (C_po)
          if (!pointTerms(ptc, x_iter, z, H, R)) continue;
        } else {
          continue;                                     // noise-induced outlier
        }
      }
      const double inv_r = 1.0 / R;
      A.noalias() += H.transpose() * inv_r * H;
      b.noalias() += H.transpose() * (inv_r * z);
    }

    // -- line 8-10: reparametrized prior, Kalman gain, increment (Eq.33) --
    const Mat18 J = reparamJacobian(x_iter, x_prior);
    const Vec18 Delta = x_iter.boxminus(x_prior);        // x_hat^l [-] x_hat
    const Mat18 Pl_inv = J.transpose() * P_prior_inv * J;  // (P^l)^{-1}
    S = A + Pl_inv;

    // dx = -S^{-1} ( b + J^T P_hat^{-1} Delta )  [equiv. to Eq.33 update term]
    const Vec18 rhs = b + J.transpose() * (P_prior_inv * Delta);
    const Vec18 dx = -S.ldlt().solve(rhs);

    x_iter = x_iter.boxplus(dx);

    if (dx.norm() < prm_.tau_converge) { ++iter; break; }  // ||x^{l+1} [-] x^l||
  }

  // -- line 12: commit state and posterior covariance P = S^{-1} (= (I-KH)P^l) --
  x_ = x_iter;
  P_ = S.inverse();
  P_ = 0.5 * (P_ + P_.transpose());
  
///////////////////////////////////////
  // CRITICAL FIX: Re-orthogonalize the posterior rotation matrix
  Eigen::Quaterniond q_norm(x_.rot);
  q_norm.normalize();
  x_.rot = q_norm.toRotationMatrix();
///////////////////////////////////
  
  return iter;
}

#ifndef GENZ_LIO_HEADER_ONLY
// ===========================================================================
//  ROS 2 node
// ===========================================================================

namespace
{
inline double stampSec(const builtin_interfaces::msg::Time & t)
{
  return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
}

// Read a per-point scalar field of static type T (via a typed const iterator,
// no raw byte arithmetic) into `out[i]` as double, for every point in the cloud.
template <typename T>
inline void fillField(const sensor_msgs::msg::PointCloud2 & msg,
                      const std::string & field, std::vector<double> & out)
{
  sensor_msgs::PointCloud2ConstIterator<T> it(msg, field);
  for (std::size_t i = 0; i < out.size(); ++i, ++it) {
    out[i] = static_cast<double>(*it);
  }
}

// Dispatch a runtime PointField datatype to the correctly typed iterator read.
inline void extractField(const sensor_msgs::msg::PointCloud2 & msg,
                         const std::string & field, uint8_t datatype,
                         std::vector<double> & out)
{
  switch (datatype) {
    case sensor_msgs::msg::PointField::FLOAT32: fillField<float>(msg, field, out);    break;
    case sensor_msgs::msg::PointField::FLOAT64: fillField<double>(msg, field, out);   break;
    case sensor_msgs::msg::PointField::UINT32:  fillField<uint32_t>(msg, field, out); break;
    case sensor_msgs::msg::PointField::INT32:   fillField<int32_t>(msg, field, out);  break;
    case sensor_msgs::msg::PointField::UINT16:  fillField<uint16_t>(msg, field, out); break;
    case sensor_msgs::msg::PointField::INT16:   fillField<int16_t>(msg, field, out);  break;
    case sensor_msgs::msg::PointField::UINT8:   fillField<uint8_t>(msg, field, out);  break;
    case sensor_msgs::msg::PointField::INT8:    fillField<int8_t>(msg, field, out);   break;
    default:                                    fillField<float>(msg, field, out);    break;
  }
}
}  // namespace

GenZLioNode::GenZLioNode() : rclcpp::Node("genz_lio_node")
{
  // ---- Topics / frames ----
  const std::string points_topic = declare_parameter<std::string>("points_topic", "/bf_lidar/point_cloud_out");
  points_topic_ = points_topic;         // retained: selects the Point Cloud axis-swap
  const std::string imu_topic    = declare_parameter<std::string>("imu_topic", "/mavros/imu/data");
  imu_topic_   = imu_topic;             // retained: selects the IMU axis-swap
  world_frame_ = declare_parameter<std::string>("world_frame", "odom");
  body_frame_  = declare_parameter<std::string>("body_frame", "base_link");
  point_time_scale_ = declare_parameter<double>("point_time_scale", 1.0);

  // ---- Output options ----
  deskew_mode_           = declare_parameter<int>("deskew_mode", 1);
  publish_tf_            = declare_parameter<bool>("publish_tf", publish_tf_);
  scan_publish_en_       = declare_parameter<bool>("scan_publish_en", scan_publish_en_);
  scan_bodyframe_pub_en_ = declare_parameter<bool>("scan_bodyframe_pub_en", scan_bodyframe_pub_en_);
  pcd_save_en_           = declare_parameter<bool>("pcd_save_en", pcd_save_en_);
  pcd_save_path_         = declare_parameter<std::string>("pcd_save_path", pcd_save_path_);
  lidar_topic_hz_        = declare_parameter<double>("lidar_topic_hz", lidar_topic_hz_);
  imu_topic_hz_          = declare_parameter<double>("imu_topic_hz", imu_topic_hz_);

  // ---- Scale-aware adaptive voxelization (Sec.IV) ----
  params_.d0       = declare_parameter<double>("d0", params_.d0);
  params_.d_min    = declare_parameter<double>("d_min", params_.d_min);
  params_.d_max    = declare_parameter<double>("d_max", params_.d_max);
  params_.N_min    = declare_parameter<int>("N_min", params_.N_min);
  params_.N_max    = declare_parameter<int>("N_max", params_.N_max);
  params_.tau_m    = declare_parameter<double>("tau_m", params_.tau_m);
  params_.p_exp    = declare_parameter<double>("p_exp", params_.p_exp);
  params_.N_w      = declare_parameter<int>("N_w", params_.N_w);
  params_.lambda_p = declare_parameter<double>("lambda_p", params_.lambda_p);
  params_.lambda_d = declare_parameter<double>("lambda_d", params_.lambda_d);
  params_.Kp_min   = declare_parameter<double>("Kp_min", params_.Kp_min);
  params_.Kp_max   = declare_parameter<double>("Kp_max", params_.Kp_max);
  params_.Kd_min   = declare_parameter<double>("Kd_min", params_.Kd_min);
  params_.Kd_max   = declare_parameter<double>("Kd_max", params_.Kd_max);

  // ---- Voxel-pruned search / map (Sec.V-B) ----
  params_.d_root        = declare_parameter<double>("d_root", params_.d_root);
  params_.tau_closest   = declare_parameter<double>("tau_closest", params_.tau_closest);
  params_.plane_min_pts = declare_parameter<int>("plane_min_pts", params_.plane_min_pts);
  params_.plane_ratio   = declare_parameter<double>("plane_ratio", params_.plane_ratio);
  params_.max_pts_voxel = declare_parameter<int>("max_pts_voxel", params_.max_pts_voxel);
  params_.gate_sigma    = declare_parameter<double>("gate_sigma", params_.gate_sigma);

  // ---- Hybrid-metric update (Sec.V) ----
  params_.eps_po       = declare_parameter<double>("eps_po", params_.eps_po);
  params_.lambda_po    = declare_parameter<double>("lambda_po", params_.lambda_po);
  params_.max_iter     = declare_parameter<int>("max_iter", params_.max_iter);
  params_.tau_converge = declare_parameter<double>("tau_converge", params_.tau_converge);

  // ---- Sensor noise ----
  params_.acc_noise        = declare_parameter<double>("acc_noise", params_.acc_noise);
  params_.gyr_noise        = declare_parameter<double>("gyr_noise", params_.gyr_noise);
  params_.acc_bias_rw      = declare_parameter<double>("acc_bias_rw", params_.acc_bias_rw);
  params_.gyr_bias_rw      = declare_parameter<double>("gyr_bias_rw", params_.gyr_bias_rw);
  params_.lidar_range_std  = declare_parameter<double>("lidar_range_std", params_.lidar_range_std);
  params_.lidar_bearing_std= declare_parameter<double>("lidar_bearing_std", params_.lidar_bearing_std);
  params_.imu_init_count   = declare_parameter<int>("imu_init_count", params_.imu_init_count);

  // ---- LiDAR geometry / range gating ----
  params_.lidar_min_range  = declare_parameter<double>("lidar_min_range", params_.lidar_min_range);
  params_.lidar_max_range  = declare_parameter<double>("lidar_max_range", params_.lidar_max_range);
  params_.lidar_fov_down   = declare_parameter<double>("lidar_fov_down", params_.lidar_fov_down);
  params_.lidar_fov_up     = declare_parameter<double>("lidar_fov_up", params_.lidar_fov_up);
  params_.lidar_scan_lines = declare_parameter<int>("lidar_scan_lines", params_.lidar_scan_lines);

  // ---- Extrinsics L (LiDAR) -> I (IMU/body) ----
  // extrinsic_rot:   3x3 rotation R_IL, row-major (rotates a LiDAR-frame vector
  //                  into the IMU/body frame).
  // extrinsic_trans: 3x1 translation t_IL (LiDAR origin expressed in the IMU
  //                  frame), i.e. I_T_L = (R_IL, t_IL) maps LiDAR -> IMU.
  const auto rot = declare_parameter<std::vector<double>>(
    "extrinsic_rot", {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  const auto trans = declare_parameter<std::vector<double>>(
    "extrinsic_trans", {0.0, 0.0, 0.0});
  if (rot.size() == 9) {
    params_.R_IL << rot[0], rot[1], rot[2],
                    rot[3], rot[4], rot[5],
                    rot[6], rot[7], rot[8];
  } else {
    RCLCPP_WARN(get_logger(),
      "extrinsic_rot must have 9 elements (got %zu); using identity.", rot.size());
  }
  if (trans.size() == 3) {
    params_.t_IL = Vec3(trans[0], trans[1], trans[2]);
  } else {
    RCLCPP_WARN(get_logger(),
      "extrinsic_trans must have 3 elements (got %zu); using zero.", trans.size());
  }

  // ---- Sub-modules ----
  voxelizer_ = std::make_unique<AdaptiveVoxelizer>(params_);
  map_       = std::make_unique<VoxelMap>(params_);
  ekf_       = std::make_unique<Esikf>(params_);

  // ---- I/O ----
  sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, rclcpp::SensorDataQoS(),
    std::bind(&GenZLioNode::imuCallback, this, std::placeholders::_1));
  sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    points_topic, rclcpp::SensorDataQoS(),
    std::bind(&GenZLioNode::cloudCallback, this, std::placeholders::_1));

  pub_odom_  = create_publisher<nav_msgs::msg::Odometry>("~/odometry", 10);
  pub_path_  = create_publisher<nav_msgs::msg::Path>("~/path", 10);
  // Best-effort (sensor-data) QoS for the high-volume registered clouds; RViz
  // subscribes best-effort to match. Use this profile for both cloud topics.
  const rclcpp::QoS cloud_qos = rclcpp::SensorDataQoS();
  pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/cloud_registered", cloud_qos);
  pub_cloud_body_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/cloud_registered_body", cloud_qos);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  path_.header.frame_id = world_frame_;

  RCLCPP_INFO(get_logger(),
    "GenZ-LIO up. LiDAR='%s'  IMU='%s'  (paper used MID-70/VLP-16/AVIA + VN-100/BMI088;"
    " here configured for Vortex).",
    points_topic.c_str(), imu_topic.c_str());
  RCLCPP_INFO(get_logger(),
    "config: range=[%.2f,%.2f] m  vert_fov=[%.1f,%.1f] deg  scan_lines=%d  "
    "rates~(lidar %.0f Hz, imu %.0f Hz)  publish_tf=%d scan_pub=%d body_pub=%d pcd_save=%d",
    params_.lidar_min_range, params_.lidar_max_range, params_.lidar_fov_down,
    params_.lidar_fov_up, params_.lidar_scan_lines, lidar_topic_hz_, imu_topic_hz_,
    static_cast<int>(publish_tf_), static_cast<int>(scan_publish_en_),
    static_cast<int>(scan_bodyframe_pub_en_), static_cast<int>(pcd_save_en_));
}

GenZLioNode::~GenZLioNode()
{
  // Task 1.4 -- flush the accumulated voxel map to a .pcd file on shutdown.
  if (!pcd_save_en_ || !map_) return;

  PointCloud pts;
  map_->collectWorldPoints(pts);
  if (pts.empty()) {
    RCLCPP_WARN(get_logger(), "pcd_save_en set but the map is empty; nothing saved.");
    return;
  }

  std::ofstream ofs(pcd_save_path_, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    RCLCPP_ERROR(get_logger(), "Could not open '%s' for writing; map not saved.",
                 pcd_save_path_.c_str());
    return;
  }

  // Minimal ASCII PCD (x y z intensity) -- readable by pcl_viewer / CloudCompare.
  ofs << "# .PCD v0.7 - Point Cloud Data file format\n"
      << "VERSION 0.7\n"
      << "FIELDS x y z intensity\n"
      << "SIZE 4 4 4 4\n"
      << "TYPE F F F F\n"
      << "COUNT 1 1 1 1\n"
      << "WIDTH " << pts.size() << "\n"
      << "HEIGHT 1\n"
      << "VIEWPOINT 0 0 0 1 0 0 0\n"
      << "POINTS " << pts.size() << "\n"
      << "DATA ascii\n";
  for (const auto & p : pts) {
    ofs << p.x << ' ' << p.y << ' ' << p.z << ' ' << p.intensity << '\n';
  }
  ofs.close();
  RCLCPP_INFO(get_logger(), "Saved voxel map (%zu points) to '%s'.",
              pts.size(), pcd_save_path_.c_str());
}

void GenZLioNode::initializeImu(const ImuData & imu)
{
  init_samples_.push_back(imu);
  if (static_cast<int>(init_samples_.size()) < params_.imu_init_count) return;

  // Static initialization: gravity from mean specific force, gyro bias from mean
  // angular rate (body assumed near-stationary at startup).
  Vec3 mean_acc = Vec3::Zero(), mean_gyr = Vec3::Zero();
  for (const auto & s : init_samples_) { mean_acc += s.acc; mean_gyr += s.gyr; }
  mean_acc /= static_cast<double>(init_samples_.size());
  mean_gyr /= static_cast<double>(init_samples_.size());

  State s0;
  s0.rot  = Mat3::Identity();
  s0.bg   = mean_gyr;
  s0.grav = -mean_acc.normalized() * 9.81;   // R*(a-ba)+g = 0 when static (Eq.10)
  ekf_->setState(s0);
  imu_inited_ = true;
  RCLCPP_INFO(get_logger(), "IMU initialized: |g|=%.3f  bg=[%.4f %.4f %.4f]",
              s0.grav.norm(), s0.bg.x(), s0.bg.y(), s0.bg.z());
}

void GenZLioNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  ImuData d;
  d.t = stampSec(msg->header.stamp);

  // Task 5.1 -- the incoming msg is a const SharedPtr and cannot be mutated, so
  // map raw axes into the standard ROS FLU body frame while constructing `d`.
  // The convention is selected from the IMU topic name (each rig publishes in a
  // different native frame: Blickfeld FRD, ArduPilot FRD-variant, camera optical).
  if (imu_topic_.find("bf_lidar") != std::string::npos) {
    d.acc = Vec3(-msg->linear_acceleration.y, msg->linear_acceleration.x, -msg->linear_acceleration.z);
    d.gyr = Vec3(-msg->angular_velocity.y, msg->angular_velocity.x, -msg->angular_velocity.z);
  } else if (imu_topic_.find("ap") != std::string::npos) {
    d.acc = Vec3(msg->linear_acceleration.x, -msg->linear_acceleration.y, -msg->linear_acceleration.z);
    d.gyr = Vec3(msg->angular_velocity.x, -msg->angular_velocity.y, -msg->angular_velocity.z);
  } else if (imu_topic_.find("camera") != std::string::npos) {
    d.acc = Vec3(msg->linear_acceleration.z, -msg->linear_acceleration.x, -msg->linear_acceleration.y);
    d.gyr = Vec3(msg->angular_velocity.z, -msg->angular_velocity.x, -msg->angular_velocity.y);
  } else {
    d.acc = Vec3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    d.gyr = Vec3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
  }

  std::lock_guard<std::mutex> lk(buf_mutex_);
  last_gyr_ = d.gyr;                 // body angular rate for the Odometry twist
  if (!imu_inited_) initializeImu(d);
  imu_buf_.push_back(d);
}

void GenZLioNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lk(buf_mutex_);
    cloud_buf_.push_back(msg);
  }
  // Drain whatever is now processable (IMU must cover each sweep's end time).
  while (tryProcess()) {}
}

bool GenZLioNode::parseCloud(const sensor_msgs::msg::PointCloud2 & msg, PointCloud & out,
                             double & scan_t0, double & scan_dur)
{
  scan_t0 = stampSec(msg.header.stamp);

  // Inspect field metadata only (names + datatypes). All point data below is
  // read through sensor_msgs::PointCloud2ConstIterator -- no raw byte offsets.
  bool has_x = false, has_y = false, has_z = false;
  std::string intensity_field, time_field;
  uint8_t intensity_dt = 0, time_dt = 0;
  for (const auto & f : msg.fields) {
    if (f.name == "x") has_x = true;
    else if (f.name == "y") has_y = true;
    else if (f.name == "z") has_z = true;
    else if (f.name == "intensity" || f.name == "i") { intensity_field = f.name; intensity_dt = f.datatype; }
    else if (f.name == "time" || f.name == "t" || f.name == "timestamp" ||
             f.name == "time_offset" || f.name == "point_time_offset" ||  f.name == "ts" || f.name == "curvature") {
      time_field = f.name; time_dt = f.datatype;
    }
  }
  if (!has_x || !has_y || !has_z) {
    RCLCPP_WARN(get_logger(), "PointCloud2 missing x/y/z fields; dropping.");
    return false;
  }

  const std::size_t n = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
  out.clear();
  out.reserve(n);

  // Pre-extract the optional intensity / per-point time fields with correctly
  // typed const iterators (handles FLOAT32/FLOAT64/UINT32/... transparently).
  std::vector<double> intens, times;
  const bool have_i = !intensity_field.empty();
  const bool have_t = !time_field.empty();
  
  if (have_t) {
    RCLCPP_INFO_ONCE(get_logger(), "Deskew is ON! Found valid time field: '%s'", time_field.c_str());
  } else {
    RCLCPP_WARN_ONCE(get_logger(), "Deskew is OFF! No valid time field found. Skipping deskew.");
  }
  
  if (have_i) { intens.assign(n, 0.0); extractField(msg, intensity_field, intensity_dt, intens); }
  if (have_t) { times.assign(n, 0.0);  extractField(msg, time_field, time_dt, times); }

  const double time_scale = point_time_scale_;
  const double rmin = params_.lidar_min_range;
  const double rmax = params_.lidar_max_range;
  double tmin = std::numeric_limits<double>::max();
  double tmax = -std::numeric_limits<double>::max();

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
  
  // Check if the topic contains "bf_lidar"
  bool is_bf_lidar = (points_topic_.find("bf_lidar") != std::string::npos);
  
  for (std::size_t i = 0; i < n; ++i, ++iter_x, ++iter_y, ++iter_z) {
    PointXYZIT p;
    // Task 5.2 -- Blickfeld static axis swap (sensor frame -> body-aligned FLU):
    //   p.x = y,  p.y = -x,  p.z = z.
    if (is_bf_lidar) {
    // Blickfeld static axis swap (sensor frame -> body-aligned FLU)
     p.x = *iter_y;
     p.y = -(*iter_x);
     p.z = *iter_z;
    } else {
      // Standard pass-through (No swap)
      p.x = *iter_x;
      p.y = *iter_y;
      p.z = *iter_z;
    }
    
    if (have_i) p.intensity = static_cast<float>(intens[i]);
    if (have_t) p.time = times[i] * time_scale;

    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    const double rng = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (rng < rmin || rng > rmax) continue;     // range gate (also drops the origin)

    if (have_t) { tmin = std::min(tmin, p.time); tmax = std::max(tmax, p.time); }
    out.push_back(p);
  }

  //// Normalize per-point time to [0, scan_dur] relative to scan start. Handles
  //// absolute-timestamp drivers (subtract the min) and relative-offset drivers.
  //if (have_t && tmax > tmin) {
  //  for (auto & p : out) p.time -= tmin;
  //  scan_dur = tmax - tmin;
  //} else {
  //  for (auto & p : out) p.time = 0.0;
  //  scan_dur = 0.0;
  //}
  
  //CRITICAL CHANGE: REPLACING THE UPPER BLOCK WITH THIS
  // ==============================================================
  // BIG BLUNT BLOCK 1: TIME HANDLING
  // ==============================================================
  if (deskew_mode_ == 0) {
    // MODE 0: Wipe the time. Lie to the EKF so it anchors at scan_t0.
    for (auto & p : out) p.time = 0.0;
    scan_dur = 0.0;
    
  } else {
    // MODE 1 & 2: Normalize time naturally for the deskew math.
    if (have_t && tmax > tmin) {
      for (auto & p : out) p.time -= tmin;
      scan_dur = tmax - tmin;
    } else {
      for (auto & p : out) p.time = 0.0;
      scan_dur = 0.0;
    }
  }
  
  RCLCPP_INFO_ONCE(get_logger(), "DIAGNOSTIC: scan_dur = %f seconds (tmax: %f, tmin: %f)", scan_dur, tmax, tmin);
  
  return !out.empty();
}

PointCloud GenZLioNode::deskew(const PointCloud & raw,
                               const std::vector<PoseStamp> & traj, double scan_t0)
{
  // // No per-point time or no trajectory -> nothing to undistort.
   //if (raw.empty() || traj.size() < 2) return raw;

   //const PoseStamp & end = traj.back();
   //const Mat3 R_end_inv = end.R.transpose();
   //const Mat3 R_LI = params_.R_IL.transpose();              // I R_L^{-1}
   //const Vec3 t_LI = -R_LI * params_.t_IL;                  // (I T_L)^{-1} translation

   //auto interp = [&](double t_abs, Mat3 & R, Vec3 & p) {
   //  // Clamp / locate bracketing trajectory samples.
   //  if (t_abs <= traj.front().t) { R = traj.front().R; p = traj.front().p; return; }
   //  if (t_abs >= traj.back().t)  { R = traj.back().R;  p = traj.back().p;  return; }
   //  size_t hi = 1;
   //  while (hi < traj.size() && traj[hi].t < t_abs) ++hi;
   //  const PoseStamp & a = traj[hi - 1];
   //  const PoseStamp & b = traj[hi];
   //  const double r = (t_abs - a.t) / std::max(b.t - a.t, 1e-9);
   //  p = a.p + r * (b.p - a.p);                              // linear position interp
   //  R = a.R * so3::Exp(r * so3::Log(a.R.transpose() * b.R));// SLERP-on-SO(3)
   //};
   
   //CRITICAL CHANGE: REPLACING THE UPPER BLOCK WITH THIS
   // ==============================================================
  // BIG BLUNT BLOCK 2: MODE 0 (STRICT BYPASS)
  // ==============================================================
  if (deskew_mode_ == 0) {
    // Because scan_dur is forced to 0.0, the EKF mapped everything at scan_t0.
    // We do absolutely no math here. Just return the raw cloud.
    return raw; 
  }

  // ==============================================================
  // BIG BLUNT BLOCK 3: MODES 1 & 2 (DESKEW MATH)
  // ==============================================================
  if (raw.empty() || traj.size() < 2 || raw.back().time == 0.0) return raw;

  const PoseStamp & start = traj.front();
  const PoseStamp & end = traj.back();
  const Mat3 R_end_inv = end.R.transpose();
  const Mat3 R_LI = params_.R_IL.transpose();              
  const Vec3 t_LI = -R_LI * params_.t_IL;                  

  const double total_time = std::max(end.t - start.t, 1e-9);
  const Vec3 v_avg = (end.p - start.p) / total_time;
  const Vec3 w_avg = so3::Log(start.R.transpose() * end.R) / total_time;

  auto interp = [&](double t_abs, Mat3 & R, Vec3 & p) {
    
    if (deskew_mode_ == 2) {
      // MODE 2: GenZ-ICP Fake Deskew
      const double dt = t_abs - start.t;
      p = start.p + v_avg * dt;
      R = start.R * so3::Exp(w_avg * dt);
      return;
    }

    // MODE 1: Exact IMU Deskew
    if (t_abs <= start.t) { R = start.R; p = start.p; return; }
    if (t_abs >= end.t)   { R = end.R;  p = end.p;  return; }
    size_t hi = 1;
    while (hi < traj.size() && traj[hi].t < t_abs) ++hi;
    const PoseStamp & a = traj[hi - 1];
    const PoseStamp & b = traj[hi];
    const double r = (t_abs - a.t) / std::max(b.t - a.t, 1e-9);
    p = a.p + r * (b.p - a.p);                               
    R = a.R * so3::Exp(r * so3::Log(a.R.transpose() * b.R)); 
  };
//-----------------------------------------------------------------

  PointCloud out;
  out.reserve(raw.size());
  for (const auto & pt : raw) {
    Mat3 R_t; Vec3 p_t;
    interp(scan_t0 + pt.time, R_t, p_t);
    // World point at its observation time: W = T_wI(t) * I_T_L * L p
    const Vec3 p_I = params_.R_IL * pt.vec() + params_.t_IL;
    const Vec3 p_world = R_t * p_I + p_t;
    // Bring back to the LiDAR frame referenced to scan end:
    //   L_end p = (I_T_L)^{-1} * (T_wI(end))^{-1} * W
    const Vec3 p_I_end = R_end_inv * (p_world - end.p);
    const Vec3 p_L_end = R_LI * p_I_end + t_LI;
    PointXYZIT q = pt;
    q.x = p_L_end.x(); q.y = p_L_end.y(); q.z = p_L_end.z();
    out.push_back(q);
  }
  
  // ADD THIS TEMPORARY DIAGNOSTIC BLOCK AT THE END OF deskew():
  if (!raw.empty() && traj.size() >= 2) {
    // Check if the very last point actually moved
    const Vec3 raw_p = raw.back().vec();
    const Vec3 deskew_p = out.back().vec();
    const double diff = (raw_p - deskew_p).norm();
    
    // Only print if the rover is moving faster than 5cm/s to avoid console spam
    if (diff > 0.05) { 
        RCLCPP_INFO(get_logger(), 
          "DESKEW ACTIVE: Moving by %f meters. (Traj size: %zu)", diff, traj.size());
    }
  
    RCLCPP_INFO(rclcpp::get_logger("deskew_diag"), 
      "DESKEW MATH: Traj size = %zu. Point moved by = %f meters. (t_abs: %f, traj_start: %f, traj_end: %f)", 
      traj.size(), diff, (scan_t0 + raw.back().time), traj.front().t, traj.back().t);
  } else {
    RCLCPP_WARN(rclcpp::get_logger("deskew_diag"), 
      "DESKEW ABORTED: Traj size = %zu (needs >= 2)", traj.size());
  }
  
  return out;
}

bool GenZLioNode::tryProcess()
{
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
  std::vector<ImuData> imus;
  double scan_t0 = 0.0, scan_dur = 0.0;
  PointCloud raw;

  {
    std::lock_guard<std::mutex> lk(buf_mutex_);
    if (cloud_buf_.empty() || !imu_inited_) return false;

    cloud_msg = cloud_buf_.front();
    if (!parseCloud(*cloud_msg, raw, scan_t0, scan_dur)) { cloud_buf_.pop_front(); return true; }
    const double scan_end = scan_t0 + scan_dur;

    // Need IMU coverage up to scan end before we can deskew + update.
    if (imu_buf_.empty() || imu_buf_.back().t < scan_end) return false;

    //// Gather IMU spanning [start_ref, scan_end]; drop older samples.
    //const double start_ref = (last_scan_end_ > 0.0) ? last_scan_end_ : scan_t0;
    //for (const auto & d : imu_buf_) {
    //  if (d.t >= start_ref - 1e-3 && d.t <= scan_end + 1e-3) imus.push_back(d);
    //}
    //while (!imu_buf_.empty() && imu_buf_.front().t < scan_end - 1e-3) imu_buf_.pop_front();
    
 
    //CRITICAL FIX: REPLACING THE UPPER BLOCK WITH THIS
    // ----- FLAWLESS IMU BRACKETING (WITH RIGID-MAP SAFETY) -----
    // 1. FAST-FORWARD: Find the target for our Left Bracket. 
    const double start_ref = (last_scan_end_ > 0.0) ? last_scan_end_ : scan_t0;

    // Safely discard old history, but ALWAYS keep the exact message just 
    // before start_ref. This guarantees a perfect Left Bracket for the next scan.
    while (imu_buf_.size() > 1 && imu_buf_[1].t <= start_ref) {
      imu_buf_.pop_front();
    }

    // 2. GATHER: Fill the trajectory array safely.
    for (const auto & d : imu_buf_) {
      if (scan_dur == 0.0) {
        // SAFETY MODE (Deskew OFF): Strictly forbid future messages. 
        // This forces the interpolation anchor to perfectly match the point time.
        if (d.t > scan_end + 1e-3) break;
        imus.push_back(d);
      } else {
        // DESKEW MODE (Deskew ON): Grab messages up to the Right Bracket, then stop.
        // This ensures the interpolation math never has to guess.
        imus.push_back(d);
        if (d.t >= scan_end) break; 
      }
    }
    // -------------------------------------------------------------
    
    cloud_buf_.pop_front(); 
    last_scan_end_ = scan_end;
  }

  if (imus.size() < 2 || raw.empty()) return true;

  // ---- Forward propagation across the sweep, recording the body trajectory ----
  std::vector<PoseStamp> traj;
  traj.reserve(imus.size());
  double t_prev = (last_imu_t_ > 0.0) ? last_imu_t_ : imus.front().t;
  for (const auto & d : imus) {
    const double dt = d.t - t_prev;
    if (dt > 0.0 && dt < 0.5) ekf_->predict(d, dt);
    t_prev = d.t;
    const State & s = ekf_->state();
    traj.push_back(PoseStamp{d.t, s.rot, s.pos});
  }
  last_imu_t_ = t_prev;

  // ---- Backward propagation (deskew) -> S_t ----
  const PointCloud S_t = deskew(raw, traj, scan_t0);

  // ---- (IV) Scale-aware adaptive voxelization -> V_t, V_merge_t ----
  const double dt_scan = (scan_dur > 1e-3) ? scan_dur : 0.1;
  const auto vox = voxelizer_->process(S_t, dt_scan);

  if (first_scan_ || map_->numVoxels() == 0) {
    // Bootstrap the map from the first sweep (no update possible yet).
    const State & s = ekf_->state();
    StoredPointVector world_pts;
    world_pts.reserve(vox.V_merge_t.size());
    for (const auto & p : vox.V_merge_t) {
      StoredPoint sp;
      sp.p = s.rot * (params_.R_IL * p.vec() + params_.t_IL) + s.pos;
      sp.cov = (s.rot * params_.R_IL *
                pointCovLidar(p.vec(), params_.lidar_range_std, params_.lidar_bearing_std) *
                (s.rot * params_.R_IL).transpose()).cast<float>();
      world_pts.push_back(sp);
    }
    map_->insert(world_pts);
    first_scan_ = false;
    RCLCPP_INFO(get_logger(), "Map bootstrapped with %zu points (d_t=%.3f).",
                world_pts.size(), vox.d_t);
    publish(scan_t0 + scan_dur);
    publishScan(scan_t0 + scan_dur, vox.V_merge_t, ekf_->state());
    return true;
  }

  // ---- (V) Hybrid-metric ESIKF state update (Algorithm 3) ----
  const int iters = ekf_->update(vox.V_t, *map_);

  // ---- Map integration with the updated pose (Eq.36): W p_i = W T_I I T_L L p_i ----
  const State & s = ekf_->state();
  const Mat3 R_wl = s.rot * params_.R_IL;
  StoredPointVector world_pts;
  world_pts.reserve(vox.V_merge_t.size());
  for (const auto & p : vox.V_merge_t) {
    StoredPoint sp;
    sp.p = R_wl * p.vec() + (s.rot * params_.t_IL + s.pos);
    sp.cov = (R_wl * pointCovLidar(p.vec(), params_.lidar_range_std, params_.lidar_bearing_std)
              * R_wl.transpose()).cast<float>();
    world_pts.push_back(sp);
  }
  map_->insert(world_pts);

  RCLCPP_DEBUG(get_logger(),
    "scan: N_temp=%d N_des=%d d_t=%.3f m_bar=%.2f |V_t|=%zu iters=%d voxels=%zu",
    vox.N_temp, vox.N_desired, vox.d_t, vox.m_bar, vox.V_t.size(), iters, map_->numVoxels());

  publish(scan_t0 + scan_dur);
  publishScan(scan_t0 + scan_dur, vox.V_merge_t, ekf_->state());
  return true;
}

void GenZLioNode::publish(double stamp)
{
  const State & s = ekf_->state();
  const Mat18 & P = ekf_->cov();
//const Eigen::Quaterniond quat(s.rot);
////////////////////////////////////////
  Eigen::Quaterniond quat(s.rot);
  quat.normalize();
//////////////////
  
  const rclcpp::Time rt(static_cast<int64_t>(stamp * 1e9));

  // nav_msgs/Odometry: pose in the world frame, twist in the child (body) frame.
  const Vec3 v_body = s.rot.transpose() * s.vel;   // R_I^T * W v  (world -> body)
  const Vec3 w_body = last_gyr_ - s.bg;            // debiased gyro = body rate

  auto fill6 = [](std::array<double, 36> & dst, const Eigen::Matrix<double, 6, 6> & C) {
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) dst[i * 6 + j] = C(i, j);
  };

  // ---- Odometry ----
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = rt;
  odom.header.frame_id = world_frame_;
  odom.child_frame_id  = body_frame_;
  odom.pose.pose.position.x = s.pos.x();
  odom.pose.pose.position.y = s.pos.y();
  odom.pose.pose.position.z = s.pos.z();
  odom.pose.pose.orientation.x = quat.x();
  odom.pose.pose.orientation.y = quat.y();
  odom.pose.pose.orientation.z = quat.z();
  odom.pose.pose.orientation.w = quat.w();
  odom.twist.twist.linear.x  = v_body.x();
  odom.twist.twist.linear.y  = v_body.y();
  odom.twist.twist.linear.z  = v_body.z();
  odom.twist.twist.angular.x = w_body.x();
  odom.twist.twist.angular.y = w_body.y();
  odom.twist.twist.angular.z = w_body.z();

  // Pose covariance: posterior over [position(3), rotation(3)] (top-left of P_).
  fill6(odom.pose.covariance, P.block<6, 6>(IDX_P, IDX_P));

  // Twist covariance: linear = R^T Sigma_vv R (world velocity cov rotated into
  // the body frame); angular = gyro-bias cov + gyro measurement noise.
  Eigen::Matrix<double, 6, 6> Ct = Eigen::Matrix<double, 6, 6>::Zero();
  Ct.block<3, 3>(0, 0) = s.rot.transpose() * P.block<3, 3>(IDX_V, IDX_V) * s.rot;
  Ct.block<3, 3>(3, 3) = P.block<3, 3>(IDX_BG, IDX_BG)
                       + (params_.gyr_noise * params_.gyr_noise) * Mat3::Identity();
  fill6(odom.twist.covariance, Ct);
  pub_odom_->publish(odom);

  // ---- TF odom -> body (optional) ----
  if (publish_tf_) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = rt;
    tf.header.frame_id = world_frame_;
    tf.child_frame_id  = body_frame_;
    tf.transform.translation.x = s.pos.x();
    tf.transform.translation.y = s.pos.y();
    tf.transform.translation.z = s.pos.z();
    tf.transform.rotation = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf);
  }

  // ---- Path ----
  geometry_msgs::msg::PoseStamped ps;
  ps.header = odom.header;
  ps.pose = odom.pose.pose;
  path_.header.stamp = rt;
  path_.poses.push_back(ps);
  if (path_.poses.size() > 5000) path_.poses.erase(path_.poses.begin());
  pub_path_->publish(path_);
}

void GenZLioNode::publishScan(double stamp, const PointCloud & lidar_pts, const State & s)
{
  if (!scan_publish_en_ && !scan_bodyframe_pub_en_) return;
  const rclcpp::Time rt(static_cast<int64_t>(stamp * 1e9));

  // World-frame registered cloud: W p = W R_I (I R_L L p + I t_L) + W t_I.
  if (scan_publish_en_) {
    const Mat3 R_wl = s.rot * params_.R_IL;
    const Vec3 t_wl = s.rot * params_.t_IL + s.pos;
    PointCloud world;
    world.reserve(lidar_pts.size());
    for (const auto & p : lidar_pts) {
      const Vec3 w = R_wl * p.vec() + t_wl;
      PointXYZIT q;
      q.x = w.x(); q.y = w.y(); q.z = w.z();
      q.intensity = p.intensity;
      world.push_back(q);
    }
    pub_cloud_->publish(toCloud2(world, world_frame_, rt));
  }

  // Body-frame cloud: I p = I R_L L p + I t_L (body == IMU frame here).
  if (scan_bodyframe_pub_en_) {
    PointCloud body;
    body.reserve(lidar_pts.size());
    for (const auto & p : lidar_pts) {
      const Vec3 b = params_.R_IL * p.vec() + params_.t_IL;
      PointXYZIT q;
      q.x = b.x(); q.y = b.y(); q.z = b.z();
      q.intensity = p.intensity;
      body.push_back(q);
    }
    pub_cloud_body_->publish(toCloud2(body, body_frame_, rt));
  }
}

sensor_msgs::msg::PointCloud2 GenZLioNode::toCloud2(
  const PointCloud & pts, const std::string & frame, const rclcpp::Time & stamp) const
{
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame;
  msg.height = 1;
  msg.is_dense = true;

  sensor_msgs::PointCloud2Modifier mod(msg);
  mod.setPointCloud2Fields(4,
    "x", 1, sensor_msgs::msg::PointField::FLOAT32,
    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32,
    "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
  mod.resize(pts.size());

  sensor_msgs::PointCloud2Iterator<float> ix(msg, "x"), iy(msg, "y"),
                                          iz(msg, "z"), ii(msg, "intensity");
  for (const auto & p : pts) {
    *ix = static_cast<float>(p.x);
    *iy = static_cast<float>(p.y);
    *iz = static_cast<float>(p.z);
    *ii = p.intensity;
    ++ix; ++iy; ++iz; ++ii;
  }
  return msg;
}

#endif  // GENZ_LIO_HEADER_ONLY

}  // namespace genz_lio

// ===========================================================================
//  main
// ===========================================================================
#ifndef GENZ_LIO_HEADER_ONLY
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<genz_lio::GenZLioNode>());
  rclcpp::shutdown();
  return 0;
}
#endif
