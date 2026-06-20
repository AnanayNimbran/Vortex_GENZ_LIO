// Unit tests for the header-inline GenZ-LIO math: SO(3) ops, boxplus/boxminus,
// the PD voxel controller (Eq.1-8), and the voxel-pruning geometry (Fig.5/6).
// Builds against Eigen only (GENZ_LIO_HEADER_ONLY drops the ROS node).
#define GENZ_LIO_HEADER_ONLY
#include "genz_lio/genz_lio.hpp"

#include <gtest/gtest.h>

using namespace genz_lio;

TEST(SO3, ExpLogRoundTrip)
{
  const Vec3 w(0.3, -0.7, 1.1);
  EXPECT_LT((so3::Log(so3::Exp(w)) - w).norm(), 1e-9);
  // Identity and tiny-angle stability.
  EXPECT_LT(so3::Log(Mat3::Identity()).norm(), 1e-12);
  const Vec3 tiny(1e-10, -2e-10, 5e-11);
  EXPECT_LT((so3::Log(so3::Exp(tiny)) - tiny).norm(), 1e-12);
}

TEST(SO3, RightJacobianInverse)
{
  const Vec3 w(0.2, 0.9, -0.4);
  EXPECT_LT((so3::rightJacobian(w) * so3::rightJacobianInv(w) - Mat3::Identity()).norm(), 1e-9);
}

TEST(State, BoxPlusMinusConsistency)
{
  State x;
  x.pos = Vec3(1, 2, 3);
  x.rot = so3::Exp(Vec3(0.1, 0.2, -0.3));
  x.vel = Vec3(0.5, -0.5, 0.1);
  Vec18 d;
  d.setLinSpaced(-0.4, 0.4);
  const State y = x.boxplus(d);
  EXPECT_LT((y.boxminus(x) - d).norm(), 1e-9);            // (x [+] d) [-] x == d
  EXPECT_LT((reparamJacobian(x, x) - Mat18::Identity()).norm(), 1e-9);  // J(x,x)=I
}

TEST(Controller, SetpointMonotoneAndClamped)
{
  Params p;
  AdaptiveVoxelizer av(p);
  // Confined (small m_bar): setpoint strictly inside (N_min, N_max).
  auto c_near = av.computeControl(5.0, 2000, 0.1);
  EXPECT_GT(c_near.N_desired, p.N_min);
  EXPECT_LT(c_near.N_desired, p.N_max);
  // Open (m_bar >= tau_m): setpoint saturates at N_max (Eq.1).
  auto c_far = av.computeControl(35.0, 2000, 0.1);
  EXPECT_EQ(c_far.N_desired, p.N_max);
  // rho monotone in m_bar -> larger scenes demand more points.
  EXPECT_GT(av.computeControl(20.0, 2000, 0.1).N_desired,
            av.computeControl(10.0, 2000, 0.1).N_desired);
  // Voxel size stays clamped to [d_min, d_max] (Eq.8).
  EXPECT_GE(c_near.d_t, p.d_min);
  EXPECT_LE(c_near.d_t, p.d_max);
}

TEST(Controller, PdReducesPositiveError)
{
  // If N_temp < N_desired (too few points), e>0 -> dd<0 -> voxel shrinks (Eq.7-8).
  Params p;
  AdaptiveVoxelizer av(p);
  const double d_before = av.currentVoxelSize();
  auto c = av.computeControl(10.0, 100 /*far below setpoint*/, 0.1);
  EXPECT_GT(c.e, 0.0);
  EXPECT_LT(c.d_t, d_before + 1e-12);
}

TEST(VoxelGeometry, CandidateCountsMatchFig5)
{
  // center / surface / edge / corner -> 0 / 1 / 3 / 7 neighbours.
  EXPECT_EQ(voxutil::candidateOffsets(Eigen::Vector3i(0, 0, 0)).size(), 0u);
  EXPECT_EQ(voxutil::candidateOffsets(Eigen::Vector3i(1, 0, 0)).size(), 1u);
  EXPECT_EQ(voxutil::candidateOffsets(Eigen::Vector3i(-1, 1, 0)).size(), 3u);
  EXPECT_EQ(voxutil::candidateOffsets(Eigen::Vector3i(1, -1, 1)).size(), 7u);
}

TEST(VoxelGeometry, RegionSignAndDistance)
{
  const double d = 0.5;
  const VoxelKey root{0, 0, 0};
  // Point near the +x high face, middle in y/z -> surface case toward +x.
  const Vec3 p(0.46, 0.25, 0.25);
  const auto s = voxutil::regionSign(p, root, d);
  EXPECT_EQ(s, Eigen::Vector3i(1, 0, 0));
  // Distance to +x neighbour == distance to the x=d face (point-to-surface).
  const double dist = voxutil::computeDistanceToVoxel(p, root, Eigen::Vector3i(1, 0, 0), d);
  EXPECT_NEAR(dist, d - p.x(), 1e-12);
  // Distance to the +x+y edge neighbour == sqrt of both face gaps.
  const double de = voxutil::computeDistanceToVoxel(p, root, Eigen::Vector3i(1, 1, 0), d);
  EXPECT_NEAR(de, std::hypot(d - p.x(), d - p.y()), 1e-12);
}

TEST(PointCov, SymmetricPositiveSemiDefinite)
{
  const Mat3 C = pointCovLidar(Vec3(3, 1, -2), 0.02, 0.0017);
  EXPECT_LT((C - C.transpose()).norm(), 1e-12);
  Eigen::SelfAdjointEigenSolver<Mat3> es(C);
  EXPECT_GE(es.eigenvalues()(0), -1e-15);
}

// NOTE: no custom main() here -- ament_add_gtest links GTest::gtest_main, which
// provides main(). Defining our own caused a duplicate-symbol link error.
