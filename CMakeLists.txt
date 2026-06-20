cmake_minimum_required(VERSION 3.16)
project(genz_lio)

# (Unofficial) GenZ-LIO reproduction. Targeted at ROS 2 Humble on Ubuntu 22.04, but the
# rclcpp / message APIs used here are stable through Iron and Jazzy as well.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -march=native lets Eigen emit SSE/AVX for the host CPU. This REQUIRES the
# EIGEN_MAKE_ALIGNED_OPERATOR_NEW / aligned_allocator additions in the headers,
# otherwise over-aligned fixed-size Eigen members would segfault at runtime.
add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

set(GENZ_DEPS
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_ros)

add_executable(genz_lio_node
  src/genz_lio.cpp)

target_include_directories(genz_lio_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIRS})

ament_target_dependencies(genz_lio_node ${GENZ_DEPS})
target_link_libraries(genz_lio_node Eigen3::Eigen)

install(TARGETS genz_lio_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

# ---------------------------------------------------------------------------
# Header-only numerical unit test for the core math (SO(3), boxplus/boxminus,
# PD voxel controller, voxel-pruning geometry). Builds against Eigen only, so
# it runs in CI without ROS or PCL.
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest QUIET)
  if(ament_cmake_gtest_FOUND)
    ament_add_gtest(test_genz_math test/test_genz_math.cpp)
    target_include_directories(test_genz_math PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(test_genz_math Eigen3::Eigen)
  endif()
endif()

ament_package()
