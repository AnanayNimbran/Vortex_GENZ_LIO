"""Launch GenZ-LIO.

Starts the estimator node and (by default) an RViz2 session pre-loaded with the
genz_lio.rviz config so the odom frame, the Path and the registered PointCloud2
are visualized out of the box. Disable RViz with `rviz:=false`.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("genz_lio")
    default_cfg = os.path.join(pkg_share, "config", "genz_lio.yaml")
    default_rviz = os.path.join(pkg_share, "config", "genz_lio.rviz")

    config_arg = DeclareLaunchArgument(
        "config", default_value=default_cfg,
        description="Path to the GenZ-LIO parameter YAML.")
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true",
        description="Launch RViz2 with the GenZ-LIO visualization config.")
    rviz_cfg_arg = DeclareLaunchArgument(
        "rviz_config", default_value=default_rviz,
        description="Path to the RViz2 .rviz config.")
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="false",
        description="Use /clock (set true when replaying a rosbag with --clock).")

    use_sim_time = LaunchConfiguration("use_sim_time")

    node = Node(
        package="genz_lio",
        executable="genz_lio_node",
        name="genz_lio_node",
        output="screen",
        parameters=[LaunchConfiguration("config"), {"use_sim_time": use_sim_time}],
        # Remap here if your drivers publish on different topics:
        # remappings=[("/bf_lidar/point_cloud_out", "/your/points"),
        #             ("/mavros/imu/data", "/your/imu")],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([
        config_arg, rviz_arg, rviz_cfg_arg, use_sim_time_arg, node, rviz,
    ])
