#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the launch directory
    pkg_dir = get_package_share_directory('terrain_aware_nav')
    
    # Use the parameter file
    config_file = os.path.join(pkg_dir, 'config', 'terrain_params.yaml')
    
    # Create launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Declare the launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')
        
    # Create the terrain navigation node
    terrain_node_cmd = Node(
        package='terrain_aware_nav',
        executable='terrain_navigation_node',
        name='terrain_navigation_node',
        output='screen',
        parameters=[config_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('/cmd_vel_raw', '/cmd_vel_raw'),
            ('/cmd_vel', '/cmd_vel'),
            ('/terrain_type', '/terrain_type')
        ]
    )
    
    # Create the launch description and add all actions
    ld = LaunchDescription()
    
    # Add all commands to the launch description
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(terrain_node_cmd)
    
    return ld