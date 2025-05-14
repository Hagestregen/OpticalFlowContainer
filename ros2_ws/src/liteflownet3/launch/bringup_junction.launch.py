#!/usr/bin/env python3
import os
import launch
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import launch_ros.actions

def generate_launch_description():
    # Create a unique output folder for the bag.

    

    lfn3_node = launch_ros.actions.Node(
        package='liteflownet3',
        executable='sub_n_pub_lfn3_node',
        name='sub_n_pub_lfn3_node',
        output='screen'
    )
    
    
    depth_node = launch_ros.actions.Node(
        package='liteflownet3',
        executable='depth_subandpub_node',
        name='depth_subandpub_node',
        output='screen'
    )
    
    junction_node = launch_ros.actions.Node(
        package='junction_point_detector',
        executable='junction_point_detector_node',
        name='junction_point_detector_node',
        output='screen'
    )

    return LaunchDescription([
        lfn3_node,
        # depth_node,
        junction_node,
    ])
    
    

if __name__ == '__main__':
    generate_launch_description()
    
    
    