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
        executable='lfn3_sub_node',
        name='lfn3_sub_node',
        output='screen'
    )
    
    
    # depth_node = launch_ros.actions.Node(
    #     package='liteflownet3',
    #     executable='depth_calculation_node',
    #     name='depth_calculation_node',
    #     output='screen'
    # )

    return LaunchDescription([
        lfn3_node,
        # depth_node,
    ])
    
    

if __name__ == '__main__':
    generate_launch_description()