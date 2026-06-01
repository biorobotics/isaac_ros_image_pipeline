import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description with multiple components."""
    container = ComposableNodeContainer(
            name='test_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::BagReaderNode',
                    name='bag_reader_node',
                    parameters=[
                        {'bag_topic': '/hsv_mask',
                         'pub_image_topic': '/hsv_mask',
                         'bag_filename_': '/workspaces/isaac_ros-dev/src/rosbag2_hsv_mask_1280x720/rosbag2_2025_08_11-13_45_51_0.db3',
                         'bframe_rate_display_': True,
                         'bimage_display_': True,}]
                )
                # ComposableNode(
                #     package='isaac_ros_image_proc',
                #     plugin='nvidia::isaac_ros::image_proc::BagReaderNode',
                #     name='bag_reader_node',
                #     parameters=[
                #         {'bag_topic': '/resized_image',
                #          'pub_image_topic': '/image_raw',
                #          'bag_filename_': '/workspaces/isaac_ros-dev/src/rosbag2_image_raw_1280x720/rosbag2_2025_05_14-12_06_14_0.db3',
                #          'bframe_rate_display_': True,
                #          'bimage_display_': True,}]
                # )           
            ],
            output='screen',
    )

    # rqt_reconfig 
    rqt_reconfig = Node(
        package='rqt_reconfigure',
        executable='rqt_reconfigure',
        name='rqt_reconfigure',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )     

    return launch.LaunchDescription([container, rqt_reconfig])