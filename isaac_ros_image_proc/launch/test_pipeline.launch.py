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
                    package='blaser_profilometer',
                    plugin='blaser::BagReaderNitrosNode',
                    name='bagreadernitros_node',
                    parameters=[
                        {'bag_topic': '/resized_image',
                         'pub_image_topic': '/image_raw',
                         'bag_filename_': '/workspaces/isaac_ros-dev/src/rosbag2_image_raw_1280x720/rosbag2_2025_05_14-12_06_14_0.db3',
                         'bframe_rate_display_': True,
                         'bimage_display_': True,}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='median_blur_node',
                    parameters=[
                        {'image_sub_topic_': '/image_raw',
                         'image_pub_topic_': '/image_out1',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MaskingNode',
                    name='masking_node',
                    parameters=[
                        {'image_sub_topic_': '/image_out1',
                         'image_pub_topic_': '/image_out2',}]
                    ),
                # ComposableNode(
                #     package='isaac_ros_image_proc',
                #     plugin='nvidia::isaac_ros::image_proc::CvtColorNode',
                #     name='cvt_color_node',
                #     parameters=[
                #         {'image_sub_topic_': '/image_out1',
                #          'image_pub_topic_': '/image_out3',}]
                #     )
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