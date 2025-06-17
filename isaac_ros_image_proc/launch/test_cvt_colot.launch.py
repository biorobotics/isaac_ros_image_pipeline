import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


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
                         'bag_filename_': '/workspaces/isaac_ros-dev/src/rosbag2_image_raw_1280x720/rosbag2_2025_05_14-12_06_14_0.db3',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::CvtColorNode',
                    name='cvt_color_node'
                    )               
            ],
            output='screen',
    )

    return launch.LaunchDescription([container])