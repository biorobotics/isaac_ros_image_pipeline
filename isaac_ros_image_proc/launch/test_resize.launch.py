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
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::BagReaderNode',
                    name='bag_reader_node',
                    parameters=[
                        {'bag_topic': '/image_raw',
                         'pub_image_topic': '/image_raw',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::Resize',
                    name='resize_node',
                    parameters=[
                        {'image_sub_topic_': '/image_raw',
                         'image_pub_topic_': '/resized_image',
                         'output_image_width_': 1280,
                         'output_image_height_': 720,}]
                    )               
            ],
            output='screen',
    )

    return launch.LaunchDescription([container])