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
                    name='bag_reader_node'
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='node1',
                    parameters=[
                        {'image_sub_topic_': '/image_raw',
                         'image_pub_topic_': '/image_out1',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::CvtColorNode',
                    name='node2',
                    parameters=[
                        {'image_sub_topic_': '/image_out1',
                         'image_pub_topic_': '/image_out2',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='node3',
                    parameters=[
                        {'image_sub_topic_': '/image_out2',
                         'image_pub_topic_': '/image_out3',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::CvtColorNode',
                    name='node4',
                    parameters=[
                        {'image_sub_topic_': '/image_out3',
                         'image_pub_topic_': '/image_out4',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='node5',
                    parameters=[
                        {'image_sub_topic_': '/image_out4',
                         'image_pub_topic_': '/image_out5',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::CvtColorNode',
                    name='node6',
                    parameters=[
                        {'image_sub_topic_': '/image_out5',
                         'image_pub_topic_': '/image_out6',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='node7',
                    parameters=[
                        {'image_sub_topic_': '/image_out6',
                         'image_pub_topic_': '/image_out7',}]
                    )
            ],
            output='screen',
    )

    return launch.LaunchDescription([container])