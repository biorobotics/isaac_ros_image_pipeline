import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description with multiple components."""
    container = ComposableNodeContainer(
            name='test_container2',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::BagReaderNode',
                    name='bag_reader_node2',
                    parameters=[
                        {'pub_image_topic': '/image_raw2',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='node01',
                    parameters=[
                        {'image_sub_topic_': '/image_raw2',
                         'image_pub_topic_': '/image_out01',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::CvtColorNode',
                    name='node02',
                    parameters=[
                        {'image_sub_topic_': '/image_out01',
                         'image_pub_topic_': '/image_out02',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='node03',
                    parameters=[
                        {'image_sub_topic_': '/image_out02',
                         'image_pub_topic_': '/image_out03',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::CvtColorNode',
                    name='node04',
                    parameters=[
                        {'image_sub_topic_': '/image_out03',
                         'image_pub_topic_': '/image_out04',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='node05',
                    parameters=[
                        {'image_sub_topic_': '/image_out04',
                         'image_pub_topic_': '/image_out05',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::CvtColorNode',
                    name='node06',
                    parameters=[
                        {'image_sub_topic_': '/image_out05',
                         'image_pub_topic_': '/image_out06',}]
                    ),
                ComposableNode(
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::MedianBlurNode',
                    name='node07',
                    parameters=[
                        {'image_sub_topic_': '/image_out06',
                         'image_pub_topic_': '/image_out07',}]
                    )
            ],
            output='screen',
    )

    return launch.LaunchDescription([container])