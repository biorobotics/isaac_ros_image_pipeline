#ifndef ISAAC_ROS_IMAGE_PROC_BAG_READER_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC_BAG_READER_NODE_HPP_

#include <string>
#include <vector>

#include <isaac_ros_image_proc/bag_reader_parameters.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_transport/reader_writer_factory.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"

#include <cuda_runtime.h>

namespace nvidia
{
    namespace isaac_ros
    {
        namespace image_proc
        {
            class BagReaderNode : public rclcpp::Node
            {
            public:
                explicit BagReaderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

            private:
                void timer_callback();

                rclcpp::TimerBase::SharedPtr timer_;
                std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>> nitros_pub_ptr_;

                rclcpp::Serialization<sensor_msgs::msg::Image> serialization_;
                std::unique_ptr<rosbag2_cpp::Reader> reader_;
                std::vector<sensor_msgs::msg::Image> image_msgs_;
                size_t image_index_;
                double frame_rate_display_;

                // Parameters and parameter listener
                std::shared_ptr<bag_reader_node::ParamListener> param_listener_;
                bag_reader_node::Params params_;
            };
        } // namespace image_proc
    } // namespace isaac_ros
} // namespace nvidia

#endif // ISAAC_ROS_IMAGE_PROC_BAG_READER_NODE_HPP_