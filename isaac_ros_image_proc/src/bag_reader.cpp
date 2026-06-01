#include "isaac_ros_image_proc/bag_reader.hpp"

#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

namespace nvidia
{
    namespace isaac_ros
    {
        namespace image_proc
        {

            //Create a struct that stoes pair of ros ecnoding and NitrosImage encoding
            struct ImageFormatPair
            {
                std::string ros_encoding;
                std::string nitros_encoding;
            };
            
            // Define the supported encodings
            const std::vector<ImageFormatPair> supported_encodings = {
                {"mono8", nvidia::isaac_ros::nitros::nitros_image_mono8_t::supported_type_name},
                {"bgr8", nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name},
                {"rgb8", nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name},
                {"rgba8", nvidia::isaac_ros::nitros::nitros_image_rgba8_t::supported_type_name},
                {"bgra8", nvidia::isaac_ros::nitros::nitros_image_bgra8_t::supported_type_name},
                {"mono16", nvidia::isaac_ros::nitros::nitros_image_mono16_t::supported_type_name},
                {"bgr16", nvidia::isaac_ros::nitros::nitros_image_bgr16_t::supported_type_name},
                {"rgb16", nvidia::isaac_ros::nitros::nitros_image_rgb16_t::supported_type_name},
            };

            // Write a function that takes in a ROS encoding and returns the ImageFormatPair
            ImageFormatPair to_image_format(const std::string &ros_encoding)
            {
                for (const auto &pair : supported_encodings)
                {
                    if (pair.ros_encoding == ros_encoding)
                    {
                        return pair;
                    }
                }
                throw std::runtime_error("Unsupported ROS encoding: " + ros_encoding);
            }

            // Constructor for BagReaderNode
            BagReaderNode::BagReaderNode(const rclcpp::NodeOptions options)
                : Node("bag_reader_node", options)
            {
                RCLCPP_INFO(this->get_logger(), "bag_reader_node is starting");

                // Initialize parameter listener and params
                param_listener_ = std::make_shared<bag_reader_node::ParamListener>(this->get_node_parameters_interface());
                params_ = param_listener_->get_params();

                // Initialize ROS bag reader
                rosbag2_storage::StorageOptions storage_options;
                storage_options.uri = params_.bag_filename_;
                reader_ = rosbag2_transport::ReaderWriterFactory::make_reader(storage_options);
                reader_->open(storage_options);

                RCLCPP_INFO(this->get_logger(), "Loading images from bag......");

                while (reader_->has_next())
                {
                    auto msgs = reader_->read_next();
                    if (msgs->topic_name != params_.bag_topic)
                    {
                        continue;
                    }

                    rclcpp::SerializedMessage serialized_msg(*msgs->serialized_data);
                    sensor_msgs::msg::Image image_msg;
                    serialization_.deserialize_message(&serialized_msg, &image_msg);
                    image_msgs_.push_back(image_msg); // image_msgs_.push_back(std::move(image_msg));
                }

                // print height and width of the first image
                if (!image_msgs_.empty())
                {
                    RCLCPP_INFO(this->get_logger(), "First image height: %d, width: %d", image_msgs_[0].height, image_msgs_[0].width);
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "No images found in the bag file.");
                    exit(0);
                }

                RCLCPP_INFO(this->get_logger(), "Loaded %zu image messages from bag.", image_msgs_.size());

                //print the image encodings from the first image
                if (!image_msgs_.empty())
                {
                    RCLCPP_INFO(this->get_logger(), "First image encoding: %s", image_msgs_[0].encoding.c_str());
                }

                // Initialize the ImageFormatPair struct
                ImageFormatPair image_format_pair = to_image_format(image_msgs_[0].encoding);

                // Initialize Nitros publisher
                nitros_pub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                    nvidia::isaac_ros::nitros::NitrosImage>>(
                    this, params_.pub_image_topic,
                    image_format_pair.nitros_encoding);

                // Initialize image index and frame rate display
                image_index_ = 0;
                frame_rate_display_ = 0.0;

                // Create a timer to call the timer_callback function
                timer_ = this->create_wall_timer(50ms, std::bind(&BagReaderNode::timer_callback, this));
            }

            void BagReaderNode::timer_callback()
            {
                // Check if any parameters have changed
                if (param_listener_->is_old(params_))
                {
                    params_ = param_listener_->get_params();
                }

                if (image_msgs_.empty())
                {
                    RCLCPP_WARN(this->get_logger(), "No image messages loaded.");
                    exit(0);
                }

                if (image_index_ >= image_msgs_.size())
                {
                    RCLCPP_INFO(this->get_logger(), "Looping image playback.");
                    image_index_ = 0;
                }

                sensor_msgs::msg::Image &image_msg = image_msgs_[image_index_];
                size_t buffer_size = image_msg.step * image_msg.height;

                void *buffer;
                cudaMalloc(&buffer, buffer_size);
                cudaMemcpy(buffer, image_msg.data.data(), buffer_size, cudaMemcpyDefault);

                std_msgs::msg::Header header;
                header.stamp = this->now();
                header.frame_id = image_msg.header.frame_id;
                nvidia::isaac_ros::nitros::NitrosImage nitros_image =
                    nvidia::isaac_ros::nitros::NitrosImageBuilder()
                        .WithHeader(header)
                        .WithEncoding(image_msg.encoding)
                        .WithDimensions(image_msg.height, image_msg.width)
                        .WithGpuData(buffer)
                        .Build();

                nitros_pub_ptr_->publish(nitros_image);

                // Print the frame rate if required
                if (params_.bframe_rate_display_)
                {
                    // calculating frame rate
                    static rclcpp::Time last_time = this->now();
                    rclcpp::Time current_time = this->now();
                    static int count = 0;
                    static double total_time = 0;
                    total_time += (current_time - last_time).seconds();
                    count++;
                    if (count == 5)
                    {
                        RCLCPP_INFO(this->get_logger(), "average frame rate: %f", 1 / (total_time / count));
                        frame_rate_display_ = 1 / (total_time / count);
                        count = 0;
                        total_time = 0;
                    }
                    last_time = current_time;
                    // end of frame rate calculation
                }
                // Display the image if required
                if (params_.bimage_display_)
                {
                    // cv::Mat image(image_msg.height, image_msg.width, CV_8UC3, image_msg.data.data());
                    cv::Mat image;
                    if (image_msg.encoding == "rgb8")
                    {
                        cv::Mat temp(image_msg.height, image_msg.width, CV_8UC3, image_msg.data.data());
                        cv::cvtColor(temp, image, cv::COLOR_RGB2BGR);
                    }
                    else if (image_msg.encoding == "bgr8")
                    {
                        image = cv::Mat(image_msg.height, image_msg.width, CV_8UC3, image_msg.data.data());
                    }
                    else if (image_msg.encoding == "mono8")
                    {
                        image = cv::Mat(image_msg.height, image_msg.width, CV_8UC1, image_msg.data.data());
                    }
                    else
                    {
                        RCLCPP_ERROR(this->get_logger(), "Unsupported image encoding for display: %s", image_msg.encoding.c_str());
                        return;
                    }
                    cv::Mat resized_image;
                    cv::resize(image, resized_image, cv::Size(), params_.resize_factor, params_.resize_factor);

                    if (params_.bframe_rate_display_)
                    {
                        std::string text = cv::format("FPS: %.2f", frame_rate_display_);
                        int font = cv::FONT_HERSHEY_SIMPLEX;
                        cv::Point origin(params_.watermark_x, params_.watermark_y);
                        cv::putText(resized_image, text, origin, font, params_.watermark_scale, CV_RGB(124, 252, 0), params_.watermark_thickness);
                    }

                    cv::imshow("Image Viewer", resized_image);
                    cv::waitKey(1);
                }

                // image_index_++;
            }
        } // namespace image_proc
    } // namespace isaac_ros
} // namespace nvidia

// Register as a component.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::BagReaderNode)