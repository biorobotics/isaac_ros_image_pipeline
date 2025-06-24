#include "isaac_ros_image_proc/laseer_points.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"

namespace nvidia
{
    namespace isaac_ros
    {
        namespace image_proc
        {
            namespace
            {
                inline void CheckCudaErrors(cudaError_t code, const char *file, const int line)
                {
                    if (code != cudaSuccess)
                    {
                        const std::string message = "CUDA error returned at " + std::string(file) + ":" +
                                                    std::to_string(line) + ", Error code: " + std::to_string(code) +
                                                    " (" + std::string(cudaGetErrorString(code)) + ")";
                        throw std::runtime_error(message);
                    }
                }
            } // namespace

            LaserPointsNode::LaserPointsNode(const rclcpp::NodeOptions options)
                : Node("laser_points_node", options),
                  frame_rate_display_(0.0)
            {
                // Initialize params from generate parameter library
                param_listener_ = std::make_shared<laser_points_node::ParamListener>(this->get_node_parameters_interface());
                params_ = param_listener_->get_params();

                // Create a subscriber
                nitros_sub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                    nvidia::isaac_ros::nitros::NitrosImageView>>(
                    this, params_.image_sub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name,
                    std::bind(&LaserPointsNode::input_callback, this, std::placeholders::_1)); // diagnostics config and qos TODO

                // Create a publisher
                nitros_pub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                    nvidia::isaac_ros::nitros::NitrosImage>>(
                    this, params_.image_pub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name); // diagnostics config and qos TODO

                CheckCudaErrors(cudaStreamCreate(&stream_), __FILE__, __LINE__);
            }

            void LaserPointsNode::input_callback(const nvidia::isaac_ros::nitros::NitrosImageView &view)
            {
                // Check if any parameters have changed
                if (param_listener_->is_old(params_))
                {
                    params_ = param_listener_->get_params();
                }

                // Get the input image data and its properties
                const uint32_t input_image_width_ = view.GetWidth();
                const uint32_t input_image_height_ = view.GetHeight();
                const int input_image_channels_ = sensor_msgs::image_encodings::numChannels(view.GetEncoding());
                const uint32_t output_image_width_ = input_image_width_;
                const uint32_t output_image_height_ = input_image_height_;
                const int output_image_channels_ = 1;

                // Wrap input GPU memory as GpuMat
                cv::cuda::GpuMat input_image_(input_image_height_, input_image_width_, CV_8UC3, const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(view.GetGpuData())));

                // Allocate the memory buffer for output ourselves
                uint8_t *raw_output_image_buffer{nullptr};
                const size_t output_image_buffer_size_ = output_image_width_ * output_image_height_ * output_image_channels_ * sizeof(uint8_t);
                CheckCudaErrors(cudaMallocAsync(&raw_output_image_buffer, params_.batch_size_ * output_image_buffer_size_, stream_), __FILE__, __LINE__);

                // Wrap output GPU memory as GpuMat
                cv::cuda::GpuMat im_v_(params_.batch_size_ * output_image_height_,
                                           output_image_width_,
                                           CV_8UC1, raw_output_image_buffer);


                CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__); // Is this needed?




                // Create a NitrosImage message and publish it
                std_msgs::msg::Header header_;
                header_.frame_id = view.GetFrameId();
                header_.stamp.sec = view.GetTimestampSeconds();
                header_.stamp.nanosec = view.GetTimestampNanoseconds();

                nvidia::isaac_ros::nitros::NitrosImage output_image_msg_ =
                    nvidia::isaac_ros::nitros::NitrosImageBuilder()
                        .WithHeader(header_)
                        .WithDimensions(output_image_height_, output_image_width_)
                        .WithEncoding(sensor_msgs::image_encodings::MONO8) // Assuming output is single channel
                        .WithGpuData(raw_output_image_buffer)
                        .Build();

                nitros_pub_ptr_->publish(output_image_msg_);

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
                //     // end of frame rate calculation
                // }

                // // Display the image if required
                // if (params_.bimage_display_)
                // {
                //     // Use opencv to view the mask
                //     cv::Mat v_cpu;
                //     im_v_.download(v_cpu);

                //     cv::Mat resized_image;
                //     cv::resize(v_cpu, resized_image, cv::Size(), params_.resize_factor, params_.resize_factor);

                //     if (params_.bframe_rate_display_)
                //     {
                //         std::string text = cv::format("FPS: %.2f", frame_rate_display_);
                //         int font = cv::FONT_HERSHEY_SIMPLEX;
                //         cv::Point origin(params_.watermark_x, params_.watermark_y);
                //         cv::putText(resized_image, text, origin, font, params_.watermark_scale, CV_RGB(252, 252, 252), params_.watermark_thickness);
                //     }

                //     cv::imshow("Laser_points", resized_image);
                //     cv::waitKey(1);
                // }
            }

            LaserPointsNode::~LaserPointsNode()
            {
                CheckCudaErrors(cudaStreamDestroy(stream_), __FILE__, __LINE__);
            }

        } // namespace image_proc
    } // namespace isaac_ros
} // namespace nvidia

// Register as a component.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::LaserPointsNode)