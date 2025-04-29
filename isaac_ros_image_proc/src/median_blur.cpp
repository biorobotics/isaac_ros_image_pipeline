#include "isaac_ros_image_proc/median_blur.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"

#include <nvcv/Size.hpp>
#include <iostream>

namespace nvidia
{
    namespace isaac_ros
    {
        namespace image_proc
        {
            MedianBlurNode::MedianBlurNode(const rclcpp::NodeOptions &options)
                : Node("median_blur_node", options),
                  median_blur_op_(max_var_shape_batch_size_) // Initialize MedianBlur operator using the constant
            {
                // Initialize params from generate parameter library
                param_listener_ = std::make_shared<median_blur_node::ParamListener>(this->get_node_parameters_interface());
                auto params_ = param_listener_->get_params();

                // Set image and processing parameters.
                batch_size_ = params_.batch_size_;
                kernel_width_ = params_.kernel_width_;
                kernel_height_ = params_.kernel_height_;

                // Create a CUDA stream for kernel execution.
                cudaError_t streamErr = cudaStreamCreate(&stream_);
                if (streamErr != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to create CUDA stream!");
                    // You might handle this error by aborting construction, throwing an exception, etc.
                    exit(0);
                }

                // Create a subscriber
                nitros_sub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                    nvidia::isaac_ros::nitros::NitrosImageView>>(
                    this, params_.image_sub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name,
                    std::bind(&MedianBlurNode::input_callback, this, std::placeholders::_1));

                // Create a publisher
                nitros_pub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                    nvidia::isaac_ros::nitros::NitrosImage>>(
                    this, params_.image_pub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name);
            }

            MedianBlurNode::~MedianBlurNode()
            {
            }

            void MedianBlurNode::input_callback(const nvidia::isaac_ros::nitros::NitrosImageView &view)
            {
                int input_image_width_ = view.GetWidth();
                int input_image_height_ = view.GetHeight();
                int input_image_channels_ = 3;
                int output_image_width_ = input_image_width_;
                int output_image_height_ = input_image_height_;
                int output_image_channels_ = input_image_channels_;
                size_t buffer_size_ = view.GetSizeInBytes();

                //
                // Calculate tensor requirements for the input image.
                //
                nvcv::Tensor::Requirements reqs_ = nvcv::Tensor::CalcRequirements(
                    batch_size_,
                    {input_image_width_, input_image_height_},
                    nvcv::FMT_BGR8);

                // Allocate input buffer and setup strides.
                input_image_buffer_.strides[3] = sizeof(uint8_t);
                input_image_buffer_.strides[2] = input_image_channels_ * input_image_buffer_.strides[3];
                input_image_buffer_.strides[1] = input_image_width_ * input_image_buffer_.strides[2];
                input_image_buffer_.strides[0] = input_image_height_ * input_image_buffer_.strides[1];

                cudaError_t err = cudaMalloc(&input_image_buffer_.basePtr, batch_size_ * input_image_buffer_.strides[0]);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to allocate CUDA memory for input image.");
                    exit(0);
                }

                nvcv::TensorDataStridedCuda in_data(
                    nvcv::TensorShape{reqs_.shape, reqs_.rank, reqs_.layout},
                    nvcv::DataType{reqs_.dtype},
                    input_image_buffer_);
                input_image_tensor_ = nvcv::TensorWrapData(in_data);

                //
                // Allocate and initialize buffer for the output image.
                //
                output_image_buffer_.strides[3] = sizeof(uint8_t);
                output_image_buffer_.strides[2] = output_image_channels_ * output_image_buffer_.strides[3];
                output_image_buffer_.strides[1] = output_image_width_ * output_image_buffer_.strides[2];
                output_image_buffer_.strides[0] = output_image_height_ * output_image_buffer_.strides[1];

                err = cudaMalloc(&output_image_buffer_.basePtr, batch_size_ * output_image_buffer_.strides[0]);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to allocate CUDA memory for output image.");
                    exit(0);
                }

                nvcv::TensorDataStridedCuda blur_data(
                    nvcv::TensorShape{reqs_.shape, reqs_.rank, reqs_.layout},
                    nvcv::DataType{reqs_.dtype},
                    output_image_buffer_);
                output_image_tensor_ = nvcv::TensorWrapData(blur_data);

                // Copy the incoming Nitros image GPU data into the input tensor.
                err = cudaMemcpy(input_image_buffer_.basePtr, view.GetGpuData(), buffer_size_, cudaMemcpyDefault);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to copy nitros input image to input image buffer.");
                    exit(0);
                }

                // Median blur processing.
                nvcv::Size2D kernel_size_(kernel_width_, kernel_height_);
                median_blur_op_(stream_, input_image_tensor_, output_image_tensor_, kernel_size_);
                RCLCPP_INFO(this->get_logger(), "Median blur processing...");

                // Prepare a temporary CUDA buffer for publishing the image.
                size_t output_image_buffer_size = output_image_width_ * output_image_height_ * output_image_channels_ * sizeof(uint8_t);
                void *buffer = nullptr;
                err = cudaMalloc(&buffer, output_image_buffer_size);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to allocate CUDA memory for publishing blur image.");
                    exit(0);
                }

                err = cudaMemcpy(buffer, output_image_buffer_.basePtr, output_image_buffer_size, cudaMemcpyDefault);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to copy crop image buffer to nitros output.");
                    cudaFree(buffer);
                    exit(0);
                }

                // Build and publish the Nitros image.
                std_msgs::msg::Header outpu_image_header;
                outpu_image_header.stamp = this->now();
                outpu_image_header.frame_id = "blur_image";

                nvidia::isaac_ros::nitros::NitrosImage ouput_nitros_image =
                    nvidia::isaac_ros::nitros::NitrosImageBuilder()
                        .WithHeader(outpu_image_header)
                        .WithEncoding(sensor_msgs::image_encodings::BGR8)
                        .WithDimensions(output_image_height_, output_image_width_) // height, width
                        .WithGpuData(buffer)
                        .Build();

                nitros_pub_ptr_->publish(ouput_nitros_image);
                // RCLCPP_INFO(this->get_logger(), "Published Nitros image with GPU pointer: %p", buffer);
            }

        } // namespace image_proc
    } // namespace isaac_ros
} // namespace nvidia

// Register as a component.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::MedianBlurNode)