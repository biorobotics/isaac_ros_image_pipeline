#include "isaac_ros_image_proc/median_blur.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"

#include <nvcv/Size.hpp>

namespace nvidia
{
    namespace isaac_ros
    {
        namespace image_proc
        {
            MedianBlurNode::MedianBlurNode(const rclcpp::NodeOptions &options)
                : Node("median_blur_node", options),
                  medianBlurOp_(maxVarShapeBatchSize) // Initialize MedianBlur operator using the constant
            {
                // Create a CUDA stream for kernel execution.
                cudaError_t streamErr = cudaStreamCreate(&stream_);
                if (streamErr != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to create CUDA stream!");
                    // You might handle this error by aborting construction, throwing an exception, etc.
                    return;
                }
                // Create a subscriber
                nitros_sub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                    nvidia::isaac_ros::nitros::NitrosImageView>>(
                    this, "image_raw",
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name,
                    std::bind(&MedianBlurNode::input_callback, this, std::placeholders::_1));

                // Create a publisher
                nitros_pub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                    nvidia::isaac_ros::nitros::NitrosImage>>(
                    this, "blur_image",
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name);

                // Set image and processing parameters.
                batch_size_ = 1;
                input_image_channels_ = 3;
                input_image_width_ = 2448;
                input_image_height_ = 1840;
                output_image_channels_ = 3;
                output_image_width_ = 2448;
                output_image_height_ = 1840;
                kernelWidth = 3; // blur kernel size
                kernelHeight = 3;

                //
                // Calculate tensor requirements for the input image.
                //
                nvcv::Tensor::Requirements reqs = nvcv::Tensor::CalcRequirements(
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
                    return;
                }

                nvcv::TensorDataStridedCuda in_data(
                    nvcv::TensorShape{reqs.shape, reqs.rank, reqs.layout},
                    nvcv::DataType{reqs.dtype},
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
                    return;
                }

                nvcv::TensorDataStridedCuda blur_data(
                    nvcv::TensorShape{reqs.shape, reqs.rank, reqs.layout},
                    nvcv::DataType{reqs.dtype},
                    output_image_buffer_);
                output_image_tensor_ = nvcv::TensorWrapData(blur_data);
            }

            MedianBlurNode::~MedianBlurNode()
            {
            }

            void MedianBlurNode::input_callback(const nvidia::isaac_ros::nitros::NitrosImageView &view)
            {
                // Copy the incoming Nitros image GPU data into the input tensor.
                size_t buffer_size = view.GetSizeInBytes();
                cudaError_t err = cudaMemcpy(input_image_buffer_.basePtr, view.GetGpuData(), buffer_size, cudaMemcpyDefault);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to copy nitros input image to input image buffer.");
                    return;
                }

                // Median blur processing.
                nvcv::Size2D kSize_(kernelWidth, kernelHeight);
                medianBlurOp_(stream_, input_image_tensor_, output_image_tensor_, kSize_);
                RCLCPP_INFO(this->get_logger(), "Median blur processing...");

                // Prepare a temporary CUDA buffer for publishing the image.
                size_t output_image_buffer_size = output_image_width_ * output_image_height_ * output_image_channels_ * sizeof(uint8_t);
                void *buffer = nullptr;
                err = cudaMalloc(&buffer, output_image_buffer_size);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to allocate CUDA memory for publishing blur image.");
                    return;
                }

                err = cudaMemcpy(buffer, output_image_buffer_.basePtr, output_image_buffer_size, cudaMemcpyDefault);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to copy crop image buffer to nitros output.");
                    cudaFree(buffer);
                    return;
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