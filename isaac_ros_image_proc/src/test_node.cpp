#include "isaac_ros_image_proc/test_node.hpp"

#include <chrono>
#include <functional>
#include <iostream>
#include <fmt/format.h>

namespace nvidia
{
    namespace isaac_ros
    {
        namespace image_proc
        {

            TestNode::TestNode(const rclcpp::NodeOptions &options)
                : Node("test_node", options),
                  medianBlurOp_(maxVarShapeBatchSize) // Initialize MedianBlur operator 
            {
                // param_listener_ = std::make_shared<ParamListener>(this->get_node_parameters_interface());
                // params_ = param_listener_->get_params();

                // Create a CUDA stream for kernel execution.
                cudaError_t streamErr = cudaStreamCreate(&stream);
                if (streamErr != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to create CUDA stream!");
                }

                // Create a subscriber that listens to "image_raw" and calls InputCallback.
                nitros_sub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                    nvidia::isaac_ros::nitros::NitrosImageView>>(
                    this, "image_raw",
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name,
                    std::bind(&TestNode::InputCallback, this, std::placeholders::_1));

                // Create a publisher to publish the cropped image.
                crop_nitros_pub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                    nvidia::isaac_ros::nitros::NitrosImage>>(
                    this, "crop_image",
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name);

                // Set image and processing parameters.
                batch_size_ = 1;
                cropX = 50;
                cropY = 50;
                cropWidth = 1500;
                cropHeight = 1500;
                input_image_channels_ = 3;
                input_image_width_ = 2448;
                input_image_height_ = 1840;
                kernelWidth = 11; // blur kernel size
                kernelHeight = 11;

                // Calculate tensor requirements for the input image.
                nvcv::Tensor::Requirements reqs = nvcv::Tensor::CalcRequirements(
                    batch_size_,
                    {input_image_width_, input_image_height_},
                    nvcv::FMT_BGR8);

                // Allocate input buffer and setup strides.
                input_image_buffer_.strides[3] = sizeof(uint8_t);
                input_image_buffer_.strides[2] = input_image_channels_ * input_image_buffer_.strides[3];
                input_image_buffer_.strides[1] = input_image_width_ * input_image_buffer_.strides[2];
                input_image_buffer_.strides[0] = input_image_height_ * input_image_buffer_.strides[1];

                cudaError_t err = cudaMalloc(&input_image_buffer_.basePtr,
                                             batch_size_ * input_image_buffer_.strides[0]);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to allocate CUDA memory for input image.");
                }

                nvcv::TensorDataStridedCuda in_data(
                    nvcv::TensorShape{reqs.shape, reqs.rank, reqs.layout},
                    nvcv::DataType{reqs.dtype},
                    input_image_buffer_);
                input_image_tensor_ = nvcv::TensorWrapData(in_data);

                // Allocate and initialize buffer for the median-blurred image.
                blur_image_buffer_.strides[3] = sizeof(uint8_t);
                blur_image_buffer_.strides[2] = input_image_channels_ * blur_image_buffer_.strides[3];
                blur_image_buffer_.strides[1] = input_image_width_ * blur_image_buffer_.strides[2];
                blur_image_buffer_.strides[0] = input_image_height_ * blur_image_buffer_.strides[1];

                err = cudaMalloc(&blur_image_buffer_.basePtr,
                                 batch_size_ * blur_image_buffer_.strides[0]);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to allocate CUDA memory for blur image.");
                }

                nvcv::TensorDataStridedCuda blur_data(
                    nvcv::TensorShape{reqs.shape, reqs.rank, reqs.layout},
                    nvcv::DataType{reqs.dtype},
                    blur_image_buffer_);
                blur_image_tensor_ = nvcv::TensorWrapData(blur_data);

                // Calculate requirements and allocate buffer for the cropped image.
                reqs = nvcv::Tensor::CalcRequirements(batch_size_,
                                                      {cropWidth, cropHeight}, nvcv::FMT_BGR8);

                crop_image_buffer_.strides[3] = sizeof(uint8_t);
                crop_image_buffer_.strides[2] = input_image_channels_ * crop_image_buffer_.strides[3];
                crop_image_buffer_.strides[1] = cropWidth * crop_image_buffer_.strides[2];
                crop_image_buffer_.strides[0] = cropHeight * crop_image_buffer_.strides[1];

                err = cudaMalloc(&crop_image_buffer_.basePtr,
                                 batch_size_ * crop_image_buffer_.strides[0]);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to allocate CUDA memory for crop image.");
                }

                nvcv::TensorDataStridedCuda out_data(
                    nvcv::TensorShape{reqs.shape, reqs.rank, reqs.layout},
                    nvcv::DataType{reqs.dtype},
                    crop_image_buffer_);
                crop_image_tensor_ = nvcv::TensorWrapData(out_data);
            }

            TestNode::~TestNode()
            {
                // Destroy the CUDA stream.
                cudaStreamDestroy(stream);

                // Free the allocated CUDA buffers if needed.
                if (input_image_buffer_.basePtr)
                {
                    cudaFree(input_image_buffer_.basePtr);
                    input_image_buffer_.basePtr = nullptr;
                }
                if (blur_image_buffer_.basePtr)
                {
                    cudaFree(blur_image_buffer_.basePtr);
                    blur_image_buffer_.basePtr = nullptr;
                }
                if (crop_image_buffer_.basePtr)
                {
                    cudaFree(crop_image_buffer_.basePtr);
                    crop_image_buffer_.basePtr = nullptr;
                }
            }

            void TestNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosImageView &view)
            {
                // Copy the incoming Nitros image GPU data into the input tensor.
                size_t buffer_size = view.GetSizeInBytes();
                cudaError_t err = cudaMemcpy(input_image_buffer_.basePtr, view.GetGpuData(),
                                             buffer_size, cudaMemcpyDefault);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to copy nitros input image to input image buffer.");
                    return;
                }

                // Median blur processing.
                nvcv::Size2D kSize_(kernelWidth, kernelHeight);
                medianBlurOp_(stream, input_image_tensor_, blur_image_tensor_, kSize_);

                // Cropping.
                crpRect_ = {cropX, cropY, cropWidth, cropHeight};
                cropOp_(stream, blur_image_tensor_, crop_image_tensor_, crpRect_);
                RCLCPP_INFO(this->get_logger(), "Cropping done...");

                // Prepare a temporary CUDA buffer for publishing the cropped image.
                size_t crop_image_buffer_size = cropWidth * cropHeight * input_image_channels_ * sizeof(uint8_t);
                void *buffer = nullptr;
                err = cudaMalloc(&buffer, crop_image_buffer_size);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to allocate CUDA memory for publishing crop image.");
                    return;
                }

                err = cudaMemcpy(buffer, crop_image_buffer_.basePtr, crop_image_buffer_size, cudaMemcpyDefault);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to copy crop image buffer to nitros output.");
                    cudaFree(buffer);
                    return;
                }
                RCLCPP_INFO(this->get_logger(), "Copying crop image to buffer done...");

                // Build and publish the Nitros image.
                std_msgs::msg::Header crop_header;
                crop_header.stamp = this->now();
                crop_header.frame_id = "crop_image";

                nvidia::isaac_ros::nitros::NitrosImage crop_nitros_image =
                    nvidia::isaac_ros::nitros::NitrosImageBuilder()
                        .WithHeader(crop_header)
                        .WithEncoding(sensor_msgs::image_encodings::BGR8)
                        .WithDimensions(cropHeight, cropWidth) // height, width of the crop
                        .WithGpuData(buffer)
                        .Build();

                crop_nitros_pub_ptr_->publish(crop_nitros_image);
                RCLCPP_INFO(this->get_logger(), "Published cropped Nitros image with GPU pointer: %p", buffer);

                // Optionally free the temporary publishing buffer if the publisher does not take ownership.
                // cudaFree(buffer);
            }

        } // namespace image_proc
    } // namespace isaac_ros
} // namespace nvidia

// Register as a component.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::TestNode)
