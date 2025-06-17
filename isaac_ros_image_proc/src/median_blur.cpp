#include "isaac_ros_image_proc/median_blur.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"

#include <nvcv/Size.hpp>

#include <opencv2/opencv.hpp>

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

            MedianBlurNode::MedianBlurNode(const rclcpp::NodeOptions options)
                : Node("median_blur_node", options),
                  median_blur_op_(max_var_shape_batch_size_), // Initialize MedianBlur operator using the constant
                  frame_rate_display_(0.0)
            {
                // Initialize params from generate parameter library
                param_listener_ = std::make_shared<median_blur_node::ParamListener>(this->get_node_parameters_interface());
                params_ = param_listener_->get_params();

                // Create a subscriber
                nitros_sub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                    nvidia::isaac_ros::nitros::NitrosImageView>>(
                    this, params_.image_sub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name,
                    std::bind(&MedianBlurNode::input_callback, this, std::placeholders::_1)); // diagnostics config and qos TODO

                // Create a publisher
                nitros_pub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                    nvidia::isaac_ros::nitros::NitrosImage>>(
                    this, params_.image_pub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name); // diagnostics config and qos TODO

                CheckCudaErrors(cudaStreamCreate(&stream_), __FILE__, __LINE__);
            }

            void MedianBlurNode::input_callback(const nvidia::isaac_ros::nitros::NitrosImageView &view)
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
                const int output_image_channels_ = input_image_channels_;

                // Create a buffer for the input image
                nvcv::TensorDataStridedCuda::Buffer input_image_buffer_;
                input_image_buffer_.strides[3] = sensor_msgs::image_encodings::bitDepth(view.GetEncoding()) / CHAR_BIT;
                input_image_buffer_.strides[2] = input_image_channels_ * input_image_buffer_.strides[3];
                input_image_buffer_.strides[1] = view.GetStride();
                input_image_buffer_.strides[0] = input_image_height_ * input_image_buffer_.strides[1];

                input_image_buffer_.basePtr = const_cast<NVCVByte *>(reinterpret_cast<const NVCVByte *>(view.GetGpuData()));

                nvcv::Tensor::Requirements input_image_reqs_ = nvcv::Tensor::CalcRequirements(
                    params_.batch_size_,
                    {static_cast<int32_t>(view.GetWidth()), static_cast<int32_t>(view.GetHeight())},
                    nvcv::FMT_BGR8); // write a function to get the NVCV format from the image encoding TODO

                nvcv::TensorDataStridedCuda input_image_data_{
                    nvcv::TensorShape{input_image_reqs_.shape, input_image_reqs_.rank, input_image_reqs_.layout},
                    nvcv::DataType{input_image_reqs_.dtype}, input_image_buffer_};

                nvcv::Tensor input_image_tensor_{nvcv::TensorWrapData(input_image_data_)};

                // Allocate the memory buffer for output ourselves rather than letting CV-CUDA allocate it
                uint8_t *raw_output_image_buffer{nullptr};
                const size_t output_image_buffer_size_ = output_image_width_ * output_image_height_ * output_image_channels_ * sizeof(uint8_t);
                CheckCudaErrors(cudaMallocAsync(&raw_output_image_buffer, params_.batch_size_ * output_image_buffer_size_, stream_), __FILE__, __LINE__);

                nvcv::TensorDataStridedCuda::Buffer output_image_buffer_;
                output_image_buffer_.strides[3] = sizeof(uint8_t);
                output_image_buffer_.strides[2] = output_image_channels_ * output_image_buffer_.strides[3];
                output_image_buffer_.strides[1] = output_image_width_ * output_image_buffer_.strides[2];
                output_image_buffer_.strides[0] = output_image_height_ * output_image_buffer_.strides[1];

                output_image_buffer_.basePtr = reinterpret_cast<NVCVByte *>(raw_output_image_buffer);

                nvcv::Tensor::Requirements output_image_reqs_ = nvcv::Tensor::CalcRequirements(
                    params_.batch_size_,
                    {static_cast<int32_t>(view.GetWidth()), static_cast<int32_t>(view.GetHeight())},
                    nvcv::FMT_BGR8); // write a function to get the NVCV format from the image encoding TODO

                nvcv::TensorDataStridedCuda output_image_data_{
                    nvcv::TensorShape{output_image_reqs_.shape, output_image_reqs_.rank, output_image_reqs_.layout},
                    nvcv::DataType{output_image_reqs_.dtype}, output_image_buffer_};

                nvcv::Tensor output_image_tensor_{nvcv::TensorWrapData(output_image_data_)};

                // Median blur processing.
                nvcv::Size2D kernel_size_(static_cast<int32_t>(params_.kernel_width_), static_cast<int32_t>(params_.kernel_height_));
                median_blur_op_(stream_, input_image_tensor_, output_image_tensor_, kernel_size_);

                CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__);

                // Create a NitrosImage message and publish it
                std_msgs::msg::Header header_;
                header_.frame_id = view.GetFrameId();
                header_.stamp.sec = view.GetTimestampSeconds();
                header_.stamp.nanosec = view.GetTimestampNanoseconds();

                nvidia::isaac_ros::nitros::NitrosImage output_image_msg_ =
                    nvidia::isaac_ros::nitros::NitrosImageBuilder()
                        .WithHeader(header_)
                        .WithDimensions(output_image_height_, output_image_width_)
                        .WithEncoding(sensor_msgs::image_encodings::BGR8)
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
                    // end of frame rate calculation
                }
                // Display the image if required
                if (params_.bimage_display_)
                {
                    sensor_msgs::msg::Image img_msg;
                    img_msg.header.frame_id = view.GetFrameId();
                    img_msg.header.stamp.sec = view.GetTimestampSeconds();
                    img_msg.header.stamp.nanosec = view.GetTimestampNanoseconds();
                    img_msg.height = output_image_height_;
                    img_msg.width = output_image_width_;
                    img_msg.encoding = sensor_msgs::image_encodings::BGR8;
                    img_msg.step = output_image_buffer_size_ / output_image_height_;

                    img_msg.data.resize(output_image_buffer_size_);
                    cudaError_t cuda_status = cudaMemcpy(img_msg.data.data(), output_image_buffer_.basePtr, output_image_buffer_size_, cudaMemcpyDefault);
                    CheckCudaErrors(cuda_status, __FILE__, __LINE__);

                    cv::Mat image(output_image_height_, output_image_width_, CV_8UC3, img_msg.data.data());
                    cv::Mat resized_image;
                    cv::resize(image, resized_image, cv::Size(), params_.resize_factor, params_.resize_factor);

                    if (params_.bframe_rate_display_)
                    {
                        std::string text = cv::format("FPS: %.2f", frame_rate_display_);
                        int font = cv::FONT_HERSHEY_SIMPLEX;
                        cv::Point origin(params_.watermark_x, params_.watermark_y);
                        cv::putText(resized_image, text, origin, font, params_.watermark_scale, CV_RGB(220, 20, 60), params_.watermark_thickness);
                    }

                    cv::imshow("Image Viewer 2", resized_image);
                    cv::waitKey(1);
                }
            }

            MedianBlurNode::~MedianBlurNode()
            {
                CheckCudaErrors(cudaStreamDestroy(stream_), __FILE__, __LINE__);
            }

        } // namespace image_proc
    } // namespace isaac_ros
} // namespace nvidia

// Register as a component.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::MedianBlurNode)
