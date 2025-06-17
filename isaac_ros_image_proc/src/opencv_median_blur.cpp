#include "isaac_ros_image_proc/opencv_median_blur.hpp"

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

            OpencvMedianBlurNode::OpencvMedianBlurNode(const rclcpp::NodeOptions options)
                : Node("median_blur_node", options)
            {
                // Initialize params from generate parameter library
                param_listener_ = std::make_shared<opencv_median_blur_node::ParamListener>(this->get_node_parameters_interface());
                params_ = param_listener_->get_params();

                // Create a subscriber
                nitros_sub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                    nvidia::isaac_ros::nitros::NitrosImageView>>(
                    this, params_.image_sub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name,
                    std::bind(&OpencvMedianBlurNode::input_callback, this, std::placeholders::_1)); // diagnostics config and qos TODO

                // Create a publisher
                nitros_pub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                    nvidia::isaac_ros::nitros::NitrosImage>>(
                    this, params_.image_pub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name); // diagnostics config and qos TODO

                CheckCudaErrors(cudaStreamCreate(&stream_), __FILE__, __LINE__);

                int kernel_size_ = params_.kernel_width_; // Assume square kernel
                median_filter_ = cv::cuda::createMedianFilter(CV_8UC1, kernel_size_);

                // create morphology filters
                mask_dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
                mask_close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                dilate_filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, mask_dilate_kernel);
                close_filter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, mask_close_kernel);

                frame_rate_display_ = 0.0;
            }

            void OpencvMedianBlurNode::input_callback(const nvidia::isaac_ros::nitros::NitrosImageView &view)
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

                // Wrap input GPU memory as GpuMat
                cv::cuda::GpuMat input_image(input_image_height_, input_image_width_, CV_8UC3, const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(view.GetGpuData())));

                // std::vector<cv::cuda::GpuMat> im_channels;
                // cv::cuda::split(input_image, im_channels);
                // cv::cuda::GpuMat blurred_r, blurred_g, blurred_b;
                // median_filter_->apply(im_channels[0], blurred_r);
                // median_filter_->apply(im_channels[1], blurred_g);
                // median_filter_->apply(im_channels[2], blurred_b);

                // Convert to HSV
                
                cv::cuda::cvtColor(input_image, input_hsv, cv::COLOR_BGR2HSV);
                
                cv::cuda::split(input_hsv, hsv_channels);
                
                dilate_filter->apply(hsv_channels[0], blurred_h);
                close_filter->apply(blurred_h, blurred_h);
                // median_filter_->apply(blurred_h, blurred_h);

                // median_filter_->apply(hsv_channels[0], blurred_h);
                // median_filter_->apply(hsv_channels[1], blurred_s);
                // median_filter_->apply(hsv_channels[2], blurred_v);
                // cv::cuda::GpuMat blurred_hsv;
                // cv::cuda::merge(std::vector<cv::cuda::GpuMat>{blurred_h, blurred_s, blurred_v}, blurred_hsv);

                // // Convert to grayscale
                // cv::cuda::GpuMat input_gray;
                // cv::cuda::cvtColor(input_image, input_gray, cv::COLOR_BGR2GRAY);

                // // Apply median filter
                // cv::cuda::GpuMat blurred_gray;
                // median_filter_->apply(input_gray, blurred_gray);

                // // Allocate the memory buffer for output ourselves
                // uint8_t *raw_output_image_buffer{nullptr};
                // const size_t output_image_buffer_size_ = output_image_width_ * output_image_height_ * output_image_channels_ * sizeof(uint8_t);
                // CheckCudaErrors(cudaMallocAsync(&raw_output_image_buffer, params_.batch_size_ * output_image_buffer_size_, stream_), __FILE__, __LINE__);

                // // Convert back to BGR
                // cv::cuda::GpuMat output_image(output_image_height_, output_image_width_, CV_8UC3, raw_output_image_buffer);
                // cv::cuda::cvtColor(blurred_gray, output_image, cv::COLOR_GRAY2BGR);

                CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__);

                // // Create a NitrosImage message and publish it
                // std_msgs::msg::Header header_;
                // header_.frame_id = view.GetFrameId();
                // header_.stamp.sec = view.GetTimestampSeconds();
                // header_.stamp.nanosec = view.GetTimestampNanoseconds();

                // nvidia::isaac_ros::nitros::NitrosImage output_image_msg_ =
                //     nvidia::isaac_ros::nitros::NitrosImageBuilder()
                //         .WithHeader(header_)
                //         .WithDimensions(output_image_height_, output_image_width_)
                //         .WithEncoding(sensor_msgs::image_encodings::BGR8)
                //         .WithGpuData(raw_output_image_buffer)
                //         .Build();

                // nitros_pub_ptr_->publish(output_image_msg_);

                // sensor_msgs::msg::Image img_msg;
                // img_msg.header.frame_id = view.GetFrameId();
                // img_msg.header.stamp.sec = view.GetTimestampSeconds();
                // img_msg.header.stamp.nanosec = view.GetTimestampNanoseconds();
                // img_msg.height = output_image_height_;
                // img_msg.width = output_image_width_;
                // img_msg.encoding = sensor_msgs::image_encodings::BGR8;
                // img_msg.step = output_image_buffer_size_ / output_image_height_;

                // img_msg.data.resize(output_image_buffer_size_);
                // cudaMemcpy(img_msg.data.data(), output_image_buffer_.basePtr, output_image_buffer_size_, cudaMemcpyDefault);

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

                    // rclcpp::Time current_time = this->now();
                    // static rclcpp::Time last_time_ = this->now();
                    // double frame_rate = 1.0 / (current_time - last_time_).seconds();
                    // RCLCPP_INFO(this->get_logger(), "Published image %zu at Frame rate: %.2f Hz", image_index_, frame_rate);
                    // last_time_ = current_time;

                    // // Convert to OpenCV Mat
                    // cv::Mat image(output_image_height_, output_image_width_, CV_8UC3, img_msg.data.data());
                    // // Show image
                    // cv::Mat resized_image;
                    // cv::resize(image, resized_image, cv::Size(), params_.resize_factor, params_.resize_factor);

                    // // Watermark framerate text
                    // std::string text = cv::format("FPS: %.2f", frame_rate_display_);
                    // int font = cv::FONT_HERSHEY_SIMPLEX;
                    // cv::Point origin(params_.watermark_x, params_.watermark_y);
                    // cv::putText(resized_image, text, origin, font, params_.watermark_scale, CV_RGB(220, 20, 60), params_.watermark_thickness);

                    // cv::imshow("Image Viewer 2", resized_image);
                    // cv::waitKey(1);
                }
            }

            OpencvMedianBlurNode::~OpencvMedianBlurNode()
            {
                CheckCudaErrors(cudaStreamDestroy(stream_), __FILE__, __LINE__);
            }

        } // namespace image_proc
    } // namespace isaac_ros
} // namespace nvidia

// Register as a component.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::OpencvMedianBlurNode)