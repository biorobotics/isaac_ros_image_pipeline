#include "isaac_ros_image_proc/detect_laser.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"

#include <vector>

// Declare the CUDA kernel
extern "C" __global__ void detect_laser_stripe_kernel(
    const unsigned char *, int, int, size_t,
    float, float, float,
    float *, float *, int *);

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

            DetectLaserNode::DetectLaserNode(const rclcpp::NodeOptions options)
                : Node("detect_laser_node", options),
                  frame_rate_display_(0.0)
            {
                // Initialize params from generate parameter library
                param_listener_ = std::make_shared<detect_laser_node::ParamListener>(this->get_node_parameters_interface());
                params_ = param_listener_->get_params();

                // Create a subscriber
                nitros_sub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
                    nvidia::isaac_ros::nitros::NitrosImageView>>(
                    this, params_.image_sub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name,
                    std::bind(&DetectLaserNode::input_callback, this, std::placeholders::_1)); // diagnostics config and qos TODO

                // Create a publisher
                nitros_pub_ptr_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
                    nvidia::isaac_ros::nitros::NitrosImage>>(
                    this, params_.image_pub_topic_,
                    nvidia::isaac_ros::nitros::nitros_image_bgr8_t::supported_type_name); // diagnostics config and qos TODO

                laser_roi_ = cv::Rect(0, 0, 1280, 720); // Example ROI, adjust as needed

                // Camera intrinsics
                camera_matrix_ = (cv::Mat_<double>(3, 3) << params_.camera_fx, 0, params_.camera_cx, 0,
                                  params_.camera_fy, params_.camera_cy, 0, 0, 1);
                dist_coeffs_ = (cv::Mat_<double>(1, 5) << params_.camera_k1, params_.camera_k2, params_.camera_p1,
                                params_.camera_p2, params_.camera_k3);

                plane_ = Eigen::Vector4d(params_.laser_plane[0], params_.laser_plane[1], params_.laser_plane[2],
                                         params_.laser_plane[3]);

                CheckCudaErrors(cudaStreamCreate(&stream_), __FILE__, __LINE__);
            }

            void DetectLaserNode::input_callback(const nvidia::isaac_ros::nitros::NitrosImageView &view)
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
                cv::cuda::GpuMat d_im_v(input_image_height_, input_image_width_, CV_8UC1, const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(view.GetGpuData())));

                // Device buffers
                int rows = d_im_v.rows, cols = d_im_v.cols;
                size_t step = d_im_v.step;
                float *d_out_x = nullptr, *d_out_y = nullptr;
                int *d_count = nullptr;
                CheckCudaErrors(cudaMalloc(&d_out_x, sizeof(float) * cols), __FILE__, __LINE__);
                CheckCudaErrors(cudaMalloc(&d_out_y, sizeof(float) * cols), __FILE__, __LINE__);
                CheckCudaErrors(cudaMalloc(&d_count, sizeof(int)), __FILE__, __LINE__);
                CheckCudaErrors(cudaMemsetAsync(d_count, 0, sizeof(int), stream_), __FILE__, __LINE__);

                // Launch kernel
                const int threads = 128;
                const int blocks = (cols + threads - 1) / threads;
                detect_laser_stripe_kernel<<<blocks, threads, 0, stream_>>>(
                    d_im_v.ptr<uchar>(), rows, cols, step,
                    float(laser_roi_.x), float(laser_roi_.y), 0.8f,
                    d_out_x, d_out_y, d_count);
                CheckCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
                CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__);

                // Copy back count
                int h_count = 0;
                CheckCudaErrors(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

                // Copy back output points
                std::vector<float> h_x(h_count), h_y(h_count);
                if (h_count > 0)
                {
                    CheckCudaErrors(cudaMemcpy(h_x.data(), d_out_x, sizeof(float) * h_count, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
                    CheckCudaErrors(cudaMemcpy(h_y.data(), d_out_y, sizeof(float) * h_count, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
                }

                // Free device buffers
                cudaFree(d_out_x);
                cudaFree(d_out_y);
                cudaFree(d_count);

                // Populate laser_points_
                laser_points_.clear();
                laser_points_.reserve(h_count);
                for (int i = 0; i < h_count; ++i)
                {
                    laser_points_.emplace_back(h_x[i], h_y[i]);
                }

                // Reject small segments
                if (static_cast<int>(laser_points_.size()) < params_.min_segment_points)
                {
                    laser_points_.clear();
                }

                if (laser_points_.empty())
                    return;

                std::vector<cv::Point2f> laser_uv_norm;
                laser_uv_norm.reserve(laser_points_.size());
                cv::undistortPoints(laser_points_, laser_uv_norm, camera_matrix_, dist_coeffs_);
                // Generate point cloud
                auto laser_point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                // Convert normalized points to 3D
                pts_norm_to_3d(laser_uv_norm, laser_point_cloud);

                // ----------------------------------
                CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__); // Is this needed?

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
            }

            void DetectLaserNode::pts_norm_to_3d(const std::vector<cv::Point2f> &pts_norm,
                                                 pcl::PointCloud<pcl::PointXYZ>::Ptr pts_3d)
            {
                pts_3d->clear();
                pts_3d->reserve(pts_norm.size());

                for (const auto &pt : pts_norm)
                {
                    double x = pt.x;
                    double y = pt.y;
                    double denom = x * plane_(0) + y * plane_(1) + plane_(2);
                    if (std::abs(denom) > 1e-9)
                    {
                        double z = -plane_(3) / denom;
                        pts_3d->emplace_back(static_cast<float>(x * z), static_cast<float>(y * z),
                                             static_cast<float>(z));
                    }
                }

                pts_3d->width = pts_3d->size();
                pts_3d->height = 1;
                pts_3d->is_dense = true;
            }

            DetectLaserNode::~DetectLaserNode()
            {
                CheckCudaErrors(cudaStreamDestroy(stream_), __FILE__, __LINE__);
            }

        } // namespace image_proc
    } // namespace isaac_ros
} // namespace nvidia

// Register as a component.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::DetectLaserNode)