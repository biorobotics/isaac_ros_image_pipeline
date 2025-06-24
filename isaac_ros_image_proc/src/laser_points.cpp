#include "isaac_ros_image_proc/laser_points.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"

#include <vector>

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
                cv::cuda::GpuMat d_im_v(input_image_height_, input_image_width_, CV_8UC1, const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(view.GetGpuData())));

                laser_points_.clear();
                laser_points_.reserve(static_cast<size_t>(laser_roi_.width));

                for (int cc = 0; cc < d_im_v.cols; ++cc)
                {
                    // grab one column on the device
                    cv::cuda::GpuMat d_col = d_im_v.col(cc);

                    // compute min/max on GPU
                    double minVal, maxVal;
                    cv::Point minLoc, maxLoc;
                    cv::cuda::minMaxLoc(d_col, &minVal, &maxVal, &minLoc, &maxLoc);

                    // Download that single column for the 80%-threshold & weighted-average steps
                    cv::Mat colHost;
                    d_col.download(colHost);

                    // Find the 80%-band around the peak
                    int j = maxLoc.y - 1;
                    int k = maxLoc.y + 1;
                    const auto threshold = 0.8 * maxVal;
                    while (j >= 0 && colHost.at<uchar>(j, 0) > threshold)
                        --j;
                    while (k < colHost.rows && colHost.at<uchar>(k, 0) > threshold)
                        ++k;

                    // Compute weighted centroid within [j+1, k)
                    double weightedSum = 0.0, valSum = 0.0;
                    for (int rr = j + 1; rr < k; ++rr)
                    {
                        double v = colHost.at<uchar>(rr, 0);
                        weightedSum += v * (rr - j);
                        valSum += v;
                    }

                    if (valSum > 0.0)
                    {
                        float y = static_cast<float>(weightedSum / valSum + j) + laser_roi_.y;
                        float x = static_cast<float>(cc) + laser_roi_.x;
                        laser_points_.emplace_back(x, y);
                    }
                }

                // Reject too‐small segments
                if (static_cast<int>(laser_points_.size()) < params_.min_segment_points)
                {
                    laser_points_.clear();
                }
                if (laser_points_.empty())
                {
                    return; // No points found, skip processing
                }

                std::vector<cv::Point2f> laser_uv_norm;
                laser_uv_norm.reserve(laser_points_.size());
                cv::undistortPoints(laser_points_, laser_uv_norm, camera_matrix_, dist_coeffs_);
                // Generate point cloud
                auto laser_point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                // Convert normalized points to 3D
                pts_norm_to_3d(laser_uv_norm, laser_point_cloud);

                // ----------------------------------
                CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__); // Is this needed?

                // // Create a NitrosImage message and publish it
                // std_msgs::msg::Header header_;
                // header_.frame_id = view.GetFrameId();
                // header_.stamp.sec = view.GetTimestampSeconds();
                // header_.stamp.nanosec = view.GetTimestampNanoseconds();

                // nvidia::isaac_ros::nitros::NitrosImage output_image_msg_ =
                //     nvidia::isaac_ros::nitros::NitrosImageBuilder()
                //         .WithHeader(header_)
                //         .WithDimensions(output_image_height_, output_image_width_)
                //         .WithEncoding(sensor_msgs::image_encodings::MONO8) // Assuming output is single channel
                //         .WithGpuData(raw_output_image_buffer)
                //         .Build();

                // nitros_pub_ptr_->publish(output_image_msg_);

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

            void LaserPointsNode::pts_norm_to_3d(const std::vector<cv::Point2f> &pts_norm,
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