#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZRGB = pcl::PointCloud<pcl::PointXYZRGB>;

class PointCloudToDepthConverter
{
public:

    struct CameraParams
    {
        int image_width;
        int image_height;
        double A11, A12, A22;          
        double u0, v0;                 
        double k2, k3, k4, k5, k6, k7; 
        double scale;                  
        int point_sampling_rate;      
        Eigen::Matrix4d Tcl;       
    };

    struct ProcessResult
    {
        cv::Mat depth_image;
        pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;
        bool success;
        std::string error_message;
    };

    explicit PointCloudToDepthConverter(const CameraParams &params);

    ProcessResult processCloudAndImage(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                                       const cv::Mat &image);

	cv::Mat customResize(const cv::Mat& src, const cv::Size& size);
    const CameraParams &getCameraParams() const { return params_; }

    void updateCameraParams(const CameraParams &params);

private:
    CameraParams params_;

    double blind_rgb_points = 0.1;

    Eigen::Matrix3d K_;
    Eigen::Matrix3d Kl_;
    Eigen::Matrix4d K_4x4_;
    Eigen::Matrix4d Kcl_;

    cv::Mat map_x_, map_y_;
    cv::Mat inv_map_x_, inv_map_y_;

    int scaled_width_, scaled_height_;

    void initializeInternalParams();

    void createDistortionMaps();

    cv::Mat projectCloudToDepth(const pcl::PointCloud<pcl::PointXYZ> &cloud_in_cam);

    cv::Mat postProcessDepthImage(const cv::Mat &depth_img);

    // pcl::PointCloud<pcl::PointXYZRGB> generateColoredCloud(const cv::Mat &depth_img,
    //                                                        const cv::Mat &color_img);

    pcl::PointCloud<pcl::PointXYZRGB> generateColoredCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud_in_cam, 
                                                           const cv::Mat &color_img);
    
    Eigen::Vector3f getInterpolatedPixel(cv::Mat img, 
                                         Eigen::Vector2d pc);
    
    PointCloudXYZ::Ptr pcl_wait_pub;

    std::pair<bool, std::string> validateInputs(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                                                const cv::Mat &image);
};
