#include "pointcloud_depth_converter.hpp"
#include <cmath>
#include <iostream>

PointCloudToDepthConverter::PointCloudToDepthConverter(const CameraParams &params)
    : params_(params)
{
    initializeInternalParams();
    createDistortionMaps();
}

void PointCloudToDepthConverter::initializeInternalParams()
{
    scaled_width_ = static_cast<int>(params_.image_width / params_.scale);
    scaled_height_ = static_cast<int>(params_.image_height / params_.scale);

    K_ = Eigen::Matrix3d::Identity();
    K_(0, 0) = params_.A11;
    K_(0, 1) = params_.A12;
    K_(0, 2) = params_.u0;
    K_(1, 1) = params_.A22;
    K_(1, 2) = params_.v0;

    Kl_ = Eigen::Matrix3d::Identity();
    Kl_(0, 0) = params_.A11 / params_.scale;
    Kl_(0, 1) = 0.0;
    Kl_(0, 2) = params_.u0 / params_.scale;
    Kl_(1, 1) = params_.A22 / params_.scale;
    Kl_(1, 2) = params_.v0 / params_.scale;

    K_4x4_ = Eigen::Matrix4d::Identity();
    K_4x4_.block<3, 3>(0, 0) = Kl_;

    Kcl_ = K_4x4_ * params_.Tcl;
}

void PointCloudToDepthConverter::createDistortionMaps()
{
    map_x_ = cv::Mat::zeros(params_.image_height, params_.image_width, CV_32FC1);
    map_y_ = cv::Mat::zeros(params_.image_height, params_.image_width, CV_32FC1);

    for (int u = 0; u < params_.image_width; ++u)
    {
        for (int v = 0; v < params_.image_height; ++v)
        {
            double y = (v - params_.v0) / params_.A22;
            double x = (u - params_.u0 - params_.A12 * y) / params_.A11;
            
            double r = sqrt(x * x + y * y);
            double theta = atan(r);

            double theta_d = theta + params_.k2 * pow(theta, 2) + params_.k3 * pow(theta, 3) +
                                params_.k4 * pow(theta, 4) + params_.k5 * pow(theta, 5) +
                                params_.k6 * pow(theta, 6) + params_.k7 * pow(theta, 7);

            double x_distorted = x * (r / theta_d);
            double y_distorted = y * (r / theta_d);

            map_x_.at<float>(v, u) = static_cast<float>(x_distorted * params_.A11 + params_.A12 * y_distorted + params_.u0);
            map_y_.at<float>(v, u) = static_cast<float>(y_distorted * params_.A22 + params_.v0);
        }
    }

    inv_map_x_ = cv::Mat::zeros(params_.image_height, params_.image_width, CV_32FC1);
    inv_map_y_ = cv::Mat::zeros(params_.image_height, params_.image_width, CV_32FC1);

    for (int u = 0; u < params_.image_width; ++u)
    {
        for (int v = 0; v < params_.image_height; ++v)
        {
            double y = (v - params_.v0) / params_.A22;
            double x = (u - params_.u0 - params_.A12 * y) / params_.A11;
            
            double r = sqrt(x * x + y * y);
            double theta = atan(r);

            double theta_d = theta + params_.k2 * pow(theta, 2) + params_.k3 * pow(theta, 3) +
                                params_.k4 * pow(theta, 4) + params_.k5 * pow(theta, 5) +
                                params_.k6 * pow(theta, 6) + params_.k7 * pow(theta, 7);

            double x_distorted = x * (theta_d / r);
            double y_distorted = y * (theta_d / r);

            inv_map_x_.at<float>(v, u) = static_cast<float>(x_distorted * params_.A11 + params_.A12 * y_distorted + params_.u0);
            inv_map_y_.at<float>(v, u) = static_cast<float>(y_distorted * params_.A22 + params_.v0);
        }
    }
}

PointCloudToDepthConverter::ProcessResult PointCloudToDepthConverter::processCloudAndImage(
    const pcl::PointCloud<pcl::PointXYZ> &cloud,
    const cv::Mat &image)
{
    ProcessResult result;
    result.success = false;

    auto validation_result = validateInputs(cloud, image);
    if (!validation_result.first)
    {
        result.error_message = validation_result.second;
        return result;
    }

    try
    {
        pcl::PointCloud<pcl::PointXYZ> cloud_in_cam;
        pcl::transformPointCloud(cloud, cloud_in_cam, Kcl_);

        cv::Mat depth_img = projectCloudToDepth(cloud_in_cam);

        cv::Mat processed_depth = postProcessDepthImage(depth_img);

        pcl::PointCloud<pcl::PointXYZRGB> colored_cloud = generateColoredCloud(cloud_in_cam, image);

        result.depth_image = processed_depth;
        result.colored_cloud = colored_cloud;
        result.success = true;
    }
    catch (const std::exception &e)
    {
        result.error_message = std::string("Processing error: ") + e.what();
    }

    return result;
}

cv::Mat PointCloudToDepthConverter::projectCloudToDepth(const pcl::PointCloud<pcl::PointXYZ> &cloud_in_cam)
{
    cv::Mat depth_img = cv::Mat::zeros(scaled_height_, scaled_width_, CV_32FC1);

    for (const auto &camera_point : cloud_in_cam)
    {
        if (camera_point.z <= 0)
            continue; 

        int u = static_cast<int>(std::round(camera_point.x / camera_point.z));
        int v = static_cast<int>(std::round(camera_point.y / camera_point.z));

        if (u >= 0 && u < scaled_width_ && v >= 0 && v < scaled_height_)
        {
            depth_img.at<float>(v, u) = static_cast<float>(camera_point.z);

            for (int du = -1; du <= 1; ++du)
            {
                for (int dv = -1; dv <= 1; ++dv)
                {
                    int nu = u + du;
                    int nv = v + dv;
                    if (nu >= 0 && nu < scaled_width_ && nv >= 0 && nv < scaled_height_)
                    {
                        if (depth_img.at<float>(nv, nu) == 0.0f)
                        {
                            depth_img.at<float>(nv, nu) = static_cast<float>(camera_point.z);
                        }
                    }
                }
            }
        }
    }

    return depth_img;
}

cv::Mat PointCloudToDepthConverter::postProcessDepthImage(const cv::Mat &depth_img) {
    if (depth_img.empty()) {
        std::cerr << "ERROR: Input depth image is empty!" << std::endl;
        return cv::Mat();
    }
    
    if (depth_img.data == nullptr) {
        std::cerr << "ERROR: Input depth image has null data pointer!" << std::endl;
        return cv::Mat();
    }
    
    if (depth_img.rows <= 0 || depth_img.cols <= 0) {
        std::cerr << "ERROR: Invalid input dimensions: " 
                  << depth_img.rows << "x" << depth_img.cols << std::endl;
        return cv::Mat();
    }
    
    if (params_.image_width <= 0 || params_.image_height <= 0) {
        std::cerr << "ERROR: Invalid target size: " 
                  << params_.image_width << "x" << params_.image_height << std::endl;
        return cv::Mat();
    }

    cv::Mat safe_input = depth_img.clone();
    if (safe_input.empty()) {
        std::cerr << "ERROR: Failed to create safe copy of input image!" << std::endl;
        return cv::Mat();
    }
    

    cv::Mat depth_img_upsampled;
    try {
        depth_img_upsampled = customResize(safe_input, cv::Size(1600, 1296));
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Custom resize failed: " << e.what() << std::endl;
        return cv::Mat();
    }
    
    if (depth_img_upsampled.empty()) {
        std::cerr << "ERROR: Resized image is empty!" << std::endl;
        return cv::Mat();
    }
    
    if (depth_img_upsampled.rows != 1296 || depth_img_upsampled.cols != 1600) {
        std::cerr << "ERROR: Resized image has wrong dimensions: " 
                  << depth_img_upsampled.cols << "x" << depth_img_upsampled.rows
                  << " (expected " << 1600 << "x" << 1296 << ")" << std::endl;
        return cv::Mat();
    }
    

    cv::Mat grad_x, grad_y, grad_magnitude;
    try {
        cv::Sobel(depth_img_upsampled, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(depth_img_upsampled, grad_y, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x, grad_y, grad_magnitude);
    } catch (const cv::Exception& e) {
        std::cerr << "ERROR: Sobel/magnitude failed: " << e.what() << std::endl;
        return cv::Mat();
    }
    
    if (grad_magnitude.type() != CV_32F) {
        std::cerr << "ERROR: grad_magnitude has wrong type: " 
                  << grad_magnitude.type() << " (expected CV_32F)" << std::endl;
        return cv::Mat();
    }
    

    cv::Mat threshold_mask;
    try {
        cv::threshold(grad_magnitude, threshold_mask, 0.75, 1, cv::THRESH_BINARY);
        threshold_mask.convertTo(threshold_mask, CV_8U);
        

        depth_img_upsampled.setTo(0, threshold_mask);
    } catch (const cv::Exception& e) {
        std::cerr << "ERROR: Threshold mask failed: " << e.what() << std::endl;
        return cv::Mat();
    }

    return depth_img_upsampled;
}

cv::Mat PointCloudToDepthConverter::customResize(const cv::Mat& src, const cv::Size& size) {
    if (src.empty()) {
        throw std::runtime_error("Source image is empty");
    }
    
    if (size.width <= 0 || size.height <= 0) {
        throw std::runtime_error("Invalid target size");
    }
    

    cv::Mat dst(size.height, size.width, src.type());
    
    float scale_x = src.cols / static_cast<float>(size.width);
    float scale_y = src.rows / static_cast<float>(size.height);
    
    if (src.channels() != 1 || src.type() != CV_32F) {
        throw std::runtime_error("Unsupported image type - expected single channel float");
    }
    

    for (int y = 0; y < dst.rows; y++) {

        int src_y = static_cast<int>(y * scale_y);
        src_y = std::min(src_y, src.rows - 1);
        
        for (int x = 0; x < dst.cols; x++) {
            int src_x = static_cast<int>(x * scale_x);
            src_x = std::min(src_x, src.cols - 1);
            dst.at<float>(y, x) = src.at<float>(src_y, src_x);
        }
    }
    
    return dst;
}

// // depth_img包含的是相机坐标系下的深度值
// pcl::PointCloud<pcl::PointXYZRGB> PointCloudToDepthConverter::generateColoredCloud(const cv::Mat &depth_img, const cv::Mat &color_img)
// {
//     cv::Mat depth_undistorted, color_undistorted;
//     depth_undistorted = depth_img.clone();
   
//     // 对输入图像进行畸变矫正
//     cv::remap(color_img, color_undistorted, inv_map_x_, inv_map_y_, cv::INTER_LINEAR);

//     pcl::PointCloud<pcl::PointXYZRGB> cloud_colored;

//     Eigen::Matrix4d Tlc = params_.Tcl.inverse();

//     for (int v = 0; v < depth_undistorted.rows; v += params_.point_sampling_rate)
//     {
//         for (int u = 0; u < depth_undistorted.cols; u += params_.point_sampling_rate)
//         {
//             float depth = depth_undistorted.at<float>(v, u);
//             if (depth > 0.1f && depth < 100.0f) 
//             {
//                 double y_cam = (v - params_.v0) * depth / params_.A22;
//                 double x_cam = ((u - params_.u0) * depth  - params_.A12 * y_cam)/ params_.A11;
                
//                 double z_cam = depth;

//                 Eigen::Vector4d point_cam(x_cam, y_cam, z_cam, 1.0);

//                 Eigen::Vector4d point_lidar = Tlc * point_cam;

//                 pcl::PointXYZRGB point;
//                 point.x = static_cast<float>(point_lidar[0]);
//                 point.y = static_cast<float>(point_lidar[1]);
//                 point.z = static_cast<float>(point_lidar[2]);

//                 if (u < color_undistorted.cols && v < color_undistorted.rows)
//                 {
//                     cv::Vec3b color = color_undistorted.at<cv::Vec3b>(v, u);
//                     point.b = color[0]; 
//                     point.g = color[1];
//                     point.r = color[2];
//                 }
//                 else
//                 {
//                     point.r = point.g = point.b = 255;
//                 }

//                 cloud_colored.points.push_back(point);
//             }
//         }
//     }

//     cloud_colored.width = cloud_colored.points.size();
//     cloud_colored.height = 1;
//     cloud_colored.is_dense = false;

//     return cloud_colored;
// }

// // pcl::PointCloud<pcl::PointXYZ> cloud_in_cam;
// // pcl::transformPointCloud(cloud, cloud_in_cam, Kcl_);
// cv::Mat PointCloudToDepthConverter::projectCloudToDepth(const pcl::PointCloud<pcl::PointXYZ> &cloud_in_cam)
// {

//     // 从左到右，列索引是X坐标，对应width
//     // 从上到下，行索引是Y坐标，对应height
//     // scaled_width_ = static_cast<int>(params_.image_width / params_.scale);
//     // scaled_height_ = static_cast<int>(params_.image_height / params_.scale);
//     cv::Mat depth_img = cv::Mat::zeros(scaled_height_, scaled_width_, CV_32FC1);

//     for (const auto &camera_point : cloud_in_cam)
//     {

//         // Z前X右Y下
//         if (camera_point.z <= 0)
//             continue; 
        
//         // 透视投影公式，归一化投影标出像素矩阵
//         int u = static_cast<int>(std::round(camera_point.x / camera_point.z));
//         int v = static_cast<int>(std::round(camera_point.y / camera_point.z));

//         if (u >= 0 && u < scaled_width_ && v >= 0 && v < scaled_height_)
//         {
//             depth_img.at<float>(v, u) = static_cast<float>(camera_point.z);
            
//             // 对深度图做领域填充防止出现空值
//             for (int du = -1; du <= 1; ++du)
//             {
//                 for (int dv = -1; dv <= 1; ++dv)
//                 {
//                     int nu = u + du;
//                     int nv = v + dv;
//                     if (nu >= 0 && nu < scaled_width_ && nv >= 0 && nv < scaled_height_)
//                     {
//                         if (depth_img.at<float>(nv, nu) == 0.0f)
//                         {
//                             depth_img.at<float>(nv, nu) = static_cast<float>(camera_point.z);
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     return depth_img;
// }

pcl::PointCloud<pcl::PointXYZRGB> PointCloudToDepthConverter::generateColoredCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud_in_cam, const cv::Mat &color_img)
{
    // 防御性编程，确保输入有效
    if (cloud_in_cam.empty() || color_img.empty()) 
        return pcl::PointCloud<pcl::PointXYZRGB>();

    pcl_wait_pub.reset(new pcl::PointCloud<pcl::PointXYZ>());
    PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());

    static int pub_num = 1;
    pcl_wait_pub->clear();
    // *pcl_wait_pub += *cloud_in_cam;

    pcl_wait_pub->points.insert(
    pcl_wait_pub->points.end(),
    cloud_in_cam.points.begin(),
    cloud_in_cam.points.end()
    );

    // 处理单帧数据，也可以设置其他步长
    if(pub_num == 1)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();

      // 预分配内存提高性能
      laserCloudWorldRGB->reserve(size);
      cv::Mat img_undistorted;

      // 对输入图像进行畸变矫正
      cv::remap(color_img, img_undistorted, inv_map_x_, inv_map_y_, cv::INTER_LINEAR);

      for (size_t i = 0; i < size; i++)
      {

        // 构建彩色点云容器并在后面赋值RGB属性
        pcl::PointXYZRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;
        
        // 相机坐标系下的点云
        Eigen::Vector3d p_f(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        
        // 筛选相机前方的点云
        if (p_f[2] < 0) continue;

        // 将点投影到RGB图像的像素坐标
        
        // 透视投影公式，归一化投影标出像素矩阵
        // int u = static_cast<int>(std::round(pcl_wait_pub->points[i].x / pcl_wait_pub->points[i].z));
        // int v = static_cast<int>(std::round(pcl_wait_pub->points[i].y / pcl_wait_pub->points[i].z));

        double u = params_.u0 + (params_.A11 * pcl_wait_pub->points[i].x + params_.A12 * pcl_wait_pub->points[i].y) / pcl_wait_pub->points[i].z;
        double v = params_.v0 + (params_.A22 * pcl_wait_pub->points[i].y) / pcl_wait_pub->points[i].z;

        Eigen::Vector2d p_c(u, v);
        
        // 判断pc对应的像素点是否在图像范围内
        if (u >= 0 && u < img_undistorted.cols && v >= 0 && v < img_undistorted.rows)
        {
           Eigen::Vector3f pixel = getInterpolatedPixel(img_undistorted, p_c);
           pointRGB.r = pixel[2];
           pointRGB.g = pixel[1];
           pointRGB.b = pixel[0];

           // 剔除距离过近的彩色点云
           if (p_f.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    }
    else
    {
      pub_num++;
    }

    return *laserCloudWorldRGB;;
}

Eigen::Vector3f PointCloudToDepthConverter::getInterpolatedPixel(cv::Mat img, Eigen::Vector2d pc)
{

    int width = img.cols;
    int height = img.rows;

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]);
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref - u_ref_i);
    const float subpix_v_ref = (v_ref - v_ref_i);
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
    float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
    float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
    float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
    Eigen::Vector3f pixel(B, G, R);
    return pixel;
}

std::pair<bool, std::string> PointCloudToDepthConverter::validateInputs(
    const pcl::PointCloud<pcl::PointXYZ> &cloud, const cv::Mat &image)
{
    if (cloud.empty())
    {
        return {false, "Empty point cloud"};
    }

    if (image.empty())
    {
        return {false, "Empty image"};
    }

    if (params_.A11 < 1e-6 || params_.A22 < 1e-6)
    {
        return {false, "Invalid camera intrinsics"};
    }

    return {true, ""};
}

void PointCloudToDepthConverter::updateCameraParams(const CameraParams &params)
{
    params_ = params;
    initializeInternalParams();
    createDistortionMaps();
}
