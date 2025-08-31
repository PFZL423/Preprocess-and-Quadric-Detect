#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Header.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <chrono>

#include "gpu_demo/GPUPreprocessor.h"

class NormalTestNode 
{
public:
    NormalTestNode() : nh_("~"), processed_first_frame_(false)
    {
        // 加载参数
        loadParameters();
        
        // 初始化GPU预处理器
        gpu_preprocessor_ = std::make_unique<GPUPreprocessor>();
        
        // 预分配GPU内存
        gpu_preprocessor_->reserveMemory(config_.max_points);
        
        // 设置发布者 (使用锁存发布)
        processed_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/processed_cloud", 1, true);
        normal_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/normal_markers", 1, true);
        
        // 设置订阅者
        cloud_sub_ = nh_.subscribe("/generated_cloud", 1, &NormalTestNode::cloudCallback, this);
        
        ROS_INFO("Normal test node initialized. Waiting for first point cloud on /generated_cloud...");
        ROS_INFO("Parameters loaded:");
        ROS_INFO("  Max points: %d", config_.max_points);
        ROS_INFO("  Voxel size: %.3f", config_.preprocess.voxel_size);
        ROS_INFO("  Normal radius: %.3f", config_.preprocess.normal_radius);
        ROS_INFO("  Compute normals: %s", config_.preprocess.compute_normals ? "true" : "false");
    }
    
    ~NormalTestNode() = default;

private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher processed_cloud_pub_;
    ros::Publisher normal_markers_pub_;
    
    std::unique_ptr<GPUPreprocessor> gpu_preprocessor_;
    bool processed_first_frame_;
    
    // 配置结构
    struct Config {
        int max_points;
        PreprocessConfig preprocess;
    } config_;
    
    void loadParameters()
    {
        // 从参数服务器加载配置
        nh_.param<int>("max_points", config_.max_points, 6000);
        
        // 预处理参数 - 调整为更适合椭球的参数
        nh_.param<float>("preprocess/voxel_size", config_.preprocess.voxel_size, 0.05f);  // 稍微增大体素
        nh_.param<bool>("preprocess/compute_normals", config_.preprocess.compute_normals, true);
        nh_.param<float>("preprocess/normal_radius", config_.preprocess.normal_radius, 0.15f); // 增大搜索半径
        nh_.param<int>("preprocess/normal_k", config_.preprocess.normal_k, 12);  // 减少期望邻居数
        
        // 离群点移除参数 - 放宽要求
        nh_.param<bool>("preprocess/enable_outlier_removal", config_.preprocess.enable_outlier_removal, false); // 暂时关闭
        nh_.param<float>("preprocess/radius_search", config_.preprocess.radius_search, 0.08f);
        nh_.param<int>("preprocess/min_radius_neighbors", config_.preprocess.min_radius_neighbors, 3);
        
        // 强制启用法线计算
        config_.preprocess.compute_normals = true;
        config_.preprocess.enable_voxel_filter = true;
        
        ROS_INFO("Adjusted parameters for better normal estimation:");
        ROS_INFO("  Voxel size: %.3f (larger for less density)", config_.preprocess.voxel_size);
        ROS_INFO("  Normal radius: %.3f (larger search)", config_.preprocess.normal_radius);
        ROS_INFO("  Normal k: %d (fewer required neighbors)", config_.preprocess.normal_k);
        ROS_INFO("  Outlier removal: %s (disabled for testing)", config_.preprocess.enable_outlier_removal ? "enabled" : "disabled");
    }
    
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        if (processed_first_frame_) {
            // 只处理第一帧
            return;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ROS_INFO("Received first point cloud with %d points. Processing...", 
                 (int)(msg->width * msg->height));
        
        try {
            // 转换为PCL点云
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud);
            
            // 限制点数
            if (cloud->size() > config_.max_points) {
                pcl::PassThrough<pcl::PointXYZ> pass;
                pass.setInputCloud(cloud);
                pass.setFilterFieldName("z");
                pass.setFilterLimits(-100.0, 100.0);  // 保留所有点，但限制数量
                pass.filter(*cloud);
                
                if (cloud->size() > config_.max_points) {
                    cloud->resize(config_.max_points);
                }
            }
            
            ROS_INFO("Processing %zu points with GPU preprocessor...", cloud->size());
            
            // GPU预处理
            ProcessingResult result = gpu_preprocessor_->process(cloud, config_.preprocess);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            ROS_INFO("GPU processing completed in %ld ms", duration.count());
            ROS_INFO("Output: %zu points with normals: %s", 
                     result.getPointCount(), 
                     result.hasNormals() ? "YES" : "NO");
            
            // 发布结果
            publishResults(result, msg->header);
            
            // 打印性能统计
            const auto& stats = gpu_preprocessor_->getLastStats();
            ROS_INFO("Performance breakdown:");
            ROS_INFO("  Upload: %.2f ms", stats.upload_time_ms);
            ROS_INFO("  Voxel filter: %.2f ms", stats.voxel_filter_time_ms);
            ROS_INFO("  Outlier removal: %.2f ms", stats.outlier_removal_time_ms);
            ROS_INFO("  Normal estimation: %.2f ms", stats.normal_estimation_time_ms);
            ROS_INFO("  Total: %.2f ms", stats.total_time_ms);
            
            processed_first_frame_ = true;
            ROS_INFO("First frame processed successfully. Node will now ignore further messages.");
            
        } catch (const std::exception& e) {
            ROS_ERROR("Error processing point cloud: %s", e.what());
        }
    }
    
    void publishResults(const ProcessingResult& result, const std_msgs::Header& header)
    {
        if (!result.hasNormals()) {
            ROS_WARN("No normals computed, cannot publish normal markers");
            return;
        }
        
        // 下载GPU结果到CPU
        std::vector<GPUPointNormal3f> points_with_normals = result.downloadPointsWithNormals();
        
        // 发布处理后的点云
        publishProcessedCloud(points_with_normals, header);
        
        // 发布法线markers
        publishNormalMarkers(points_with_normals, header);
    }
    
    void publishProcessedCloud(const std::vector<GPUPointNormal3f>& points_with_normals, 
                               const std_msgs::Header& header)
    {
        // 转换为PCL PointNormal点云
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
        cloud->resize(points_with_normals.size());
        
        for (size_t i = 0; i < points_with_normals.size(); ++i) {
            const auto& gpu_point = points_with_normals[i];
            auto& pcl_point = cloud->at(i);
            
            pcl_point.x = gpu_point.x;
            pcl_point.y = gpu_point.y;
            pcl_point.z = gpu_point.z;
            pcl_point.normal_x = gpu_point.normal_x;
            pcl_point.normal_y = gpu_point.normal_y;
            pcl_point.normal_z = gpu_point.normal_z;
            
            // 计算曲率 (简单估计)
            pcl_point.curvature = 0.0f;
        }
        
        // 转换为ROS消息并发布
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*cloud, output_msg);
        output_msg.header = header;
        output_msg.header.frame_id = "base_link";  // 确保有正确的frame_id
        
        processed_cloud_pub_.publish(output_msg);
        ROS_INFO("Published processed cloud with %zu points", cloud->size());
    }
    
    void publishNormalMarkers(const std::vector<GPUPointNormal3f>& points_with_normals,
                              const std_msgs::Header& header)
    {
        visualization_msgs::MarkerArray marker_array;
        
        // 统计有效和无效法线的数量
        int valid_normals = 0;
        int invalid_normals = 0;
        
        // 为了避免过多的marker，我们只显示部分点的法线
        int skip = std::max(1, (int)points_with_normals.size() / 500);  // 最多显示500个法线
        
        int marker_id = 0;
        for (size_t i = 0; i < points_with_normals.size(); i += skip) {
            const auto& point = points_with_normals[i];
            
            // 检查法线是否有效
            float normal_length = sqrt(point.normal_x * point.normal_x + 
                                     point.normal_y * point.normal_y + 
                                     point.normal_z * point.normal_z);
            
            if (normal_length < 0.1f) {
                invalid_normals++;
                continue;  // 跳过无效法线
            }
            
            valid_normals++;
            
            visualization_msgs::Marker marker;
            marker.header = header;
            marker.header.frame_id = "base_link";
            marker.ns = "normals";
            marker.id = marker_id++;
            marker.type = visualization_msgs::Marker::ARROW;
            marker.action = visualization_msgs::Marker::ADD;
            
            // 设置箭头起点和终点
            geometry_msgs::Point start, end;
            start.x = point.x;
            start.y = point.y;
            start.z = point.z;
            
            // 法线长度设为0.05米
            float arrow_length = 0.05f;
            end.x = point.x + point.normal_x * arrow_length;
            end.y = point.y + point.normal_y * arrow_length;
            end.z = point.z + point.normal_z * arrow_length;
            
            marker.points.push_back(start);
            marker.points.push_back(end);
            
            // 根据法线z分量设置颜色，便于调试
            marker.scale.x = 0.003;  // 箭头轴的直径
            marker.scale.y = 0.006;  // 箭头头部的直径
            marker.scale.z = 0.01;   // 箭头头部的长度
            
            marker.color.a = 0.8;
            
            // 颜色编码：z分量接近1的为红色（可能有问题），其他为绿色
            if (abs(point.normal_z) > 0.9f) {
                marker.color.r = 1.0;  // 红色 - 可能是朝上的错误法线
                marker.color.g = 0.0;
                marker.color.b = 0.0;
            } else {
                marker.color.r = 0.0;  // 绿色 - 正常法线
                marker.color.g = 1.0;
                marker.color.b = 0.0;
            }
            
            marker.lifetime = ros::Duration(0);  // 永久显示
            
            marker_array.markers.push_back(marker);
        }
        
        // 统计所有点的法线情况
        int total_valid = 0, total_invalid = 0;
        for (const auto& point : points_with_normals) {
            float normal_length = sqrt(point.normal_x * point.normal_x + 
                                     point.normal_y * point.normal_y + 
                                     point.normal_z * point.normal_z);
            if (normal_length < 0.1f) {  // 无效法线 (包括零向量)
                total_invalid++;
            } else {
                total_valid++;
            }
        }
        
        normal_markers_pub_.publish(marker_array);
        ROS_INFO("Normal statistics:");
        ROS_INFO("  Total points: %zu", points_with_normals.size());
        ROS_INFO("  Valid normals: %d (%.1f%%)", total_valid, 100.0 * total_valid / points_with_normals.size());
        ROS_INFO("  Invalid normals: %d (%.1f%%)", total_invalid, 100.0 * total_invalid / points_with_normals.size());
        ROS_INFO("  Published markers: %zu (showing %d valid, %d invalid)", marker_array.markers.size(), valid_normals, invalid_normals);
        ROS_INFO("  Color code: GREEN=normal directions, RED=suspicious (z>0.9)");
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "normal_test");
    
    try {
        NormalTestNode node;
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Normal test node failed: %s", e.what());
        return 1;
    }
    
    return 0;
}
