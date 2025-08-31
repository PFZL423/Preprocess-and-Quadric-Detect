/**
 * @file gpu_check.cpp
 * @brief 完整的GPU预处理+二次曲面检测测试节点
 * 
 * 功能：
 * 1. 接收点云话题
 * 2. GPU预处理 -> GPU二次曲面检测
 * 3. 全程GPU处理，零CPU-GPU传输瓶颈
 * 4. 完整性能计时分析
 * 
 * @author PFZL-423
 * @date 2025-08-28
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <chrono>
#include <memory>

#include "gpu_demo/GPUPreprocessor.h"
#include "gpu_demo/QuadricDetect.h"

/**
 * @class IntegratedGPUTestNode
 * @brief 完整的GPU预处理+检测测试节点
 */
class IntegratedGPUTestNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // ROS通信
    ros::Subscriber cloud_sub_;
    
    // GPU处理器
    std::unique_ptr<GPUPreprocessor> preprocessor_;
    std::unique_ptr<QuadricDetect> detector_;
    
    // 配置参数
    std::string input_topic_;
    int max_points_;
    bool has_processed_;
    
    // 预处理配置
    PreprocessConfig preprocess_config_;
    
    // 检测参数
    DetectorParams detector_params_;
    
public:
    IntegratedGPUTestNode() : private_nh_("~"), has_processed_(false)
    {
        // Step 1: 加载参数
        loadParameters();
        
        // Step 2: 初始化GPU处理器
        initializeProcessors();
        
        // Step 3: 设置ROS通信
        setupROS();
        
        ROS_INFO("🚀 Integrated GPU Test Node Initialized Successfully!");
        ROS_INFO("   - Input topic: %s", input_topic_.c_str());
        ROS_INFO("   - Will process ONLY the first point cloud");
        ROS_INFO("   - Pipeline: Input -> GPU Preprocess -> GPU Quadric Detect -> Results");
    }
    
private:
    /**
     * @brief 加载ROS参数
     */
    void loadParameters()
    {
        // ROS话题配置
        private_nh_.param<std::string>("input_topic", input_topic_, "/generated_cloud");
        private_nh_.param<int>("max_points", max_points_, 6000);
        
        // 预处理参数 (参考test_preprocess.cpp)
        private_nh_.param("preprocess/voxel_size", preprocess_config_.voxel_size, 0.08f);
        private_nh_.param("preprocess/compute_normals", preprocess_config_.compute_normals, false);
        private_nh_.param("preprocess/enable_outlier_removal", preprocess_config_.enable_outlier_removal, true);
        private_nh_.param("preprocess/radius_search", preprocess_config_.radius_search, 0.08f);
        private_nh_.param("preprocess/min_radius_neighbors", preprocess_config_.min_radius_neighbors, 5);
        
        // 检测参数
        private_nh_.param<double>("detector/min_remaining_points_percentage", detector_params_.min_remaining_points_percentage, 0.03);
        private_nh_.param<double>("detector/quadric_distance_threshold", detector_params_.quadric_distance_threshold, 0.02);
        private_nh_.param<int>("detector/min_quadric_inlier_count_absolute", detector_params_.min_quadric_inlier_count_absolute, 500);
        private_nh_.param<int>("detector/quadric_max_iterations", detector_params_.quadric_max_iterations, 5000);
        private_nh_.param<int>("detector/verbosity", detector_params_.verbosity, 1);
        
        ROS_INFO("[Parameters] Loaded successfully:");
        ROS_INFO("  - Voxel size: %.3f", preprocess_config_.voxel_size);
        ROS_INFO("  - Distance threshold: %.3f", detector_params_.quadric_distance_threshold);
        ROS_INFO("  - Min inlier count: %d", detector_params_.min_quadric_inlier_count_absolute);
    }
    
    /**
     * @brief 初始化GPU处理器
     */
    void initializeProcessors()
    {
        try {
            // 初始化预处理器
            preprocessor_ = std::make_unique<GPUPreprocessor>();
            preprocessor_->reserveMemory(max_points_);
            
            // 初始化检测器
            detector_ = std::make_unique<QuadricDetect>(detector_params_);
            
            ROS_INFO("[Processors] GPU processors initialized successfully");
        }
        catch (const std::exception& e) {
            ROS_ERROR("[Processors] Failed to initialize: %s", e.what());
            exit(EXIT_FAILURE);
        }
    }
    
    /**
     * @brief 设置ROS通信
     */
    void setupROS()
    {
        cloud_sub_ = nh_.subscribe(input_topic_, 1, &IntegratedGPUTestNode::cloudCallback, this);
    }
    
    /**
     * @brief 点云回调函数 - 完整的GPU流水线
     */
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        // 只处理第一帧
        if (has_processed_) {
            return;
        }
        has_processed_ = true;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        ROS_INFO("========================================");
        ROS_INFO("[IntegratedGPU] 🎯 Processing THE ONLY point cloud");
        ROS_INFO("  - Point count: %d", msg->width * msg->height);
        ROS_INFO("  - Frame ID: %s", msg->header.frame_id.c_str());
        
        try {
            // Step 1: 转换为PCL
            auto pcl_start = std::chrono::high_resolution_clock::now();
            pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *input_cloud);
            auto pcl_end = std::chrono::high_resolution_clock::now();
            float pcl_time = std::chrono::duration<float, std::milli>(pcl_end - pcl_start).count();
            
            // Step 2: GPU预处理
            auto preprocess_start = std::chrono::high_resolution_clock::now();
            ProcessingResult preprocess_result = preprocessor_->process(input_cloud, preprocess_config_);
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            float preprocess_time = std::chrono::duration<float, std::milli>(preprocess_end - preprocess_start).count();
            
            // Step 3: 获取GPU数据引用 (zero-copy)
            auto gpu_data_start = std::chrono::high_resolution_clock::now();
            const auto& gpu_points = preprocess_result.getPoints();
            auto gpu_data_end = std::chrono::high_resolution_clock::now();
            float gpu_data_time = std::chrono::duration<float, std::milli>(gpu_data_end - gpu_data_start).count();
            
            // Step 4: GPU二次曲面检测 (使用重载接口)
            auto detect_start = std::chrono::high_resolution_clock::now();
            bool success = detector_->processCloud(gpu_points);
            auto detect_end = std::chrono::high_resolution_clock::now();
            float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();
            
            auto total_end = std::chrono::high_resolution_clock::now();
            float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();
            
            // 输出结果
            if (success) {
                const auto& detected_primitives = detector_->getDetectedPrimitives();
                
                ROS_INFO("🎯 Integrated GPU Pipeline Results:");
                ROS_INFO("  - Detected %lu quadric surfaces", detected_primitives.size());
                ROS_INFO("  - Input points: %zu", input_cloud->size());
                ROS_INFO("  - Preprocessed points: %zu", gpu_points.size());
                
                // 🆕 输出每个检测到的二次曲面详细信息
                for (size_t i = 0; i < detected_primitives.size(); ++i) {
                    const auto& primitive = detected_primitives[i];
                    ROS_INFO("🔸 Quadric Surface #%zu:", i + 1);
                    ROS_INFO("  - Type: %s", primitive.type.c_str());
                    ROS_INFO("  - Inlier count: %zu", primitive.inliers->size());
                    
                    // 输出4x4二次曲面矩阵
                    ROS_INFO("  🎯 Quadric Surface #%zu Matrix Parameters (4×4):", i + 1);
                    const auto& Q = primitive.model_coefficients;
                    ROS_INFO("    [ %8.6f, %8.6f, %8.6f, %8.6f]", Q(0,0), Q(0,1), Q(0,2), Q(0,3));
                    ROS_INFO("    [ %8.6f, %8.6f, %8.6f, %8.6f]", Q(1,0), Q(1,1), Q(1,2), Q(1,3));
                    ROS_INFO("    [ %8.6f, %8.6f, %8.6f, %8.6f]", Q(2,0), Q(2,1), Q(2,2), Q(2,3));
                    ROS_INFO("    [ %8.6f, %8.6f, %8.6f, %8.6f]", Q(3,0), Q(3,1), Q(3,2), Q(3,3));
                    
                    // 分析二次曲面类型
                    float det = Q.determinant();
                    int positive_eigenvals = 0, negative_eigenvals = 0;
                    if (Q(0,0) > 0) positive_eigenvals++; else if (Q(0,0) < 0) negative_eigenvals++;
                    if (Q(1,1) > 0) positive_eigenvals++; else if (Q(1,1) < 0) negative_eigenvals++;
                    if (Q(2,2) > 0) positive_eigenvals++; else if (Q(2,2) < 0) negative_eigenvals++;
                    
                    ROS_INFO("  🔍 Matrix determinant: %.6f", det);
                    ROS_INFO("  📈 Diagonal signs: +%d / -%d → Likely %s", 
                             positive_eigenvals, negative_eigenvals,
                             (negative_eigenvals == 3) ? "Ellipsoid" : 
                             (negative_eigenvals == 2) ? "Hyperboloid" : "Other");
                    ROS_INFO("  🎯 Main Coefficients: a=%.4f, b=%.4f, c=%.4f", Q(0,0), Q(1,1), Q(2,2));
                }
                
                ROS_INFO("📊 Detailed Timing Breakdown:");
                ROS_INFO("  - PCL conversion: %.2f ms", pcl_time);
                ROS_INFO("  - GPU preprocessing: %.2f ms", preprocess_time);
                ROS_INFO("  - GPU data reference: %.2f ms", gpu_data_time);
                ROS_INFO("  - GPU quadric detection: %.2f ms", detect_time);
                ROS_INFO("  - Total pipeline: %.2f ms", total_time);
                
                float compression_ratio = 100.0f * gpu_points.size() / input_cloud->size();
                ROS_INFO("  - Compression ratio: %.1f%% (%zu -> %zu points)", 
                         compression_ratio, input_cloud->size(), gpu_points.size());
                
                ROS_INFO("✅ [IntegratedGPU] Pipeline completed successfully!");
            } else {
                ROS_ERROR("❌ [IntegratedGPU] Detection failed");
            }
        }
        catch (const std::exception& e) {
            ROS_ERROR("[IntegratedGPU] ❌ Error: %s", e.what());
        }
        
        ROS_INFO("🔒 Processing locked. Node will ignore all future point clouds.");
        ROS_INFO("========================================");
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "integrated_gpu_test_node");
    
    try {
        IntegratedGPUTestNode node;
        ros::spin();
    }
    catch (const std::exception& e) {
        ROS_ERROR("Node failed: %s", e.what());
        return -1;
    }
    
    return 0;
}


