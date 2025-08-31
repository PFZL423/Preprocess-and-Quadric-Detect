#include "gpu_demo/QuadricDetect.h"
#include "gpu_demo/QuadricDetect_kernels.cuh"
#include <pcl/common/io.h>
#include <thrust/copy.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

QuadricDetect::QuadricDetect(const DetectorParams &params) : params_(params)
{
    cudaStreamCreate(&stream_);
}

QuadricDetect::~QuadricDetect()
{
    cudaStreamDestroy(stream_);
}

bool QuadricDetect::processCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud)
{
    if (!input_cloud || input_cloud->empty())
        return false;

    auto total_start = std::chrono::high_resolution_clock::now();

    detected_primitives_.clear();
    
    // Step 1: PCL转换和GPU上传
    auto convert_start = std::chrono::high_resolution_clock::now();
    convertPCLtoGPU(input_cloud);
    auto convert_end = std::chrono::high_resolution_clock::now();
    float convert_time = std::chrono::duration<float, std::milli>(convert_end - convert_start).count();

    // Step 2: 主要的二次曲面检测
    auto detect_start = std::chrono::high_resolution_clock::now();
    findQuadrics_BatchGPU();
    auto detect_end = std::chrono::high_resolution_clock::now();
    float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    std::cout << "[QuadricDetect] Timing breakdown:" << std::endl;
    std::cout << "  PCL->GPU convert: " << convert_time << " ms" << std::endl;
    std::cout << "  Quadric detection: " << detect_time << " ms" << std::endl;
    std::cout << "  Total: " << total_time << " ms" << std::endl;

    return true;
}

void QuadricDetect::convertPCLtoGPU(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    // Step 1: CPU数据转换
    auto cpu_convert_start = std::chrono::high_resolution_clock::now();
    std::vector<GPUPoint3f> h_points;
    h_points.reserve(cloud->size());

    for (const auto &pt : cloud->points)
    {
        GPUPoint3f gpu_pt;
        gpu_pt.x = pt.x;
        gpu_pt.y = pt.y;
        gpu_pt.z = pt.z;
        h_points.push_back(gpu_pt);
    }
    auto cpu_convert_end = std::chrono::high_resolution_clock::now();
    float cpu_convert_time = std::chrono::duration<float, std::milli>(cpu_convert_end - cpu_convert_start).count();

    // Step 2: GPU上传
    auto gpu_upload_start = std::chrono::high_resolution_clock::now();
    uploadPointsToGPU(h_points);
    auto gpu_upload_end = std::chrono::high_resolution_clock::now();
    float gpu_upload_time = std::chrono::duration<float, std::milli>(gpu_upload_end - gpu_upload_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    std::cout << "[convertPCLtoGPU] Detailed timing:" << std::endl;
    std::cout << "  CPU data conversion: " << cpu_convert_time << " ms" << std::endl;
    std::cout << "  GPU upload: " << gpu_upload_time << " ms" << std::endl;
    std::cout << "  Total: " << total_time << " ms" << std::endl;
}

Eigen::Matrix4f QuadricDetect::convertGPUModelToEigen(const GPUQuadricModel &gpu_model)
{
    Eigen::Matrix4f eigen_mat;
    for (int i = 0; i < 16; ++i)
    {
        eigen_mat(i / 4, i % 4) = gpu_model.coeffs[i];
    }
    return eigen_mat;
}

void QuadricDetect::findQuadrics_BatchGPU()
{
    auto total_detect_start = std::chrono::high_resolution_clock::now();

    const int batch_size = 1024;
    const int max_iterations = 10;

    // Step 1: 初始化GPU内存
    auto init_start = std::chrono::high_resolution_clock::now();
    initializeGPUMemory(batch_size);
    launchInitCurandStates(batch_size);
    auto init_end = std::chrono::high_resolution_clock::now();
    float init_time = std::chrono::duration<float, std::milli>(init_end - init_start).count();

    size_t remaining_points = d_remaining_indices_.size();
    size_t min_points = static_cast<size_t>(params_.min_remaining_points_percentage * d_all_points_.size());

    int iteration = 0;

    if (params_.verbosity > 0)
    {
        std::cout << "[findQuadrics_BatchGPU] 开始检测，总点数: " << d_all_points_.size()
                  << ", 最小剩余点数: " << min_points << std::endl;
    }

    float total_sampling_time = 0.0f;
    float total_inverse_power_time = 0.0f;
    float total_inlier_count_time = 0.0f;
    float total_best_model_time = 0.0f;
    float total_extract_inliers_time = 0.0f;
    float total_remove_points_time = 0.0f;

    while (remaining_points >= min_points && iteration < max_iterations)
    {
        if (params_.verbosity > 0)
        {
            std::cout << "== 第 " << iteration + 1 << " 次迭代，剩余点数 : " << remaining_points << " == " << std::endl;
        }

        // Step 2: 采样和构建矩阵
        auto sampling_start = std::chrono::high_resolution_clock::now();
        launchSampleAndBuildMatrices(batch_size);
        auto sampling_end = std::chrono::high_resolution_clock::now();
        float sampling_time = std::chrono::duration<float, std::milli>(sampling_end - sampling_start).count();
        total_sampling_time += sampling_time;
        
        // Step 3: 批量反幂迭代
        auto inverse_power_start = std::chrono::high_resolution_clock::now();
        performBatchInversePowerIteration(batch_size);
        auto inverse_power_end = std::chrono::high_resolution_clock::now();
        float inverse_power_time = std::chrono::duration<float, std::milli>(inverse_power_end - inverse_power_start).count();
        total_inverse_power_time += inverse_power_time;

        // Step 4: 计算内点数
        auto inlier_count_start = std::chrono::high_resolution_clock::now();
        launchCountInliersBatch(batch_size);
        auto inlier_count_end = std::chrono::high_resolution_clock::now();
        float inlier_count_time = std::chrono::duration<float, std::milli>(inlier_count_end - inlier_count_start).count();
        total_inlier_count_time += inlier_count_time;

        // Step 5: 找最优模型
        auto best_model_start = std::chrono::high_resolution_clock::now();
        launchFindBestModel(batch_size);
        auto best_model_end = std::chrono::high_resolution_clock::now();
        float best_model_time = std::chrono::duration<float, std::milli>(best_model_end - best_model_start).count();
        total_best_model_time += best_model_time;

        // 获取最优结果
        thrust::host_vector<int> h_best_index(1);
        thrust::host_vector<int> h_best_count(1);
        getBestModelResults(h_best_index, h_best_count);

        int best_count = h_best_count[0];
        int best_model_idx = h_best_index[0];

        if (best_count < params_.min_quadric_inlier_count_absolute)
        {
            if (params_.verbosity > 0)
            {
                std::cout << "最优模型内点数不足，结束检测" << std::endl;
            }
            break;
        }

        // Step 6: 获取最优模型
        thrust::host_vector<GPUQuadricModel> h_best_model(1);
        thrust::copy_n(d_batch_models_.begin() + best_model_idx, 1, h_best_model.begin());
        GPUQuadricModel best_gpu_model = h_best_model[0];

        // 🆕 添加：输出最优模型详情
        if (params_.verbosity > 0)
        {
            outputBestModelDetails(best_gpu_model, best_count, best_model_idx, iteration + 1);
        }

        // Step 7: 提取内点索引
        auto extract_inliers_start = std::chrono::high_resolution_clock::now();
        launchExtractInliers(&best_gpu_model);
        auto extract_inliers_end = std::chrono::high_resolution_clock::now();
        float extract_inliers_time = std::chrono::duration<float, std::milli>(extract_inliers_end - extract_inliers_start).count();
        total_extract_inliers_time += extract_inliers_time;

        // Step 8: 构建内点点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud = extractInlierCloud();

        // Step 9: 保存检测结果
        DetectedPrimitive detected_quadric;
        detected_quadric.type = "quadric";
        detected_quadric.model_coefficients = convertGPUModelToEigen(best_gpu_model);
        detected_quadric.inliers = inlier_cloud;
        detected_primitives_.push_back(detected_quadric);

        if (params_.verbosity > 0)
        {
            std::cout << "已保存第 " << detected_primitives_.size() << " 个二次曲面" << std::endl;
        }

        // Step 10: 移除内点
        auto remove_points_start = std::chrono::high_resolution_clock::now();
        std::vector<int> dummy_vector; // 实际使用GPU数据
        removeFoundPoints(dummy_vector);
        auto remove_points_end = std::chrono::high_resolution_clock::now();
        float remove_points_time = std::chrono::duration<float, std::milli>(remove_points_end - remove_points_start).count();
        total_remove_points_time += remove_points_time;

        // 输出本次迭代的详细计时
        if (params_.verbosity > 0)
        {
            std::cout << "[Iteration " << iteration + 1 << "] Timing breakdown:" << std::endl;
            std::cout << "  Sampling & matrices: " << sampling_time << " ms" << std::endl;
            std::cout << "  Inverse power iteration: " << inverse_power_time << " ms" << std::endl;
            std::cout << "  Inlier counting: " << inlier_count_time << " ms" << std::endl;
            std::cout << "  Best model finding: " << best_model_time << " ms" << std::endl;
            std::cout << "  Extract inliers: " << extract_inliers_time << " ms" << std::endl;
            std::cout << "  Remove points: " << remove_points_time << " ms" << std::endl;
            float iteration_total = sampling_time + inverse_power_time + inlier_count_time + 
                                  best_model_time + extract_inliers_time + remove_points_time;
            std::cout << "  Iteration total: " << iteration_total << " ms" << std::endl;
        }

        // 更新循环条件
        remaining_points = d_remaining_indices_.size();
        iteration++;
    }

    auto total_detect_end = std::chrono::high_resolution_clock::now();
    float total_detect_time = std::chrono::duration<float, std::milli>(total_detect_end - total_detect_start).count();

    if (params_.verbosity > 0)
    {
        std::cout << "\n[findQuadrics_BatchGPU] Final timing summary:" << std::endl;
        std::cout << "  Initialization: " << init_time << " ms" << std::endl;
        std::cout << "  Total sampling: " << total_sampling_time << " ms" << std::endl;
        std::cout << "  Total inverse power: " << total_inverse_power_time << " ms" << std::endl;
        std::cout << "  Total inlier counting: " << total_inlier_count_time << " ms" << std::endl;
        std::cout << "  Total best model finding: " << total_best_model_time << " ms" << std::endl;
        std::cout << "  Total extract inliers: " << total_extract_inliers_time << " ms" << std::endl;
        std::cout << "  Total remove points: " << total_remove_points_time << " ms" << std::endl;
        std::cout << "  Total detection time: " << total_detect_time << " ms" << std::endl;
        std::cout << "== 检测完成，共找到 " << detected_primitives_.size() << " 个二次曲面 == " << std::endl;
    }
}

void QuadricDetect::performBatchInversePowerIteration(int batch_size)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    if (params_.verbosity > 0)
    {
        std::cout << "[performBatchInversePowerIteration] 启动批量反幂迭代，batch_size=" << batch_size << std::endl;
    }

    // Step 1: 从9×10矩阵计算10×10的A^T*A矩阵
    auto ata_start = std::chrono::high_resolution_clock::now();
    launchComputeATA(batch_size);
    auto ata_end = std::chrono::high_resolution_clock::now();
    float ata_time = std::chrono::duration<float, std::milli>(ata_end - ata_start).count();

    // Step 2: 对A^T*A进行QR分解
    auto qr_start = std::chrono::high_resolution_clock::now();
    launchBatchQR(batch_size);
    auto qr_end = std::chrono::high_resolution_clock::now();
    float qr_time = std::chrono::duration<float, std::milli>(qr_end - qr_start).count();

    // Step 3: 反幂迭代求最小特征向量
    auto power_start = std::chrono::high_resolution_clock::now();
    launchBatchInversePower(batch_size);
    auto power_end = std::chrono::high_resolution_clock::now();
    float power_time = std::chrono::duration<float, std::milli>(power_end - power_start).count();

    // Step 4: 将特征向量转换为二次曲面模型
    auto extract_start = std::chrono::high_resolution_clock::now();
    launchExtractQuadricModels(batch_size);
    auto extract_end = std::chrono::high_resolution_clock::now();
    float extract_time = std::chrono::duration<float, std::milli>(extract_end - extract_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    if (params_.verbosity > 0)
    {
        std::cout << "[InversePower] Detailed timing:" << std::endl;
        std::cout << "  Compute A^T*A: " << ata_time << " ms" << std::endl;
        std::cout << "  QR decomposition: " << qr_time << " ms" << std::endl;
        std::cout << "  Inverse power iteration: " << power_time << " ms" << std::endl;
        std::cout << "  Extract quadric models: " << extract_time << " ms" << std::endl;
        std::cout << "  Total: " << total_time << " ms" << std::endl;
        std::cout << "[performBatchInversePowerIteration] 批量反幂迭代完成" << std::endl;
    }

    // 🆕 添加：验证反幂迭代结果
    if (params_.verbosity > 0)
    {
        validateInversePowerResults(batch_size);
    }
}

void QuadricDetect::performLO_RANSAC(GPUQuadricModel &best_model, int &inlier_count)
{
    // LO-RANSAC 实现
}

void QuadricDetect::removeFoundPoints(const std::vector<int> &indices_to_remove)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    if (d_temp_inlier_indices_.empty() || current_inlier_count_ == 0)
    {
        return;
    }

    if (params_.verbosity > 0)
    {
        std::cout << "[removeFoundPoints] 移除前剩余点数: " << d_remaining_indices_.size() << std::endl;
    }

    // 🚀 方案：使用自定义CUDA内核，完全避免Thrust set_difference
    auto kernel_start = std::chrono::high_resolution_clock::now();
    launchRemovePointsKernel();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    float kernel_time = std::chrono::duration<float, std::milli>(kernel_end - kernel_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    if (params_.verbosity > 0)
    {
        std::cout << "[removeFoundPoints] Remove kernel: " << kernel_time << " ms" << std::endl;
        std::cout << "[removeFoundPoints] Total time: " << total_time << " ms" << std::endl;
        std::cout << "[removeFoundPoints] 移除了 " << current_inlier_count_
                  << " 个内点，剩余 " << d_remaining_indices_.size() << " 个点" << std::endl;
    }
}

void QuadricDetect::findPlanes_PCL()
{
    // PCL平面检测实现
}

const std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> &
QuadricDetect::getDetectedPrimitives() const
{
    return detected_primitives_;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr QuadricDetect::getFinalCloud() const
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    return final_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr QuadricDetect::extractInlierCloud() const
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (d_temp_inlier_indices_.empty() || current_inlier_count_ == 0)
    {
        return inlier_cloud;
    }

    // 从GPU拷贝内点索引到CPU
    thrust::host_vector<int> h_inlier_indices = d_temp_inlier_indices_;
    thrust::host_vector<GPUPoint3f> h_all_points = d_all_points_;

    inlier_cloud->reserve(current_inlier_count_);

    // 根据索引构建点云
    for (int i = 0; i < current_inlier_count_; ++i)
    {
        int idx = h_inlier_indices[i];
        if (idx >= 0 && idx < h_all_points.size())
        {
            pcl::PointXYZ pt;
            pt.x = h_all_points[idx].x;
            pt.y = h_all_points[idx].y;
            pt.z = h_all_points[idx].z;
            inlier_cloud->push_back(pt);
        }
    }

    inlier_cloud->width = inlier_cloud->size();
    inlier_cloud->height = 1;
    inlier_cloud->is_dense = true;

    if (params_.verbosity > 0)
    {
        std::cout << "[extractInlierCloud] 构建了包含 " << inlier_cloud->size() << " 个点的点云" << std::endl;
    }

    return inlier_cloud;
}






// 🆕 新增函数：验证反幂迭代结果
void QuadricDetect::validateInversePowerResults(int batch_size)
{
    std::cout << "[validateInversePowerResults] 🔍 验证反幂迭代结果..." << std::endl;

    // 检查前几个特征向量和模型
    int check_count = std::min(3, batch_size);

    // 1. 检查特征向量
    thrust::host_vector<float> h_eigenvectors(check_count * 10);
    thrust::copy_n(d_batch_eigenvectors_.begin(), check_count * 10, h_eigenvectors.begin());

    // 2. 检查生成的模型
    thrust::host_vector<GPUQuadricModel> h_models(check_count);
    thrust::copy_n(d_batch_models_.begin(), check_count, h_models.begin());

    bool all_valid = true;

    for (int i = 0; i < check_count; ++i)
    {
        std::cout << "  📊 模型 " << i << ":" << std::endl;

        // 检查特征向量
        float *eigenvec = &h_eigenvectors[i * 10];
        float norm_sq = 0.0f;
        bool has_nan = false;

        for (int j = 0; j < 10; ++j)
        {
            norm_sq += eigenvec[j] * eigenvec[j];
            if (!std::isfinite(eigenvec[j]) || std::isnan(eigenvec[j]))
            {
                has_nan = true;
            }
        }

        float norm = std::sqrt(norm_sq);

        if (has_nan)
        {
            std::cout << "    ❌ 特征向量包含NaN/Inf值" << std::endl;
            all_valid = false;
        }
        else if (norm < 1e-12f)
        {
            std::cout << "    ❌ 特征向量模长过小: " << norm << std::endl;
            all_valid = false;
        }
        else
        {
            std::cout << "    ✅ 特征向量正常，模长: " << norm << std::endl;
        }

        // 检查模型系数
        const GPUQuadricModel &model = h_models[i];
        bool model_valid = true;
        float coeff_sum = 0.0f;

        for (int j = 0; j < 16; ++j)
        {
            coeff_sum += std::abs(model.coeffs[j]);
            if (!std::isfinite(model.coeffs[j]) || std::isnan(model.coeffs[j]))
            {
                model_valid = false;
                break;
            }
        }

        if (!model_valid)
        {
            std::cout << "    ❌ 模型系数包含NaN/Inf值" << std::endl;
            all_valid = false;
        }
        else if (coeff_sum < 1e-12f)
        {
            std::cout << "    ❌ 模型系数全为零" << std::endl;
            all_valid = false;
        }
        else
        {
            std::cout << "    ✅ 模型系数正常，系数和: " << coeff_sum << std::endl;
        }

        // 显示前几个系数
        if (params_.verbosity > 1)
        {
            std::cout << "    📋 前6个系数: [";
            for (int j = 0; j < 6; ++j)
            {
                std::cout << model.coeffs[j];
                if (j < 5)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    if (all_valid)
    {
        std::cout << "[validateInversePowerResults] ✅ 反幂迭代结果验证通过" << std::endl;
    }
    else
    {
        std::cout << "[validateInversePowerResults] ❌ 反幂迭代结果存在问题，请检查算法实现" << std::endl;
    }
}

// 🆕 新增函数：输出最优模型详情
void QuadricDetect::outputBestModelDetails(const GPUQuadricModel &best_model, int inlier_count, int model_idx, int iteration)
{
    std::cout << "\n🏆 ========== 第" << iteration << "次迭代最优模型详情 ==========" << std::endl;
    std::cout << "📍 模型索引: " << model_idx << " (在1024个候选中)" << std::endl;
    std::cout << "👥 内点数量: " << inlier_count << std::endl;
    std::cout << "📊 内点比例: " << (100.0 * inlier_count / d_remaining_indices_.size()) << "%" << std::endl;

    // 转换为Eigen矩阵便于显示
    Eigen::Matrix4f Q = convertGPUModelToEigen(best_model);

    std::cout << "🔢 二次曲面矩阵 Q:" << std::endl;
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "   [";
        for (int j = 0; j < 4; ++j)
        {
            std::cout << std::setw(10) << std::setprecision(6) << std::fixed << Q(i, j);
            if (j < 3)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // 分析二次曲面类型（简单判断）
    float det = Q.determinant();
    std::cout << "🔍 矩阵行列式: " << det << std::endl;

    // 检查对角线元素符号
    int pos_diag = 0, neg_diag = 0, zero_diag = 0;
    for (int i = 0; i < 3; ++i) // 只看前3×3部分
    {
        if (Q(i, i) > 1e-6f)
            pos_diag++;
        else if (Q(i, i) < -1e-6f)
            neg_diag++;
        else
            zero_diag++;
    }

    std::cout << "📈 对角线符号分布: +" << pos_diag << " / -" << neg_diag << " / 0:" << zero_diag;

    // 简单的曲面类型推断
    if (pos_diag == 3 || neg_diag == 3)
    {
        std::cout << " → 可能是椭球面" << std::endl;
    }
    else if ((pos_diag == 2 && neg_diag == 1) || (pos_diag == 1 && neg_diag == 2))
    {
        std::cout << " → 可能是双曲面" << std::endl;
    }
    else if (zero_diag > 0)
    {
        std::cout << " → 可能是抛物面或退化曲面" << std::endl;
    }
    else
    {
        std::cout << " → 曲面类型待进一步分析" << std::endl;
    }

    std::cout << "================================================\n"
              << std::endl;
}



//重载实现
bool QuadricDetect::processCloud(const thrust::device_vector<GPUPoint3f> &input_cloud)
{
    if (input_cloud.empty())
        return false;

    auto total_start = std::chrono::high_resolution_clock::now();

    detected_primitives_.clear();

    // Step 1: GPU数据直接赋值 (无CPU-GPU传输)
    auto convert_start = std::chrono::high_resolution_clock::now();
    uploadPointsToGPU(input_cloud);
    auto convert_end = std::chrono::high_resolution_clock::now();
    float convert_time = std::chrono::duration<float, std::milli>(convert_end - convert_start).count();

    // Step 2: 主要的二次曲面检测
    auto detect_start = std::chrono::high_resolution_clock::now();
    findQuadrics_BatchGPU();
    auto detect_end = std::chrono::high_resolution_clock::now();
    float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    std::cout << "[QuadricDetect] GPU-Direct Timing breakdown:" << std::endl;
    std::cout << "  GPU data assignment: " << convert_time << " ms" << std::endl;
    std::cout << "  Quadric detection: " << detect_time << " ms" << std::endl;
    std::cout << "  Total: " << total_time << " ms" << std::endl;

    return true;
}