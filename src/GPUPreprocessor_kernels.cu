#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <iostream>
#include <numeric>
#include <chrono>

#include "gpu_demo/GPUPreprocessor_kernels.cuh"
#include "gpu_demo/GPUPreprocessor.h"

// ========== GPU Kernel实现 (保持不变) ==========
namespace VoxelFilter
{
    __device__ inline uint64_t computeVoxelHash(float x, float y, float z, float voxel_size)
    {
        // ✅ 添加输入验证
        if (!isfinite(x) || !isfinite(y) || !isfinite(z) || voxel_size <= 0.0f)
        {
            return 0; // 返回安全的默认值
        }

        int vx = __float2int_rd(x / voxel_size);
        int vy = __float2int_rd(y / voxel_size);
        int vz = __float2int_rd(z / voxel_size);

        // ✅ 限制范围，避免溢出
        vx = max(-1048576, min(1048575, vx)); // ±2^20
        vy = max(-1048576, min(1048575, vy)); // ±2^20
        vz = max(-512, min(511, vz));         // ±2^9

        uint32_t ux = static_cast<uint32_t>(vx + (1 << 20));
        uint32_t uy = static_cast<uint32_t>(vy + (1 << 20));
        uint32_t uz = static_cast<uint32_t>(vz + (1 << 9));

        uint64_t hash = (static_cast<uint64_t>(ux) << 32) |
                        (static_cast<uint64_t>(uy) << 10) |
                        static_cast<uint64_t>(uz);
        return hash;
    }

    __global__ void computeVoxelKeysKernel(
        const GPUPoint3f *points,
        uint64_t *voxel_keys,
        float voxel_size,
        int point_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;

        const GPUPoint3f &point = points[idx];
        voxel_keys[idx] = computeVoxelHash(point.x, point.y, point.z, voxel_size);
    }
}

namespace OutlierRemoval
{
    __device__ inline float computeDistance(const GPUPoint3f &p1, const GPUPoint3f &p2)
    {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return sqrtf(dx * dx + dy * dy + dz * dz);
    }

    __global__ void radiusOutlierKernel(
        const GPUPoint3f *points,
        bool *valid_flags,
        int point_count,
        float radius,
        int min_neighbors)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;

        const GPUPoint3f &query_point = points[idx];
        int neighbor_count = 0;

        for (int i = 0; i < point_count; ++i)
        {
            if (i == idx)
                continue;

            float dist = computeDistance(query_point, points[i]);
            if (dist <= radius)
            {
                neighbor_count++;
            }
        }

        valid_flags[idx] = (neighbor_count >= min_neighbors);
    }

    __global__ void statisticalOutlierKernel(
        const GPUPoint3f *points,
        bool *valid_flags,
        int point_count,
        int k,
        float std_dev_multiplier)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;
        valid_flags[idx] = true;
    }
}

namespace NormalEstimation
{
    __global__ void estimateNormalsKernel(
        const GPUPoint3f *points,
        GPUPointNormal3f *points_with_normals,
        int point_count,
        float radius,
        int k)
    {
        // int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // if (idx >= point_count)
        //     return;

        // points_with_normals[idx].x = points[idx].x;
        // points_with_normals[idx].y = points[idx].y;
        // points_with_normals[idx].z = points[idx].z;

        // points_with_normals[idx].normal_x = 0.0f;
        // points_with_normals[idx].normal_y = 0.0f;
        // points_with_normals[idx].normal_z = 1.0f;
    }
}

namespace GroundRemoval
{
    __global__ void ransacGroundDetectionKernel(
        const GPUPoint3f *points,
        bool *ground_flags,
        int point_count,
        float threshold,
        int max_iterations)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;
        ground_flags[idx] = (points[idx].z < threshold);
    }
}

namespace Utils
{
    __global__ void compactPointsKernel(
        const GPUPoint3f *input_points,
        const bool *valid_flags,
        GPUPoint3f *output_points,
        int *output_count,
        int point_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;

        if (valid_flags[idx])
        {
            int output_idx = atomicAdd(output_count, 1);
            output_points[output_idx] = input_points[idx];
        }
    }

    __global__ void convertToPointNormalKernel(
        const GPUPoint3f *input_points,
        GPUPointNormal3f *output_points,
        int point_count)
    {
        // int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // if (idx >= point_count)
        //     return;

        // output_points[idx].x = input_points[idx].x;
        // output_points[idx].y = input_points[idx].y;
        // output_points[idx].z = input_points[idx].z;
        // output_points[idx].normal_x = 0.0f;
        // output_points[idx].normal_y = 0.0f;
        // output_points[idx].normal_z = 1.0f;
    }
}

// ========== 🔥 核心改造：GPUPreprocessor成员函数 (全GPU框架风格) ==========

// void GPUPreprocessor::cuda_performNormalEstimation(
//     GPUPoint3f *points, GPUPointNormal3f *points_with_normals,
//     size_t point_count, float radius, int k)
// {
//     if (point_count == 0)
//         return;

//     dim3 block(256);
//     dim3 grid((point_count + block.x - 1) / block.x);

//     // ✅ 直接使用传入的指针，避免device_vector构造
//     NormalEstimation::estimateNormalsKernel<<<grid, block>>>(
//         points, points_with_normals, point_count, radius, k);
//     cudaDeviceSynchronize();
// }

size_t GPUPreprocessor::cuda_compactValidPoints(
    GPUPoint3f *input_points, bool *valid_flags,
    GPUPoint3f *output_points, size_t input_count)
{
    if (input_count == 0)
        return 0;

    // 复制valid_flags到成员变量
    thrust::copy(thrust::device_ptr<bool>(valid_flags),
                 thrust::device_ptr<bool>(valid_flags + input_count),
                 d_valid_flags_.begin());

    d_output_points_.clear();
    d_output_points_.reserve(input_count);

    std::vector<GPUPoint3f> h_temp_output(input_count);
    thrust::device_vector<GPUPoint3f> d_temp_output = h_temp_output;

    auto new_end = thrust::copy_if(
        d_temp_points_.begin(), d_temp_points_.begin() + input_count,
        d_valid_flags_.begin(),
        d_temp_output.begin(),
        thrust::identity<bool>());

    size_t output_count = new_end - d_temp_output.begin();
    thrust::copy(d_temp_output.begin(), d_temp_output.begin() + output_count,
                 thrust::device_ptr<GPUPoint3f>(output_points));

    return output_count;
}

// void GPUPreprocessor::cuda_convertToPointsWithNormals(
//     GPUPoint3f *input_points, GPUPointNormal3f *output_points, size_t point_count)
// {
//     // if (point_count == 0)
//     //     return;

//     // dim3 block(256);
//     // dim3 grid((point_count + block.x - 1) / block.x);

//     // // ✅ 直接操作指针，避免device_vector构造
//     // Utils::convertToPointNormalKernel<<<grid, block>>>(
//     //     input_points, output_points, point_count);
//     // cudaDeviceSynchronize();
// }

// ========== 在.cu文件末尾添加所有GPU内存管理函数 ==========

void GPUPreprocessor::cuda_initializeMemory(size_t max_points)
{
    // ✅ 在.cu文件中，所有resize都是安全的
    // 只调整大小，不初始化数据，等待后续填充
    if (d_voxel_keys_.size() < max_points)
    {
        d_voxel_keys_.resize(max_points);
    }
    if (d_valid_flags_.size() < max_points)
    {
        d_valid_flags_.resize(max_points);
    }
    if (d_neighbor_counts_.size() < max_points)
    {
        d_neighbor_counts_.resize(max_points);
    }
    if (d_knn_indices_.size() < max_points * 20)
    {
        d_knn_indices_.resize(max_points * 20);
    }
    if (d_knn_distances_.size() < max_points * 20)
    {
        d_knn_distances_.resize(max_points * 20);
    }
    if (d_voxel_boundaries_.size() < max_points)
    {
        d_voxel_boundaries_.resize(max_points);
    }
    if (d_unique_keys_.size() < max_points)
    {
        d_unique_keys_.resize(max_points);
    }

    // POD结构体只需要reserve即可，大小会在使用时正确设置
    d_temp_points_.reserve(max_points);
    d_output_points_.reserve(max_points);
    // d_output_points_normal_.reserve(max_points);
}
void GPUPreprocessor::cuda_launchVoxelFilter(float voxel_size)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    std::cout << "[GPUPreprocessor] Starting voxel filter with size " << voxel_size << std::endl;

    size_t input_count = d_temp_points_.size();
    if (input_count == 0)
        return;

    // Step 1: 准备内存
    auto memory_start = std::chrono::high_resolution_clock::now();
    d_voxel_keys_.clear();
    d_voxel_keys_.resize(input_count);
    auto memory_end = std::chrono::high_resolution_clock::now();
    float memory_time = std::chrono::duration<float, std::milli>(memory_end - memory_start).count();

    // Step 2: 计算体素keys
    auto kernel_start = std::chrono::high_resolution_clock::now();
    dim3 block(256);
    dim3 grid((input_count + block.x - 1) / block.x);

    VoxelFilter::computeVoxelKeysKernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_temp_points_.data()),
        thrust::raw_pointer_cast(d_voxel_keys_.data()),
        voxel_size,
        static_cast<int>(input_count));

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        std::cerr << "[ERROR] Voxel kernel failed: " << cudaGetErrorString(kernel_error) << std::endl;
        return;
    }
    cudaDeviceSynchronize();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    float kernel_time = std::chrono::duration<float, std::milli>(kernel_end - kernel_start).count();

    // Step 3: 大小检查
    auto check_start = std::chrono::high_resolution_clock::now();
    if (d_voxel_keys_.size() != input_count)
    {
        std::cerr << "[ERROR] Voxel keys size mismatch: " << d_voxel_keys_.size()
                  << " vs " << input_count << std::endl;
        d_voxel_keys_.resize(input_count);
    }

    if (d_temp_points_.size() != input_count)
    {
        std::cerr << "[ERROR] Temp points size mismatch: " << d_temp_points_.size()
                  << " vs " << input_count << std::endl;
        return;
    }
    auto check_end = std::chrono::high_resolution_clock::now();
    float check_time = std::chrono::duration<float, std::milli>(check_end - check_start).count();

    // Step 4: GPU排序尝试，CPU排序作为最终fallback
    auto sort_start = std::chrono::high_resolution_clock::now();
    bool sort_success = false;

    // 🔥 首先尝试thrust::sort_by_key
    try
    {
        auto keys_first = d_voxel_keys_.begin();
        auto keys_last = keys_first + input_count;
        auto values_first = d_temp_points_.begin();

        thrust::sort_by_key(keys_first, keys_last, values_first);
        sort_success = true;
        std::cout << "[INFO] GPU sort_by_key succeeded" << std::endl;
    }
    catch (const thrust::system::system_error &e)
    {
        std::cerr << "[WARNING] Thrust sort_by_key failed: " << e.what() << std::endl;

        // 备用方案：stable_sort_by_key
        try
        {
            thrust::stable_sort_by_key(d_voxel_keys_.begin(),
                                       d_voxel_keys_.begin() + input_count,
                                       d_temp_points_.begin());
            sort_success = true;
            std::cout << "[INFO] GPU stable_sort_by_key succeeded" << std::endl;
        }
        catch (const thrust::system::system_error &e2)
        {
            std::cerr << "[WARNING] Stable sort also failed: " << e2.what() << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[WARNING] Generic exception in GPU sort: " << e.what() << std::endl;
    }

    // 🔄 最终fallback：CPU排序
    if (!sort_success)
    {
        std::cout << "[INFO] Falling back to CPU sort..." << std::endl;
        if (!cpuFallbackSort(input_count))
        {
            return;
        }
    }

    auto sort_end = std::chrono::high_resolution_clock::now();
    float sort_time = std::chrono::duration<float, std::milli>(sort_end - sort_start).count();

    // Step 5: 后续处理
    auto process_start = std::chrono::high_resolution_clock::now();
    processVoxelCentroids(input_count);
    auto process_end = std::chrono::high_resolution_clock::now();
    float process_time = std::chrono::duration<float, std::milli>(process_end - process_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    std::cout << "[VoxelFilter] Timing breakdown:" << std::endl;
    std::cout << "  Memory setup: " << memory_time << " ms" << std::endl;
    std::cout << "  Kernel compute: " << kernel_time << " ms" << std::endl;
    std::cout << "  Size check: " << check_time << " ms" << std::endl;
    std::cout << "  CPU sort: " << sort_time << " ms" << std::endl;
    std::cout << "  Process centroids: " << process_time << " ms" << std::endl;
    std::cout << "  Total: " << total_time << " ms" << std::endl;
}

// 🚀 基数排序实现 - 专门优化64位整数keys
void radixSort(std::vector<size_t> &indices, const std::vector<uint64_t> &keys)
{
    const size_t n = indices.size();
    if (n <= 1)
        return;

    std::vector<size_t> temp_indices(n);
    const int RADIX_BITS = 8;                                  // 每次处理8位
    const int RADIX_SIZE = 1 << RADIX_BITS;                    // 256
    const int NUM_PASSES = (64 + RADIX_BITS - 1) / RADIX_BITS; // 8次遍历

    for (int pass = 0; pass < NUM_PASSES; ++pass)
    {
        // 计数数组
        std::vector<int> count(RADIX_SIZE, 0);
        int shift = pass * RADIX_BITS;

        // 统计每个桶的元素数量
        for (size_t i = 0; i < n; ++i)
        {
            int digit = (keys[indices[i]] >> shift) & (RADIX_SIZE - 1);
            count[digit]++;
        }

        // 转换为累积计数
        for (int i = 1; i < RADIX_SIZE; ++i)
        {
            count[i] += count[i - 1];
        }

        // 从后往前分配到临时数组
        for (int i = static_cast<int>(n) - 1; i >= 0; --i)
        {
            int digit = (keys[indices[i]] >> shift) & (RADIX_SIZE - 1);
            temp_indices[--count[digit]] = indices[i];
        }

        // 复制回原数组
        indices = temp_indices;
    }
}

bool GPUPreprocessor::cpuFallbackSort(size_t input_count)
{
    auto cpu_total_start = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Using CPU radix sort fallback..." << std::endl;

    try
    {
        // Step 1: 下载数据到CPU
        auto download_start = std::chrono::high_resolution_clock::now();
        thrust::host_vector<GPUPoint3f> h_points = d_temp_points_;
        thrust::host_vector<uint64_t> h_keys = d_voxel_keys_;
        auto download_end = std::chrono::high_resolution_clock::now();
        float download_time = std::chrono::duration<float, std::milli>(download_end - download_start).count();

        // Step 2: 创建索引
        auto index_start = std::chrono::high_resolution_clock::now();
        std::vector<size_t> indices(input_count);
        std::iota(indices.begin(), indices.end(), 0);
        auto index_end = std::chrono::high_resolution_clock::now();
        float index_time = std::chrono::duration<float, std::milli>(index_end - index_start).count();

        // 🔍 调试：检查原始keys
        std::vector<uint64_t> std_keys(h_keys.begin(), h_keys.end());
        std::cout << "[DEBUG] First 10 voxel keys: ";
        for (size_t i = 0; i < std::min(size_t(10), input_count); ++i)
        {
            std::cout << std_keys[i] << " ";
        }
        std::cout << std::endl;

        // 检查是否所有keys都相同
        uint64_t first_key = std_keys[0];
        bool all_same = true;
        for (size_t i = 1; i < input_count; ++i)
        {
            if (std_keys[i] != first_key)
            {
                all_same = false;
                break;
            }
        }
        std::cout << "[DEBUG] All keys same? " << (all_same ? "YES" : "NO") << std::endl;

        // Step 3: CPU基数排序 (专门优化64位keys)
        auto sort_start = std::chrono::high_resolution_clock::now();

        if (all_same)
        {
            std::cout << "[WARNING] All voxel keys are identical - skipping sort" << std::endl;
        }
        else
        {
            radixSort(indices, std_keys);
        }

        auto sort_end = std::chrono::high_resolution_clock::now();
        float sort_time = std::chrono::duration<float, std::milli>(sort_end - sort_start).count();

        // 🔍 调试：检查排序后的前几个索引
        std::cout << "[DEBUG] First 10 sorted indices: ";
        for (size_t i = 0; i < std::min(size_t(10), input_count); ++i)
        {
            std::cout << indices[i] << " ";
        }
        std::cout << std::endl; // Step 4: 重新排列数据
        auto rearrange_start = std::chrono::high_resolution_clock::now();
        thrust::host_vector<GPUPoint3f> sorted_points(input_count);
        thrust::host_vector<uint64_t> sorted_keys(input_count);

        for (size_t i = 0; i < input_count; ++i)
        {
            sorted_points[i] = h_points[indices[i]];
            sorted_keys[i] = h_keys[indices[i]];
        }
        auto rearrange_end = std::chrono::high_resolution_clock::now();
        float rearrange_time = std::chrono::duration<float, std::milli>(rearrange_end - rearrange_start).count();

        // Step 5: 上传回GPU
        auto upload_start = std::chrono::high_resolution_clock::now();
        d_temp_points_ = sorted_points;
        d_voxel_keys_ = sorted_keys;
        auto upload_end = std::chrono::high_resolution_clock::now();
        float upload_time = std::chrono::duration<float, std::milli>(upload_end - upload_start).count();

        auto cpu_total_end = std::chrono::high_resolution_clock::now();
        float cpu_total_time = std::chrono::duration<float, std::milli>(cpu_total_end - cpu_total_start).count();

        std::cout << "[CPUSort] Detailed timing breakdown (Radix Sort):" << std::endl;
        std::cout << "  GPU->CPU download: " << download_time << " ms" << std::endl;
        std::cout << "  Index creation: " << index_time << " ms" << std::endl;
        std::cout << "  CPU radix sort: " << sort_time << " ms" << std::endl;
        std::cout << "  Data rearrange: " << rearrange_time << " ms" << std::endl;
        std::cout << "  CPU->GPU upload: " << upload_time << " ms" << std::endl;
        std::cout << "  CPU total: " << cpu_total_time << " ms" << std::endl;

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] CPU fallback sort failed: " << e.what() << std::endl;
        return false;
    }
}

// 🔧 将后续处理拆分为独立函数
void GPUPreprocessor::processVoxelCentroids(size_t input_count)
{
    // Step 3: 计算体素质心
    thrust::device_vector<int> d_point_counts(input_count);
    thrust::device_vector<int> d_ones(input_count, 1);

    d_unique_keys_.resize(input_count);
    thrust::device_vector<GPUPoint3f> d_temp_centroids(input_count);

    // reduce_by_key计算
    auto count_end = thrust::reduce_by_key(
        d_voxel_keys_.begin(), d_voxel_keys_.begin() + input_count,
        d_ones.begin(),
        d_unique_keys_.begin(),
        d_point_counts.begin());

    auto sum_end = thrust::reduce_by_key(
        d_voxel_keys_.begin(), d_voxel_keys_.begin() + input_count,
        d_temp_points_.begin(),
        d_unique_keys_.begin(),
        d_temp_centroids.begin(),
        thrust::equal_to<uint64_t>(),
        [] __device__(const GPUPoint3f &a, const GPUPoint3f &b)
        {
            return GPUPoint3f{a.x + b.x, a.y + b.y, a.z + b.z};
        });

    size_t unique_count = count_end.second - d_point_counts.begin();

    std::cout << "Found " << unique_count << " unique voxels" << std::endl;

    if (unique_count == 0)
    {
        std::cerr << "[WARNING] No unique voxels found!" << std::endl;
        d_output_points_.clear();
        d_temp_points_.clear();
        return;
    }

    // Step 4: 计算平均值
    thrust::transform(
        d_temp_centroids.begin(), d_temp_centroids.begin() + unique_count,
        d_point_counts.begin(),
        d_temp_centroids.begin(),
        [] __device__(const GPUPoint3f &sum_point, int count)
        {
            float inv_count = 1.0f / count;
            return GPUPoint3f{
                sum_point.x * inv_count,
                sum_point.y * inv_count,
                sum_point.z * inv_count};
        });

    // 安全地更新输出
    if (unique_count > 0)
    {
        thrust::host_vector<GPUPoint3f> h_result(unique_count);
        thrust::copy_n(d_temp_centroids.begin(), unique_count, h_result.begin());
        d_output_points_ = h_result;
        d_temp_points_ = d_output_points_;

        std::cout << "[GPUPreprocessor] Voxel filter: " << input_count
                  << " -> " << unique_count << " points" << std::endl;
    }
    else
    {
        d_output_points_.clear();
        d_temp_points_.clear();
    }
}

void GPUPreprocessor::cuda_launchOutlierRemoval(const PreprocessConfig &config)
{
    std::cout << "[GPUPreprocessor] Starting outlier removal" << std::endl;

    size_t input_count = d_temp_points_.size();
    if (input_count == 0)
        return;

    // 直接在成员变量上操作
    dim3 block(256);
    dim3 grid((input_count + block.x - 1) / block.x);

    if (config.outlier_method == PreprocessConfig::STATISTICAL)
    {
        OutlierRemoval::statisticalOutlierKernel<<<grid, block>>>(
            thrust::raw_pointer_cast(d_temp_points_.data()),
            thrust::raw_pointer_cast(d_valid_flags_.data()),
            input_count, config.statistical_k, config.statistical_stddev);
    }
    else
    {
        OutlierRemoval::radiusOutlierKernel<<<grid, block>>>(
            thrust::raw_pointer_cast(d_temp_points_.data()),
            thrust::raw_pointer_cast(d_valid_flags_.data()),
            input_count, config.radius_search, config.min_radius_neighbors);
    }
    cudaDeviceSynchronize();

    // ✅ 直接过滤到临时存储，然后安全赋值
    thrust::device_vector<GPUPoint3f> d_temp_result(input_count);

    auto new_end = thrust::copy_if(
        d_temp_points_.begin(), d_temp_points_.begin() + input_count,
        d_valid_flags_.begin(),
        d_temp_result.begin(),
        thrust::identity<bool>());

    size_t output_count = new_end - d_temp_result.begin();

    // 安全地更新成员变量
    if (output_count > 0)
    {
        thrust::host_vector<GPUPoint3f> h_result(output_count);
        thrust::copy_n(d_temp_result.begin(), output_count, h_result.begin());
        d_output_points_ = h_result;
    }
    else
    {
        d_output_points_.clear();
    }

    d_temp_points_ = d_output_points_;

    std::cout << "[GPUPreprocessor] Outlier removal: " << input_count << " -> " << output_count << " points" << std::endl;
}

void GPUPreprocessor::cuda_launchGroundRemoval(float threshold)
{
    std::cout << "[GPUPreprocessor] Starting ground removal" << std::endl;

    size_t input_count = d_temp_points_.size();
    if (input_count == 0)
        return;

    dim3 block(256);
    dim3 grid((input_count + block.x - 1) / block.x);

    // 重用d_valid_flags_作为ground_flags
    GroundRemoval::ransacGroundDetectionKernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_temp_points_.data()),
        thrust::raw_pointer_cast(d_valid_flags_.data()),
        input_count, threshold, 1000);
    cudaDeviceSynchronize();

    // ✅ 直接过滤非地面点
    thrust::device_vector<GPUPoint3f> d_temp_result(input_count);

    auto new_end = thrust::copy_if(
        d_temp_points_.begin(), d_temp_points_.begin() + input_count,
        d_valid_flags_.begin(),
        d_temp_result.begin(),
        [] __device__(bool is_ground)
        { return !is_ground; });

    size_t output_count = new_end - d_temp_result.begin();

    // 安全地更新成员变量
    if (output_count > 0)
    {
        thrust::host_vector<GPUPoint3f> h_result(output_count);
        thrust::copy_n(d_temp_result.begin(), output_count, h_result.begin());
        d_output_points_ = h_result;
    }
    else
    {
        d_output_points_.clear();
    }

    d_temp_points_ = d_output_points_;

    std::cout << "[GPUPreprocessor] Ground removal: " << input_count << " -> " << output_count << " points" << std::endl;
}

void GPUPreprocessor::cuda_compactValidPoints()
{
    size_t input_count = d_temp_points_.size();
    if (input_count == 0)
        return;

    // ✅ 直接使用thrust::copy_if进行压缩
    thrust::device_vector<GPUPoint3f> d_temp_result(input_count);

    auto new_end = thrust::copy_if(
        d_temp_points_.begin(), d_temp_points_.begin() + input_count,
        d_valid_flags_.begin(),
        d_temp_result.begin(),
        thrust::identity<bool>());

    size_t output_count = new_end - d_temp_result.begin();

    // 安全地更新成员变量
    if (output_count > 0)
    {
        thrust::host_vector<GPUPoint3f> h_result(output_count);
        thrust::copy_n(d_temp_result.begin(), output_count, h_result.begin());
        d_output_points_ = h_result;
    }
    else
    {
        d_output_points_.clear();
    }

    d_temp_points_ = d_output_points_;
}

void GPUPreprocessor::cuda_uploadGPUPoints(const std::vector<GPUPoint3f> &cpu_points)
{
    if (cpu_points.empty())
        return;

    auto start = std::chrono::high_resolution_clock::now();

    // 直接使用预分配的空间，不再resize
    size_t required_size = cpu_points.size();
    if (d_input_points_.size() < required_size)
    {
        std::cerr << "[ERROR] Input size (" << required_size
                  << ") exceeds pre-allocated capacity (" << d_input_points_.size() << ")" << std::endl;
        return;
    }

    // 🔥 关键：直接使用原始CUDA内存传输到预分配的空间
    cudaError_t err = cudaMemcpy(
        thrust::raw_pointer_cast(d_input_points_.data()), // 预分配的GPU空间
        cpu_points.data(),                                // CPU源
        cpu_points.size() * sizeof(GPUPoint3f),           // 字节数
        cudaMemcpyHostToDevice                            // 传输方向
    );
    if (err != cudaSuccess)
    {
        std::cerr << "[ERROR] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 🚀 GPU内部拷贝（超快）
    err = cudaMemcpy(
        thrust::raw_pointer_cast(d_temp_points_.data()),
        thrust::raw_pointer_cast(d_input_points_.data()),
        cpu_points.size() * sizeof(GPUPoint3f),
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "[ERROR] GPU internal copy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 确保传输完成
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    float upload_time = std::chrono::duration<float, std::milli>(end - start).count();

    std::cout << "[GPUPreprocessor] ⚡ FAST upload: " << cpu_points.size()
              << " points in " << upload_time << " ms (pre-allocated)" << std::endl;
}

void GPUPreprocessor::convertToPointsWithNormals()
{
    size_t point_count = d_temp_points_.size();
    if (point_count == 0)
        return;

    // ✅ 在 .cu 文件中，resize 应该工作正常
    // d_output_points_normal_.clear();
    // d_output_points_normal_.resize(point_count);

    // cuda_convertToPointsWithNormals(
    //     thrust::raw_pointer_cast(d_temp_points_.data()),
    //     thrust::raw_pointer_cast(d_output_points_normal_.data()),
    //     point_count);
}
void GPUPreprocessor::reserveMemory(size_t max_points)
{
    // 使用resize()而不是reserve()来预分配内存
    d_input_points_.resize(max_points);
    d_temp_points_.resize(max_points);
    d_output_points_.resize(max_points);
    // d_output_points_normal_.resize(max_points);
    d_voxel_keys_.resize(max_points);
    d_valid_flags_.resize(max_points);

    std::cout << "[GPUPreprocessor] Pre-allocated memory for " << max_points << " points" << std::endl;
}