/*
 * Edited by: Jingwei Xu, ShanghaiTech University
 * Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
*/

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cub/cub.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

__global__ void findNearestMortonIndices(
    uint32_t query_num, uint32_t* queried_point_morton,
    uint32_t reference_num, uint32_t* reference_morton_sorted, uint32_t* reference_indices_sorted,
    uint32_t* nearest_morton_indices
)
{
    int idx = cg::this_grid().thread_rank();

    if(idx >= query_num)
        return;

    uint32_t current_queried_point_morton = queried_point_morton[idx];

    nearest_morton_indices[idx] = reference_indices_sorted[
        cub::LowerBound(
            reference_morton_sorted,
            reference_num,
            current_queried_point_morton
        )
    ];
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

    // accessed by the thread in the same block
	__shared__ MinMax redResult[BOX_SIZE];

	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
	    // copy the minmax info of this thread to corresponding shared array index
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

        // merge the minmax of the top half to the bottom half
		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

    // the merged minmax of all thread in one block is stored in corresponding box
	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

// K means the size of heap knn
template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

// K means the size of heap knn, its index is also maintained
template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn, int index, int* index_heap)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
			int _index = index_heap[j];
			index_heap[j] = index;
		    index = _index;
		}
	}
}

__global__ void box3MeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	// a stack of size 3
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

    // stencil with length of 6
	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}

    // 3knn among the 6 neighbor point (use the reject to filter the most impossible bounding box)
	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

        // iterate the points in the possible box (the 6 neighbor point will also be iterated again)
		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

__global__ void box10MeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	// a stack of size 3
	float best[10] = { FLT_MAX, FLT_MAX, FLT_MAX , FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

    // stencil with length of 6
	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<10>(point, points[indices[i]], best);
	}

    // 3knn among the 6 neighbor point (use the reject to filter the most impossible bounding box)
	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;
    best[3] = FLT_MAX;
	best[4] = FLT_MAX;
	best[5] = FLT_MAX;
	best[6] = FLT_MAX;
	best[7] = FLT_MAX;
	best[8] = FLT_MAX;
	best[9] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

        // iterate the points in the possible box (the 6 neighbor point will also be iterated again)
		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<10>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2] + best[3] + best[4] + best[5] + best[6] + best[7] + best[8] + best[9]) / 10.0f;
}

__global__ void boxMeanDist(
    uint32_t query_num, float3* query_point_cloud, uint32_t* nearest_morton_indices,
    uint32_t reference_num, float3* reference_point_cloud, uint32_t* indices_sorted,
    uint32_t num_boxes, MinMax* boxes,
    float* dists
)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= query_num)
		return;

	float3 point = query_point_cloud[idx];
	// a stack of size 3
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

    uint32_t nearest_morton_index = nearest_morton_indices[idx];
    // stencil with length of 6
	for (int i = max(0, nearest_morton_index - 3); i <= min(reference_num - 1, nearest_morton_index + 3); i++)
	{
		updateKBest<3>(point, reference_point_cloud[indices_sorted[i]], best);
	}

    // 3knn among the 6 neighbor point (use the reject to filter the most impossible bounding box)
	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < num_boxes; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

        // iterate the points in the possible box (the 6 neighbor point will also be iterated again)
		for (int i = b * BOX_SIZE; i < min(reference_num, (b + 1) * BOX_SIZE); i++)
		{
			updateKBest<3>(point, reference_point_cloud[indices_sorted[i]], best);
		}
	}
	dists[idx] = (best[0] + best[1] + best[2]) / 3.0f;
}

__global__ void boxMeanDistIndex(
    uint32_t query_num, float3* query_point_cloud, uint32_t* nearest_morton_indices,
    uint32_t reference_num, float3* reference_point_cloud, uint32_t* indices_sorted,
    uint32_t num_boxes, MinMax* boxes,
    int* knn_index_in_reference
)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= query_num)
		return;

	float3 point = query_point_cloud[idx];
	// a stack of size 3
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

    uint32_t nearest_morton_index = nearest_morton_indices[idx];
    // stencil with length of 6
	for (int i = max(0, nearest_morton_index - 3); i <= min(reference_num - 1, nearest_morton_index + 3); i++)
	{
		updateKBest<3>(point, reference_point_cloud[indices_sorted[i]], best);
	}

    // 3knn among the 6 neighbor point (use the reject to filter the most impossible bounding box)
	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	int32_t index_heap[3] = {0, 0, 0};

	for (int b = 0; b < num_boxes; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

        // iterate the points in the possible box (the 6 neighbor point will also be iterated again)
		for (int i = b * BOX_SIZE; i < min(reference_num, (b + 1) * BOX_SIZE); i++)
		{
		    // warning: int assigned with uint32_t
		    int index_in_reference = indices_sorted[i];
			updateKBest<3>(point, reference_point_cloud[index_in_reference], best, index_in_reference, index_heap);
		}
	}
	int base_index = 3 * idx;
	knn_index_in_reference[base_index] = index_heap[0];
	knn_index_in_reference[base_index + 1] = index_heap[1];
	knn_index_in_reference[base_index + 2] = index_heap[2];
}

void SimpleKNN::knn3(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

    // get the min value of in all x,y and z
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

    // get the max value of in all x,y and z
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

    // project 3 dim data to 1 dim, close 3d point will have similar 1d morton value
	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

    // sort the index based on morton value
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	// one block for one box
	thrust::device_vector<MinMax> boxes(num_boxes);
	// generate a bounding box for the points in each block
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	// compute the mean 3knn points of every points
	box3MeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}

void SimpleKNN::knn10(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

    // get the min value of in all x,y and z
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

    // get the max value of in all x,y and z
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

    // project 3 dim data to 1 dim, close 3d point will have similar 1d morton value
	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

    // sort the index based on morton value
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	// one block for one box
	thrust::device_vector<MinMax> boxes(num_boxes);
	// generate a bounding box for the points in each block
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	// compute the mean 3knn points of every points
	box10MeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}

void SimpleKNN::query_knn(
    int query_num, float3* query_point_cloud,
    int reference_num, float3* reference_point_cloud,
    float* meanDists, int* knn_index_in_reference,
    bool return_index
)
{
    float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

    float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, reference_point_cloud, result, reference_num, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

    // get the min value of in all x,y and z
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, reference_point_cloud, result, reference_num, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

    // get the max value of in all x,y and z
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, reference_point_cloud, result, reference_num, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

    // project 3 dim data to 1 dim, close 3d point will have similar 1d morton value
	thrust::device_vector<uint32_t> morton(reference_num);
	thrust::device_vector<uint32_t> morton_sorted(reference_num);
	coord2Morton << <(reference_num + 255) / 256, 256 >> > (reference_num, reference_point_cloud, minn, maxx, morton.data().get());

    thrust::device_vector<uint32_t> indices(reference_num);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(reference_num);

    // sort the index based on morton value
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), reference_num);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), reference_num);

	// need to get an array with array[i] is the index of point in reference with nearest morton to query_point_cloud[i]
    // 1) maintain the min max of query pcd
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, query_point_cloud, result, query_num, CustomMin(), init);
	temp_storage.resize(temp_storage_bytes);

    cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, query_point_cloud, result, query_num, CustomMin(), init);
    cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

    cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, query_point_cloud, result, query_num, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

    // 2) calculate the morton of query pcd
	thrust::device_vector<uint32_t> queried_point_morton(query_num);
    coord2Morton << <(query_num + 255) / 256, 256 >> > (query_num, query_point_cloud, minn, maxx, queried_point_morton.data().get());

    // 3) maintain the lower bound index of queried point in reference point with the help of sorted reference points' indices
    thrust::device_vector<uint32_t> nearest_morton_indices(query_num);

    findNearestMortonIndices << <(query_num + 255) / 256, 256>> > (
        query_num, queried_point_morton.data().get(),
        reference_num, morton_sorted.data().get(), indices_sorted.data().get(),
        nearest_morton_indices.data().get()
    );

	uint32_t num_boxes = (reference_num + BOX_SIZE - 1) / BOX_SIZE;
	// one block for one box
	thrust::device_vector<MinMax> boxes(num_boxes);
	// generate a bounding box for the points in each block
	boxMinMax << <num_boxes, BOX_SIZE >> > (reference_num, reference_point_cloud, indices_sorted.data().get(), boxes.data().get());
	if (return_index){
        boxMeanDistIndex << <num_boxes, BOX_SIZE >> > (
            query_num, query_point_cloud, nearest_morton_indices.data().get(),
            reference_num, reference_point_cloud, indices_sorted.data().get(),
            num_boxes, boxes.data().get(),
            knn_index_in_reference
        );
    }
	else{
        boxMeanDist << <num_boxes, BOX_SIZE >> > (
            query_num, query_point_cloud, nearest_morton_indices.data().get(),
            reference_num, reference_point_cloud, indices_sorted.data().get(),
            num_boxes, boxes.data().get(),
            meanDists
        );
    }

    cudaFree(result);
}
