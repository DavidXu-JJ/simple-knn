/*
 * Edited by: Jingwei Xu, ShanghaiTech University
 * Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
*/

#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED

class SimpleKNN
{
public:
    static void knn3(int P, float3* points, float* meanDists);
    static void knn10(int P, float3* points, float* meanDists);
	static void query_knn(
        int query_num, float3* query_point_cloud,
        int reference_num, float3* reference_point_cloud,
        float* meanDistsm, int* knn_index_in_reference,
        bool return_index
	);
};

#endif
