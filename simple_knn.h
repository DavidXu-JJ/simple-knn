/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
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