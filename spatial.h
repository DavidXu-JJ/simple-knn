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

#include <torch/extension.h>

torch::Tensor dist3knn(const torch::Tensor& points);

torch::Tensor dist10knn(const torch::Tensor& points);

torch::Tensor meanDistFromReferencePcd(const torch::Tensor& query_pcd, const torch::Tensor& reference_pcd, const bool return_index);