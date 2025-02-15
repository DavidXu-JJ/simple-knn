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

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
dist3knn(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn3(P, (float3*)points.contiguous().data_ptr<float>(), means.contiguous().data_ptr<float>());

  return means;
}

torch::Tensor
dist10knn(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);

  SimpleKNN::knn10(P, (float3*)points.contiguous().data_ptr<float>(), means.contiguous().data_ptr<float>());

  return means;
}


torch::Tensor
meanDistFromReferencePcd(const torch::Tensor& query_pcd, const torch::Tensor& reference_pcd, const bool return_index)
{
  const int query_num = query_pcd.size(0);
  const int reference_num = reference_pcd.size(0);

  auto int_opts = query_pcd.options().dtype(torch::kInt32);
  auto float_opts = query_pcd.options().dtype(torch::kFloat32);

  torch::Tensor knn_index_in_reference;
  if (return_index)
      knn_index_in_reference = torch::full({query_num , 3}, 0, int_opts);
  else
      knn_index_in_reference = torch::full({0}, 0, int_opts);

  torch::Tensor means = torch::full({return_index ? 0 : query_num}, 0.0, float_opts);

  const int batch = 1<<8; 
  for (int i = 0; i < query_num; i += batch)
  SimpleKNN::query_knn(
      std::max(query_num - i, batch), (float3*)query_pcd.contiguous().data_ptr<float>() + i,
      reference_num, (float3*)reference_pcd.contiguous().data_ptr<float>(),
      means.contiguous().data_ptr<float>() + i, knn_index_in_reference.contiguous().data_ptr<int>() + i,
      return_index
  );

  return return_index ? knn_index_in_reference : means;
}
