/*
 * Edited by: Jingwei Xu, ShanghaiTech University
 * Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
*/

#include <torch/extension.h>

torch::Tensor dist3knn(const torch::Tensor& points);

torch::Tensor dist10knn(const torch::Tensor& points);

torch::Tensor meanDistFromReferencePcd(const torch::Tensor& query_pcd, const torch::Tensor& reference_pcd, const bool return_index);
