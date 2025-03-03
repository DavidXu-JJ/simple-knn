/*
 * Edited by: Jingwei Xu, ShanghaiTech University
 * Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
*/

#include <torch/extension.h>
#include "spatial.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dist3knn", &dist3knn);
  m.def("dist10knn", &dist10knn);
  m.def("meanDistFromReferencePcd", &meanDistFromReferencePcd);
}
