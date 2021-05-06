/**
 * @file resnet.hpp
 * @author Aakash Kaushik
 * 
 * Definition of ResNet models.
 * 
 * For more information, kindly refer to the following paper.
 * 
 * Paper for ResNet.
 *
 * @code
 * @article{Kaiming He2015,
 *  author = {Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun},
 *  title = {Deep Residual Learning for Image Recognition},
 *  year = {2015},
 *  url = {https://arxiv.org/pdf/1512.03385.pdf}
 * }
 * @endcode
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_RESNET_RESNET_HPP
#define MODELS_MODELS_RESNET_RESNET_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>

namespace mlpack {
namespace ann /* Artificial neural networks */{
namespace models {

template<
  typename OutputLayerType = CrossEntropyError<>
  typename InitializationRuleType = RandomInitialization,
  size_t ResNetVersion = 18
>
class ResNet{
 public:

  ResNet();

  ResNet(const size_t inputChannel,
         const size_t inputWidth,
         const size_t inputHeight,
         const size_t numClasses = 1000,
         const std::string& weights = "none",
         const bool includeTop = true);

  ResNet(std::tuple<size_t, size_t, size_t> inputShape,
         const size_t numClasses = 1000,
         const std::string& weights = "none",
         const bool includeTop = true);

  FFN<OutputLayerType, InitializationRuleType> GetModel() { return resNet; }

  void LoadModel(const std::string& filePath);

  void SaveModel(const std::string& filepath);

 private:
  FFN<OutputLayerType, InitializationRuleType> resNet;
  size_t inputChannel;
  size_t inputWidth;
  size_t inputHeight;
  size_t numClasses;
  std::string weights;
}; // ResNet class

} // namespace models
} // namespace ann
} // namespace mlpack

#include "resnet_impl.hpp"

#endif
