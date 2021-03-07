/**
 * @file vgg16.hpp
 * @author Vishwas Chepuri
 * 
 * Definition of VGG16 model.
 * 
 * For more information, kinldy refer to the following paper.
 * 
 * [Very Deep Convolutional Networks for Large-Scale Image Recognition]
 * (https://arxiv.org/abs/1409.1556) (ICLR 2015)
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_VGG16_HPP
#define MODELS_VGG16_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Definition of a VGG16 model.
 * 
 * @tparam OutputLayerType The output layer type used to evaluate the model.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
  typename OutputLayerType = CrossEntropyError<>,
  typename InitializationRuleType = RandomInitialization
>
class VGG16
{
 public:
  //! Create the VGG16 model.
  VGG16();
  
  /**
   * VGG16 constructor initializes input shape and number of classes.
   * 
   * @param includeTop : Whether to include the 3 fully-connected layers at the top of the model.
   * @param weights : One of 'none'(random initialization), 'imagenet'(pre-trained on ImageNet) or path to weights.
   * @param inputShape : A three-valued tuple indicating input shape.
   *        First value is number of channels (channels-first), second and third values are input width and height respectively.
   *        It should have exactly 3 channels and width, height should be no smaller than 32. 
   * @param numClasses : Number of classes to classify images into, only to be specified if `includeTop` is `true` 
   *        and no `weights` argument is specified. 
   */
  VGG16(const std::tuple<size_t, size_t, size_t> inputShape,
        const bool includeTop = true,
        const std::string& weights = "none",
        const size_t numClasses = 1000);

  //! Get Layers of the model.
  FFN<OutputLayerType, InitializationRuleType>& GetModel() { return vgg16Network;}

  //! Load weights into the model.
  void LoadModel(const std::string& filePath);

  //! Save weights of the model.
  void SaveModel(const std::string& filePath);

 private:

  /**
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param padding "same" or "valid" padding
   * @return The convolution output size.
   */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const std::string& padding)
  { 
    size_t pad;
    if(padding == "same")
    {
      pad = (size * (s - 1) + k - s) / 2;
    }
    else
    {
      pad = 0;
    }
    return std::floor(size + 2 * pad - k) / s + 1;
  }

  //! Locally stored VGG16 model.
  FFN<OutputLayerType, InitializationRuleType> vgg16Network;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored type of pre-trained weigths.
  std::string weights;
}; // VGG16 class.

} // namespace ann
} // namespace mlpack

# include "vgg16_impl.hpp"

#endif
