/**
 * @file zfnet.hpp
 * @author Prince Gupta
 *
 * Definition of ZFNet model.
 *
 * For more information, kindly refer to the following paper.
 * 
 * @code
 * @article{
 *  author = {Matthew D Zeiler, Rob Fergus},
 *  title = {Visualizing and Understanding Convolutional Networks},
 *  journal = {IEEE},
 *  year = {2013},
 *  url = {https://arxiv.org/abs/1311.2901}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_ZFNET_HPP
#define MODELS_ZFNET_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

namespace mlpack {
namespace ann /** Artifical Neural Network. */ {

/**
 * Definition of ZFNet CNN.
 */
class ZFNet {
  
 public:
  //! Create ZFNet object
  ZFNet();

  /**
   * ZFNet contructor, initializes input shape and number of classes.
   *
   * @param inputChannels Number of input channels of the input image.
   * @param inputWidth Width of the input image.
   * @param inputHeight Height of the input image.
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param weights One of 'none', 'imagenet'(pre-training on mnist) or path to weights.
   */
  ZFNet(const size_t inputChannel,
        const size_t inputWidth,
        const size_t inputHeight,
        const size_t numClasses = 1000,
        const std::string& weights = "none");

  ZFNet(const std::tuple<size_t, size_t, size_t> inputShape,
        const size_t numClasses = 1000,
        const std::string& weights = "none");

  //! Get the model.
  FFN<>* GetModel() { return zfnet; }

  //! Load weights into the model.
  FFN<>* LoadModel(const std::string& filePath);

  //! Save weights for the model.
  void SaveModel(const std::string& filePath);

 private:
  /**
   * Adds Convolution Block.
   * 
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   */
  void ConvolutionBlock(const size_t inSize,
                        const size_t outSize,
                        const size_t kernelWidth,
                        const size_t kernelHeight,
                        const size_t strideWidth = 1,
                        const size_t strideHeight = 1,
                        const size_t padW = 0,
                        const size_t padH = 0)
  {
    zfnet->Add<Convolution<>>(inSize, outSize, kernelWidth,
        kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
        inputHeight);
    zfnet->Add<ReLULayer<>>();

    // Update inputWidth and input Height.
    inputWidth = layerOutSize(inputWidth, kernelWidth, strideWidth, padW);
    inputHeight = layerOutSize(inputHeight, kernelHeight, strideHeight, padH);
    return;
   }

  /**
   * Adds Max Pooling Block.
   *
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   */
  void MaxPoolingBlock(const size_t kernelWidth,
                    const size_t kernelHeight,
                    const size_t strideWidth,
                    const size_t strideHeight)
  {
    zfnet->Add<MaxPooling<>>(kernelWidth, kernelHeight,
        strideWidth, strideHeight, true);
    // Update inputWidth and inputHeight.
    inputWidth = layerOutSize(inputWidth, kernelWidth, strideWidth, 0);
    inputHeight = layerOutSize(inputHeight, kernelHeight, strideHeight, 0);
    return;
  }

  /**
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param padding The size of the padding (width or height) on one side.
   * @return The convolution output size.
   */
  size_t layerOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t padding)
  {
    return std::floor((size + 2 * padding - k) / s) + 1;
  }

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored type of pre-trained weights.
  std::string weights;

  //! Locally stored ZFNet Model.
  FFN<>* zfnet;
}; // class ZFNet

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "zfnet_impl.hpp"

#endif
