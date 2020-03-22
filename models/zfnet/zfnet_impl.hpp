/**
 * @file zfnet_impl.hpp
 * @author Prince Gupta
 *
 * Implementation of ZFNet using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_ZFNET_IMPL_HPP
#define MODELS_ZFNET_IMPL_HPP

// In case it isn't already included.
#include "zfnet.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

ZFNet::ZFNet(const size_t inputChannel,
             const size_t inputWidth,
             const size_t inputHeight,
             const size_t numClasses,
             const std::string& weights) :
             ZFNet(
                std::tuple<size_t, size_t, size_t>(
                    inputChannel,
                    inputWidth,
                    inputHeight),
                numClasses,
                weights)
{
  // Nothing to do here.
}

ZFNet::ZFNet(const std::tuple<size_t, size_t, size_t> inputShape,
             const size_t numClasses,
             const std::string &weights) :
             inputChannel(std::get<0>(inputShape)),
             inputWidth(std::get<1>(inputShape)),
             inputHeight(std::get<2>(inputShape)),
             numClasses(numClasses),
             weights(weights)
{
  if(inputWidth < 197 || inputHeight < 197)
  {
    std::cout << "Min dimensions for input are 197x197";
    return;
  }

  zfnet = new FFN<>;
  ConvolutionBlock(inputChannel, 96, 7, 7, 2, 2, 0, 0);
  MaxPoolingBlock(3, 3, 2, 2);

  ConvolutionBlock(96, 256, 5, 5, 2, 2, 0, 0);
  MaxPoolingBlock(3, 3, 2, 2);

  ConvolutionBlock(256, 256, 3, 3, 1, 1, 1, 1);
  ConvolutionBlock(256, 384, 3, 3, 1, 1, 1, 1);
  ConvolutionBlock(384, 256, 3, 3, 1, 1, 1, 1);
  MaxPoolingBlock(3, 3, 2, 2);

  zfnet->Add<Linear<>>(256 * inputWidth * inputHeight, 4096);
  zfnet->Add<BatchNorm<>>(4096);
  zfnet->Add<ReLULayer<>>();
  zfnet->Add<Dropout<>>();

  zfnet->Add<Linear<>>(4096, 4096);
  zfnet->Add<BatchNorm<>>(4096);
  zfnet->Add<ReLULayer<>>();
  zfnet->Add<Dropout<>>();

  zfnet->Add<Linear<>>(4096, numClasses);

}


FFN<>* ZFNet::LoadModel(const std::string &filePath)
{
  std::cout << "Loading model" << std::endl;
  data::Load(filePath, "ZFNet", zfnet);
  return zfnet;
}

void ZFNet::SaveModel(const std::string &filePath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filePath, "ZFNet", zfnet);
  std::cout << "Model saved in " << filePath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
