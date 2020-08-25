/**
 * @file yolo_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of YOLO models using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_YOLO_IMPL_HPP
#define MODELS_YOLO_IMPL_HPP

#include "yolo.hpp"

namespace mlpack {
namespace ann {

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    std::string YOLOVersion = "v1-tiny"
>
YOLO<OutputLayerType, InitializationRuleType, YOLOVersion>::YOLO() :
    inputChannel(0),
    inputWidth(0),
    inputHeight(0),
    numClasses(0),
    numBoxes(0),
    featureWidth(0),
    featureHeight(0),
    weights("none")
{
  // Nothing to do here.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    std::string YOLOVersion = "v1-tiny"
>
YOLO<OutputLayerType, InitializationRuleType, YOLOVersion>::YOLO(
  const size_t inputChannel,
  const size_t inputWidth,
  const size_t inputHeight,
  const size_t numClasses,
  const size_t numBoxes,
  const size_t featureWidth,
  const size_t featureHeight,
  const std::string& weights,
  const bool includeTop) :
  YOLO<OutputLayerType, InitializationRuleType, YOLOVersion>(
    std::tuple<size_t, size_t, size_t>(
      inputChannel,
      inputWidth,
      inputHeight),
      numClasses,
      numBoxes,
      std::tuple<size_t, size_t>(featureWidth, featureHeight),
      weights,
      includeTop)
{
  // Nothing to do here.
}

template<
     typename OutputLayerType,
     typename InitializationRuleType,
     std::string YOLOVersion
>
YOLO<OutputLayerType, InitializationRuleType, YOLOVersion>::YOLO(
    const std::tuple<size_t, size_t, size_t> inputShape,
    const size_t numClasses,
    const size_t numBoxes,
    const std::tuple<size_t, size_t> featureShape,
    const std::string& weights,
    const bool includeTop) :
    inputChannel(std::get<0>(inputShape)),
    inputWidth(std::get<1>(inputShape)),
    inputHeight(std::get<2>(inputShape)),
    numClasses(numClasses),
    numBoxes(numBoxes),
    featureWidth(std::get<0>(featureShape)),
    featureHeight(std::get<1>(featureShape)),
    weights(weights)
{
  std::set<string> supportedVersion({"v1-tiny"});
  mlpack::Log::Assert(supportedVersion.count(YOLOVersion),
      "Unsupported YOLO version. Trying to find :", YOLOVersion);

  if (weights == "voc")
  {
    // Download weights here.
    LoadModel("./../weights/YOLO/yolo" + YOLOVersion + "_voc.bin");
    return;
  }
  else if (weights != "none")
  {
    LoadModel(weights);
    return;
  }

  if (YOLOVersion == "v1-tiny")
  {
    yolo.Add(new IdentityLayer<>());

    // Convolution and activation function in a block.
    ConvolutionBlock(inputChannel, 16, 3, 3, 1, 1, 1, 1, true);
    PoolingBlock(2);

    size_t numBlocks = 5;
    size_t outChannels = 16;
    for (size_t blockId = 0; blockId < numBlocks; blockId++)
    {
      ConvolutionBlock(outChannels, outChannels * 2, 3, 3, 1, 1, 1, 1, true);
      PoolingBlock(2);
      outChannels *= 2;
    }

    ConvolutionBlock(outChannels, outChannels * 2, 3, 3, 1, 1, 1, 1, true);
    outChannels *= 2;
    ConvolutionBlock(outChannels, 256, 3, 3, 1, 1, 1, 1, true);
    outChannels 256;

    if (includeTop)
    {
      yolo.Add(new Linear<>(inputWidth * inputHeight * outChannels,
          featureWidth * featureHeight * (5 * numBoxes + numClasses)));
      yolo.Add(new Sigmoid<>());
    }

    yolo.ResetParameters();
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    std::string YOLOVersion
>
void YOLO<
    OutputLayerType, InitializationRuleType, YOLOVersion
>::LoadModel(const std::string& filePath)
{
  data::Load(filePath, "yolo" + YOLOVersion, yolo);
  Log::Info << "Loaded model." << std::endl;
}

template<
     typename OutputLayerType,
     typename InitializationRuleType,
     std::string YOLOVersion
>
void YOLO<
    OutputLayerType, InitializationRuleType, YOLOVersion
>::SaveModel(const std::string& filePath)
{
  Log::Info<< "Saving model." << std::endl;
  data::Save(filePath, "yolo" + YOLOVerson, yolo);
  Log::Info << "Model saved in " << filePath << "." << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
