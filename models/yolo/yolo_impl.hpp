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
    typename InitializationRuleType
>
YOLO<OutputLayerType, InitializationRuleType>::YOLO() :
    inputChannel(0),
    inputWidth(0),
    inputHeight(0),
    numClasses(0),
    numBoxes(0),
    featureWidth(0),
    featureHeight(0),
    weights("none"),
    yoloVersion("none")
{
  // Nothing to do here.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType
>
YOLO<OutputLayerType, InitializationRuleType>::YOLO(
  const size_t inputChannel,
  const size_t inputWidth,
  const size_t inputHeight,
  const std::string yoloVersion,
  const size_t numClasses,
  const size_t numBoxes,
  const size_t featureWidth,
  const size_t featureHeight,
  const std::string& weights,
  const bool includeTop) :
  YOLO<OutputLayerType, InitializationRuleType>(
    std::tuple<size_t, size_t, size_t>(
      inputChannel,
      inputWidth,
      inputHeight),
      yoloVersion,
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
     typename InitializationRuleType
>
YOLO<OutputLayerType, InitializationRuleType>::YOLO(
    const std::tuple<size_t, size_t, size_t> inputShape,
    const std::string yoloVersion,
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
    weights(weights),
    yoloVersion(yoloVersion)
{
  std::set<std::string> supportedVersion({"v1-tiny"});
  mlpack::Log::Assert(supportedVersion.count(yoloVersion),
      "Unsupported YOLO version. Trying to find :" + yoloVersion);

  if (weights == "voc")
  {
    // Download weights here.
    LoadModel("./../weights/YOLO/yolo" + yoloVersion + "_voc.bin");
    return;
  }
  else if (weights != "none")
  {
    LoadModel(weights);
    return;
  }

  if (yoloVersion == "v1-tiny")
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
    outChannels = 256;

    if (includeTop)
    {
      yolo.Add(new Linear<>(inputWidth * inputHeight * outChannels,
          featureWidth * featureHeight * (5 * numBoxes + numClasses)));
      yolo.Add(new SigmoidLayer<>());
    }

    yolo.ResetParameters();
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType
>
void YOLO<
    OutputLayerType, InitializationRuleType
>::LoadModel(const std::string& filePath)
{
  data::Load(filePath, "yolo" + yoloVersion, yolo);
  Log::Info << "Loaded model." << std::endl;
}

template<
     typename OutputLayerType,
     typename InitializationRuleType
>
void YOLO<
    OutputLayerType, InitializationRuleType
>::SaveModel(const std::string& filePath)
{
  Log::Info<< "Saving model." << std::endl;
  data::Save(filePath, "yolo" + yoloVersion, yolo);
  Log::Info << "Model saved in " << filePath << "." << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
