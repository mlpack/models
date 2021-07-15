/**
 * @file mobilenet_v1_impl.hpp
 * @author Aakash Kaushik
 *
 * Implementation of MobileNetV1 using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_MOBILENET_MOBILENET_V1_IMPL_HPP
#define MODELS_MODELS_MOBILENET_MOBILENET_V1_IMPL_HPP

#include "mobilenet_v1.hpp"

namespace mlpack {
namespace models {

template<typename OutputLayerType, typename InitializationRuleType>
MobileNetV1<OutputLayerType, InitializationRuleType>::MobileNetV1() :
    inputChannel(0),
    inputWidth(0),
    inputHeight(0),
    numClasses(0),
    alpha(0),
    depthMultiplier(0)
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType>
MobileNetV1<OutputLayerType, InitializationRuleType>::MobileNetV1(
    const size_t inputChannel,
    const size_t inputWidth,
    const size_t inputHeight,
    const float alpha,
    const size_t depthMultiplier,
    const bool includeTop,
    const bool preTrained,
    const size_t numClasses) :
    MobileNetV1<OutputLayerType, InitializationRuleType>(
        std::tuple<size_t, size_t, size_t>(
        inputChannel,
        inputWidth,
        inputHeight),
        alpha,
        depthMultiplier,
        includeTop,
        preTrained,
        numClasses)
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType>
MobileNetV1<OutputLayerType, InitializationRuleType>::MobileNetV1(
    std::tuple<size_t, size_t, size_t> inputShape,
    const float alpha,
    const size_t depthMultiplier,
    const bool includeTop,
    const bool preTrained,
    const size_t numClasses) :
    inputChannel(std::get<0>(inputShape)),
    inputWidth(std::get<1>(inputShape)),
    inputHeight(std::get<2>(inputShape)),
    alpha(alpha),
    depthMultiplier(depthMultiplier),
    numClasses(numClasses)
{
  if (inputWidth < 32 || inputHeight < 32)
  {
    mlpack::Log::Fatal << "input width and input height cannot be smaller than"
        " 32.\nGiven input width and height: (" << inputWidth << ", "
        << inputHeight << ")" << std::endl;
  }

  if (preTrained)
  {
    if (numClasses != 1000)
    {
      mlpack::Log::Fatal << "Number of classes should be 1000 when pre-trained is"
          " true" << std::endl;
    }

    if (inputWidth != inputHeight)
    {
      mlpack::Log::Fatal << "When pre-trained is true image height should be"
          " equal to image width." << std::endl;
    }

    std::map<size_t, std::string>::iterator imageSizeString =
        imageSizeToString.find(inputWidth);
    if (imageSizeString == imageSizeToString.end())
    {
      mlpack::Log::Fatal << "Image size can only be one of the following when"
          " pre-trained is true: (128, 160, 192, 224)" << std::endl;
    }

    std::map<double, std::string>::iterator alphaString =
        alphaToString.find(alpha);
    if (alphaString == alphaToString.end())
    {
      mlpack::Log::Fatal << "Alpha can only be one of the following when"
          " pre-trained is true: (0.25, 0.5, 0.75, 1.0)" << std::endl;
    }

    std::string home = getenv("HOME");
    preTrainedPath = home + "/.cache/mlpack/models/weights/mobilenetv1/"
        "mobilenetv1_" + alphaString->second + "_" + imageSizeString->second +
        ".bin";
    if (Utils::PathExists(preTrainedPath, true) == false)
    {
      std::cout << "mobilenetv1_" + alphaString->second + "_" +
          imageSizeString->second + ".bin to " + preTrainedPath << std::endl;
      Utils::DownloadFile("mobilenetv1_" + alphaString->second + "_" +
          imageSizeString->second + ".bin", preTrainedPath, "", false, false,
          "http://models.mlpack.org/mobilenetv1/");
    }

    LoadModel(preTrainedPath);
    return;
  }
  outSize = size_t(32 * alpha);
  mobileNet.Add(new ann::Convolution<>(inputChannel, outSize, 3, 3, 2, 2,
      std::make_tuple(0, 1), std::make_tuple(0, 1), inputWidth, inputHeight));
  mlpack::Log::Info << "Convolution: " << "(" << "3, " << inputWidth + 1 << ", "
      << inputHeight + 1 << ")" << " ---> (" << outSize << ", ";
  inputWidth = ConvOutSize(inputWidth, 3, 2, 1);
  inputHeight = ConvOutSize(inputHeight, 3, 2, 1);
  mlpack::Log::Info << inputWidth << ", " << inputHeight << ")" << std::endl;
  mobileNet.Add(new ann::BatchNorm<>(outSize, 1e-3, true));
  mlpack::Log::Info << "BatchNorm: " << "(" << outSize << ")"
        << " ---> (" << outSize << ")" << std::endl;
  ReLU6Layer();
  outSize = DepthWiseConvBlock(outSize, 64, alpha, depthMultiplier);

  for (const auto& blockConfig : mobileNetConfig)
  {
    outSize = DepthWiseConvBlock(outSize, blockConfig.first, alpha,
        depthMultiplier, 2);

    for (size_t numBlock = 1; numBlock < blockConfig.second; ++numBlock)
    {
      outSize = DepthWiseConvBlock(outSize, blockConfig.first, alpha,
          depthMultiplier);
    }
  }

  mobileNet.Add(new ann::AdaptiveMeanPooling<>(1, 1));
  mlpack::Log::Info << "Adaptive mean pooling: (" << size_t(1024 * alpha)
      << ", " << inputWidth << ", " << inputHeight << ") ---> ("
      << size_t(1024 * alpha) << ", 1, 1)" << std::endl;

  if (includeTop)
  {
    mobileNet.Add(new ann::Dropout<>(1e-3));
    mlpack::Log::Info << "Dropout" << std::endl;
    mobileNet.Add(new ann::Convolution<>(size_t(1024 * alpha), numClasses, 1,
        1, 1, 1, 0, 0, 1, 1, "same"));
    mlpack::Log::Info << "Convolution: (" << size_t(1024 * alpha)
        << ", 1, 1) ---> (" << numClasses << " , 1, 1)" << std::endl;
    mobileNet.Add(new ann::Softmax<>);
    mlpack::Log::Info << "Softmax" << std::endl;
  }

  // Reset parameters for a new network.
  mobileNet.ResetParameters();
}

template<typename OutputLayerType, typename InitializationRuleType>
void MobileNetV1<OutputLayerType, InitializationRuleType>::LoadModel(
    const std::string& filePath)
{
  data::Load(filePath, "mobilenet_v1", mobileNet);
  Log::Info << "Loaded model" << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType>
void MobileNetV1<OutputLayerType, InitializationRuleType>::SaveModel(
    const std::string& filePath)
{
  Log::Info<< "Saving model." << std::endl;
  data::Save(filePath, "mobilenet_v1", mobileNet);
  Log::Info << "Model saved in " << filePath << "." << std::endl;
}

} // namespace models
} // namespace mlpack

#endif
