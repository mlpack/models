/**
 * @file methods/ann/layer/separable_convolution_impl.hpp
 * @author Kartik Dutt
 * @author Aakash Kaushik
 * @author Sidharth
 *
 * Implementation of the Separable Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SEPARABLE_CONVOLUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SEPARABLE_CONVOLUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "separable_convolution.hpp"

namespace mlpack {

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::SeparableConvolutionType()
{
  // Nothing to do here.
}

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::SeparableConvolutionType(
    const size_t inSize,
    const size_t outSize,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const size_t padW,
    const size_t padH,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t numGroups,
    const std::string &paddingType) :
    inSize(inSize),
    outSize(outSize),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    padWLeft(padW),
    padWRight(padW),
    padHBottom(padH),
    padHTop(padH),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    outputWidth(0),
    outputHeight(0),
    numGroups(numGroups)
{
  if (inSize % numGroups != 0 || outSize % numGroups != 0)
  {
    Log::Fatal << "The output maps / input maps is not possible given "
        << "the number of groups. Input maps / output maps must be "
        << " divisible by number of groups." << std::endl;
  }

  weights.set_size((outSize * (inSize / numGroups) * kernelWidth *
      kernelHeight) + outSize, 1);

  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });

  if (paddingTypeLow == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding();
  }

  padding = Padding(padWLeft, padWRight, padHTop, padHBottom);
}

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::SeparableConvolutionType(
    const size_t inSize,
    const size_t outSize,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const std::tuple<size_t, size_t> padW,
    const std::tuple<size_t, size_t> padH,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t numGroups,
    const std::string &paddingType) :
    inSize(inSize),
    outSize(outSize),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    padWLeft(std::get<0>(padW)),
    padWRight(std::get<1>(padW)),
    padHBottom(std::get<1>(padH)),
    padHTop(std::get<0>(padH)),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    outputWidth(0),
    outputHeight(0),
    numGroups(numGroups)
{
  if (inSize % numGroups != 0 || outSize % numGroups != 0)
  {
    Log::Fatal << "The output maps / input maps is not possible given "
        << "the number of groups. Input maps / output maps must be "
        << " divisible by number of groups." << std::endl;
  }

  weights.set_size((outSize * (inSize / numGroups) * kernelWidth *
      kernelHeight) + outSize, 1);

  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });

  if (paddingTypeLow == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding();
  }

  padding = Padding(padWLeft, padWRight, padHTop, padHBottom);
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::Reset()
{
  weight = arma::cube(weights.memptr(), kernelWidth, kernelHeight,
      outSize * (inSize / numGroups), false, false);
  bias = arma::mat(weights.memptr() + weight.n_elem,
      outSize, 1, false, false);
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::Forward(const MatType& input, MatType& output)
{
  typedef typename arma::Cube<typename MatType::elem_type> CubeType;
  batchSize = input.n_cols;
  inputTemp = arma::cube(const_cast<MatType&>(input).memptr(),
      inputWidth, inputHeight, inSize * batchSize, false, false);

  if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
  {
    inputPaddedTemp.set_size(inputTemp.n_rows + padWLeft + padWRight,
        inputTemp.n_cols + padHTop + padHBottom, inputTemp.n_slices);

    for (size_t i = 0; i < inputTemp.n_slices; ++i)
    {
      padding.Forward(inputTemp.slice(i), inputPaddedTemp.slice(i));
    }
  }

  size_t wConv = ConvOutSize(inputWidth, kernelWidth, strideWidth, padWLeft,
      padWRight);
  size_t hConv = ConvOutSize(inputHeight, kernelHeight, strideHeight, padHTop,
      padHBottom);

  output.set_size(wConv * hConv * outSize, batchSize);
  outputTemp = CubeType(output.memptr(), wConv, hConv,
      outSize * batchSize, false, false);
  outputTemp.zeros();

  for (size_t curGroup = 0; curGroup < numGroups; curGroup++)
  {
    for (size_t outMap = outSize * curGroup * batchSize / numGroups,
        outMapIdx = outSize * curGroup / numGroups, batchCount = 0;
        outMap < outSize * (curGroup + 1) * batchSize / numGroups; outMap++)
    {
      if (outMap != 0 && outMap % outSize == 0)
      {
        batchCount++;
        outMapIdx = 0;
      }

      for (size_t inMap = inSize * curGroup / numGroups;
          inMap < inSize * (curGroup + 1) / numGroups; inMap++, outMapIdx++)
      {
        MatType convOutput;
        if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
        {
          ForwardConvolutionRule::Convolution(inputPaddedTemp.slice(inMap +
              batchCount * inSize), weight.slice(outMapIdx), convOutput,
              strideWidth, strideHeight);
        }
        else
        {
          ForwardConvolutionRule::Convolution(inputTemp.slice(inMap +
              batchCount * inSize), weight.slice(outMapIdx), convOutput,
              strideWidth, strideHeight);
        }

        outputTemp.slice(outMap) += convOutput;
      }
      outputTemp.slice(outMap) += bias(outMap % outSize);
    }
  }
  outputWidth = outputTemp.n_rows;
  outputHeight = outputTemp.n_cols;
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::Backward(const MatType& /* input */,
            const MatType& gy,
            MatType& g)
{
  typedef typename arma::Cube<typename MatType::elem_type> CubeType;
  arma::cube mappedError(const_cast<MatType&>(gy).memptr(), outputWidth,
      outputHeight, outSize * batchSize, false, false);

  g.set_size(inputTemp.n_rows * inputTemp.n_cols * inSize, batchSize);
  gTemp = CubeType(g.memptr(), inputTemp.n_rows,
      inputTemp.n_cols, inputTemp.n_slices, false, false);
  gTemp.zeros();

  for (size_t curGroup = 0; curGroup < numGroups; curGroup++)
  {
    for (size_t outMap = outSize * curGroup * batchSize / numGroups,
        outMapIdx = outSize * curGroup / numGroups, batchCount = 0;
        outMap < outSize * (curGroup + 1) * batchSize / numGroups; outMap++)
    {
      if (outMap != 0 && outMap % outSize == 0)
      {
        batchCount++;
        outMapIdx = 0;
      }

      for (size_t inMap = inSize * curGroup / numGroups;
          inMap < inSize * (curGroup + 1) / numGroups; inMap++, outMapIdx++)
      {
        MatType output, rotatedFilter;
        Rotate180(weight.slice(outMapIdx), rotatedFilter);
        BackwardConvolutionRule::Convolution(mappedError.slice(outMap),
        rotatedFilter, output, strideWidth, strideHeight);

        if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
        {
          gTemp.slice(inMap + batchCount * inSize) += output.submat(padWLeft,
            padHTop, padWLeft + gTemp.n_rows - 1, padHTop + gTemp.n_cols - 1);
        }
        else
        {
          gTemp.slice(inMap + batchCount * inSize) += output;
        }
      }
    }
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::Gradient(const MatType& /* input */,
            const MatType& error,
            MatType& gradient)
{
  typedef typename arma::Cube<typename MatType::elem_type> CubeType;
  arma::cube mappedError(const_cast<MatType&>(error).memptr(),
      outputWidth, outputHeight, outSize * batchSize, false, false);

  gradient.set_size(weights.n_elem, 1);
  gradientTemp = CubeType(gradient.memptr(), weight.n_rows,
      weight.n_cols, weight.n_slices, false, false);
  gradientTemp.zeros();

  for (size_t curGroup = 0; curGroup < numGroups; curGroup++)
  {
    for (size_t outMap = outSize * curGroup * batchSize / numGroups,
        outMapIdx = outSize * curGroup / numGroups, batchCount = 0;
        outMap < outSize * (curGroup + 1) * batchSize / numGroups; outMap++)
    {
      if (outMap != 0 && outMap % outSize == 0)
      {
        batchCount++;
        outMapIdx = 0;
      }

      for (size_t inMap = inSize * curGroup / numGroups;
          inMap < inSize * (curGroup + 1) / numGroups; inMap++, outMapIdx++)
      {
        MatType inputSlice;
        if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
        {
          inputSlice = inputPaddedTemp.slice(inMap + batchCount * inSize);
        }
        else
        {
          inputSlice = inputTemp.slice(inMap + batchCount * inSize);
        }

        MatType deltaSlice = mappedError.slice(outMap);

        MatType output;
        GradientConvolutionRule::Convolution(inputSlice, deltaSlice,
          output, strideWidth, strideHeight);

        if (gradientTemp.n_rows < output.n_rows ||
            gradientTemp.n_cols < output.n_cols)
        {
          gradientTemp.slice(outMapIdx) += output.submat(0, 0,
            gradientTemp.n_rows - 1, gradientTemp.n_cols - 1);
        }
        else if (gradientTemp.n_rows > output.n_rows ||
            gradientTemp.n_cols > output.n_cols)
        {
          gradientTemp.slice(outMapIdx).submat(0, 0, output.n_rows - 1,
            output.n_cols - 1) += output;
        }
        else
        {
          gradientTemp.slice(outMapIdx) += output;
        }
      }
    gradient.submat(weight.n_elem + (outMap % outSize), 0, weight.n_elem +
        (outMap % outSize), 0) = arma::accu(mappedError.slice(outMap));
    }
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
template<typename Archive>
void SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::serialize(Archive& ar, const uint32_t /* version*/)
{
  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(kernelWidth));
  ar(CEREAL_NVP(kernelHeight));
  ar(CEREAL_NVP(strideWidth));
  ar(CEREAL_NVP(strideHeight));
  ar(CEREAL_NVP(padWLeft));
  ar(CEREAL_NVP(padWRight));
  ar(CEREAL_NVP(padHBottom));
  ar(CEREAL_NVP(padHTop));
  ar(CEREAL_NVP(inputWidth));
  ar(CEREAL_NVP(inputHeight));
  ar(CEREAL_NVP(outputWidth));
  ar(CEREAL_NVP(outputHeight));
  ar(CEREAL_NVP(numGroups));
  ar(CEREAL_NVP(padding));

  if (cereal::is_loading<Archive>())
  {
    weights.set_size((outSize * (inSize / numGroups) * kernelWidth *
        kernelHeight) + outSize, 1);
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void SeparableConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::InitializeSamePadding()
{
  /*
   * Using O = (W - F + 2P) / s + 1;
   */
    size_t totalVerticalPadding = (strideWidth - 1) * inputWidth + kernelWidth -
        strideWidth;
    size_t totalHorizontalPadding = (strideHeight - 1) * inputHeight +
        kernelHeight - strideHeight;

    padWLeft = totalVerticalPadding / 2;
    padWRight = totalVerticalPadding - totalVerticalPadding / 2;
    padHTop = totalHorizontalPadding / 2;
    padHBottom = totalHorizontalPadding - totalHorizontalPadding / 2;
}

} // namespace mlpack

#endif

