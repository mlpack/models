/**
 * @file methods/ann/layer/separable_convolution.hpp
 * @author Kartik Dutt
 * @author Aakash Kaushik
 * @author Sidharth
 * 
 * For more information, kindly refer to the following paper.
 *
 * @code
 * @article{
 *  author = {Laurent Sifre, Stéphane Mallat},
 *  title = {Rigid-Motion Scattering for Texture Classification},
 *  year = {2014},
 *  url = {https://arxiv.org/pdf/1403.1687}
 * }
 * @endcode
 * 
 * Definition of the Separable Convolution module class.
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SEPARABLE_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_LAYER_SEPARABLE_CONVOLUTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>
#include <mlpack/methods/ann/layer/padding.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

namespace mlpack {

/**
 * Depthwise separable convolution is a neural network operation that 
 * splits standard convolutions into two stages: depthwise convolution
 * and pointwise convolution. In the depthwise stage, individual channels
 * of the input are convolved separately, reducing computation.
 * 
 * The pointwise stage combines these channel-wise results through
 * a 1x1 convolution, preserving inter-channel relationships. 
 * 
 * This approach significantly reduces model parameters and computations,
 * making it computationally efficient for mobile and edge devices.
 * 
 * The expected input for a depthwise separable convolution layer is a
 * 3D tensor with dimensions (height, width, channels).
 * 
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 * @tparam MatType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = NaiveConvolution<FullConvolution>,
    typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename MatType = arma::mat
>
class SeparableConvolutionType: public Layer<MatType>
{
 public:
  //! Create the Separable Convolution object.
  SeparableConvolutionType();

  /**
   * Create the Separable Convolution object using the specified number of input maps,
   * output maps, filter size, stride, padding parameter and number of groups.
   * 
   * @param inSize The number of input maps.
   * @param outSize The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param inputWidth The width of the input data.
   * @param inputHeight The height of the input data.
   * @param numGroups The number of groups in which input maps will be divided.
   *                  numGroups = inSize implies depthwise convolution.
   *                  Defaults to 1.
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */
  SeparableConvolutionType(const size_t inSize,
                           const size_t outSize,
                           const size_t kernelWidth,
                           const size_t kernelHeight,
                           const size_t strideWidth = 1,
                           const size_t strideHeight = 1,
                           const size_t padW = 0,
                           const size_t padH = 0,
                           const size_t inputWidth = 0,
                           const size_t inputHeight = 0,
                           const size_t numGroups = 1,
                           const std::string& paddingType = "None");

  /**
   * Create the Separable Convolution object using the specified number of input maps,
   * output maps, filter size, stride, padding parameter and number of groups.
   * 
   * @param inSize The number of input maps.
   * @param outSize The number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW A two-value tuple indicating padding widths of the input.
   *             First value is padding at left side. Second value is padding on
   *             right side.
   * @param padH A two-value tuple indicating padding heights of the input.
   *             First value is padding at top. Second value is padding on
   *             bottom.
   * @param inputWidth The width of the input data.
   * @param inputHeight The height of the input data.
   * @param numGroups The number of groups in which input maps will be divided.
   *                  numGroups = inSize implies depthwise convolution.
   *                  Defaults to 1.
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */
  SeparableConvolutionType(const size_t inSize,
                           const size_t outSize,
                           const size_t kernelWidth,
                           const size_t kernelHeight,
                           const size_t strideWidth,
                           const size_t strideHeight,
                           const std::tuple<size_t, size_t> padW,
                           const std::tuple<size_t, size_t> padH,
                           const size_t inputWidth = 0,
                           const size_t inputHeight = 0,
                           const size_t numGroups = 1,
                           const std::string& paddingType = "None");

  /*
   * Set the weight and bias term.
   */
  void Reset();
  
  //! Clone the SeparableConvolutionType object. This handles polymorphism
  //! correctly.
  SeparableConvolutionType* Clone() const override
  {
    return new SeparableConvolutionType(*this);
  }
  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& gy,
                MatType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& /* input */,
                const MatType& error,
                MatType& gradient);

  //! Get the input parameter.
  MatType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  MatType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  MatType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  MatType& OutputParameter() { return outputParameter; }

  //! Get the parameters.
  MatType const& Parameters() const { return weights; }
  //! Modify the parameters.
  MatType& Parameters() { return weights; }

  //! Get the delta.
  MatType const& Delta() const { return delta; }
  //! Modify the delta.
  MatType& Delta() { return delta; }

  //! Get the gradient.
  MatType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  MatType& Gradient() { return gradient; }

  //! Get the bias of the layer.
  arma::mat const& Bias() const { return bias; }
  //! Modify the bias of the layer.
  arma::mat& Bias() { return bias; }

  //! Get the input width.
  size_t InputWidth() const { return inputWidth; }
  //! Modify input the width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the input height.
  size_t InputHeight() const { return inputHeight; }
  //! Modify the input height.
  size_t& InputHeight() { return inputHeight; }

  //! Get the output width.
  size_t OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  size_t OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the number of input maps.
  size_t InputSize() const { return inSize; }

  //! Get the number of output maps.
  size_t OutputSize() const { return outSize; }

  //! Get the kernel width.
  size_t KernelWidth() const { return kernelWidth; }
  //! Modify the kernel width.
  size_t& KernelWidth() { return kernelWidth; }

  //! Get the kernel height.
  size_t KernelHeight() const { return kernelHeight; }
  //! Modify the kernel height.
  size_t& KernelHeight() { return kernelHeight; }

  //! Get the stride width.
  size_t StrideWidth() const { return strideWidth; }
  //! Modify the stride width.
  size_t& StrideWidth() { return strideWidth; }

  //! Get the stride height.
  size_t StrideHeight() const { return strideHeight; }
  //! Modify the stride height.
  size_t& StrideHeight() { return strideHeight; }

  //! Get number of Groups for Grouped Convolution.
  size_t NumGroups() const { return numGroups; }
  //! Modify the number of Groups.
  size_t& NumGroups() { return numGroups; }

  //! Get the top padding height.
  size_t PadHTop() const { return padHTop; }
  //! Modify the top padding height.
  size_t& PadHTop() { return padHTop; }

  //! Get the bottom padding height.
  size_t PadHBottom() const { return padHBottom; }
  //! Modify the bottom padding height.
  size_t& PadHBottom() { return padHBottom; }

  //! Get the left padding width.
  size_t PadWLeft() const { return padWLeft; }
  //! Modify the left padding width.
  size_t& PadWLeft() { return padWLeft; }

  //! Get the right padding width.
  size_t PadWRight() const { return padWRight; }
  //! Modify the right padding width.
  size_t& PadWRight() { return padWRight; }

  /**
   * Serialize the layer.
   */
  template <typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
  * Return the convolution output size.
  *
  * @param size The size of the input (row or column).
  * @param k The size of the filter (width or height).
  * @param s The stride size (x or y direction).
  * @param pSideOne The size of the padding (width or height) on one side.
  * @param pSideTwo The size of the padding (width or height) on another side.
  * @return The convolution output size.
  */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t pSideOne,
                     const size_t pSideTwo)
  {
    return std::floor(size + pSideOne + pSideTwo - k) / s + 1;
  }

  /*
   * Function to assign padding such that output size is same as input size.
   */
  void InitializeSamePadding();

  /**
   * Rotates a 3rd-order tensor counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  template <typename eT>
  void Rotate180(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    output = arma::Cube<eT>(input.n_rows, input.n_cols, input.n_slices);

    // * left-right flip, up-down flip */
    for (size_t s = 0; s < output.n_slices; s++)
      output.slice(s) = arma::fliplr(arma::flipud(input.slice(s)));
  }

  /**
   * Rotates a dense matrix counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  
  void Rotate180(const MatType& input, MatType& output)
  {
    /* left-right flip, up-down flip */
    output = arma::fliplr(arma::flipud(input));
  }

  //! Locally-stored number of input channels.
  size_t inSize;

  //! Locally-stored number of output channels.
  size_t outSize;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Locally-stored filter/kernel width.
  size_t kernelWidth;

  //! Locally-stored filter/kernel height.
  size_t kernelHeight;

  //! Locally-stored stride of the filter in x-direction.
  size_t strideWidth;

  //! Locally-stored stride of the filter in y-direction.
  size_t strideHeight;

  //! Locally-stored left-side padding width.
  size_t padWLeft;

  //! Locally-stored right-side padding width.
  size_t padWRight;

  //! Locally-stored bottom padding height.
  size_t padHBottom;

  //! Locally-stored top padding height.
  size_t padHTop;

  //! Locally-stored weights object.
  MatType weights;

  //! Locally-stored weight object.
  arma::cube weight;

  //! Locally-stored bias object.
  arma::mat bias;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally stored number of Groups parameter.
  size_t numGroups;

  //! Locally-stored transformed output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed padded input parameter.
  arma::cube inputPaddedTemp;

  //! Locally-stored transformed error parameter.
  arma::cube gTemp;

  //! Locally-stored transformed gradient parameter.
  arma::cube gradientTemp;

  //! Locally-stored padding layer.
  Padding padding;

  //! Locally-stored delta object.
  MatType delta;

  //! Locally-stored gradient object.
  MatType gradient;

  //! Locally-stored input parameter object.
  MatType inputParameter;

  //! Locally-stored output parameter object.
  MatType outputParameter;

}; // Separable Convolution Class.

typedef SeparableConvolutionType<
    NaiveConvolution<ValidConvolution>,
    NaiveConvolution<FullConvolution>,
    NaiveConvolution<ValidConvolution>,
    arma::mat> 
    SeparableConvolution;
} // namespace mlpack

// Include implementation.
#include "separable_convolution_impl.hpp"

#endif

