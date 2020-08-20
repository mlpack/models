/**
 * @file skip_layer_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of LeNet using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_YOLOV3_LAYER_IMPL_HPP
#define MODELS_YOLOV3_LAYER_IMPL_HPP

#include "yolov3_layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
YOLOv3<InputDataType, OutputDataType>::YOLOv3(
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t inSize)
{
  // Not sure if we need to do anything here.
  // Let's write the forward first.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void YOLOv3<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
    std::vector<arma::Mat<eT>> outputVector;
    std::vector<std::tuple<size_t, size_t, size_t>> outputInfo;
    for (size_t i = 0; i < network.size(); ++i)
    {
      boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
          boost::apply_visitor(outputParameterVisitor, network[i]))),
          network[i]);
      size_t moduleOutputWidth = boost::apply_visitor(outputWidthVisitor,
          network[i]);
      size_t moduleOutputHeight = boost::apply_visitor(outputHeightVisitor,
          network[i]);
      size_t moduleOutputChannels = boost::apply_visitor(outputParameterVisitor,
          network[i]).n_elem / (moduleOutputWidth * moduleOutputHeight);
      outputVector.push_back(arma::Mat<eT>(input.n_cols,
          moduleOutputWidth * moduleOutputHeight * moduleOutputChannels));
      outputVector[i] = arma::mat(boost::apply_visitor(outputParameterVisitor,
          network[i]).memptr(), input.n_cols,
          moduleOutputWidth * moduleOutputHeight * moduleOutSize);
      outputInfo.push_back({moduleOutputWidth, moduleOutputHeight, moduleOutputChannels});
    }

    // Upsample and concat. The feature performed in YOLOv3.
    for (size_t i = network.size() - 2; i >= 0; i--)
    {
      arma::Mat<eT> upsampledOutput;
      BilinearInterpolation<InputDataType, OutputDataType> upscale({
          std::get<0>(outputInfo[i]), std::get<1>(outputInfo[i]),
          std::get<0>(outputInfo[i + 1]), std::get<0>(outputInfo[i + 1]
          std::get<2>(outputInfo[i]))});
      upscale.Forward(outputVector[i + 1], upsampledOutput);
      outputVector[i] += umsampledOutput;
    }
}
} // namespace ann
} // namespace mlpack

#endif
