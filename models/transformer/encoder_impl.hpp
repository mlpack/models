/**
 * @file models/transformer/encoder_impl.hpp
 * @author Mikhail Lozhnikov
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Transformer Encoder class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_TRANSFORMER_ENCODER_IMPL_HPP
#define MODELS_TRANSFORMER_ENCODER_IMPL_HPP

#include "encoder.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
TransformerEncoder<ActivationFunction, RegularizerType, InputDataType,
OutputDataType>::TransformerEncoder(
    const size_t numLayers,
    const size_t srcSeqLen,
    const size_t dModel,
    const size_t numHeads,
    const size_t dimFFN,
    const double dropout,
    const InputDataType& attentionMask,
    const InputDataType& keyPaddingMask) :
    numLayers(numLayers),
    srcSeqLen(srcSeqLen),
    dModel(dModel),
    numHeads(numHeads),
    dimFFN(dimFFN),
    dropout(dropout),
    attentionMask(attentionMask),
    keyPaddingMask(keyPaddingMask)
{
  encoder = new Sequential<InputDataType, OutputDataType, false>();

  for (size_t n = 0; n < numLayers; ++n)
  {
    AttentionBlock();
    PositionWiseFFNBlock();
  }
}

template <typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
void TransformerEncoder<ActivationFunction, RegularizerType, InputDataType,
OutputDataType>::LoadModel(const std::string& filePath)
{
  data::Load(filePath, "TransformerEncoder", encoder);
  std::cout << "Loaded model" << std::endl;
}

template <typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
void TransformerEncoder<ActivationFunction, RegularizerType, InputDataType,
OutputDataType>::SaveModel(const std::string& filePath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filePath, "TransformerEncoder", encoder);
  std::cout << "Model saved in " << filePath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
