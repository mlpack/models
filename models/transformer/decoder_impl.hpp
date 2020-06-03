/**
 * @file models/transformer/decoder_impl.hpp
 * @author Mikhail Lozhnikov
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Transformer Decoder class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_TRANSFORMER_DECODER_IMPL_HPP
#define MODELS_TRANSFORMER_DECODER_IMPL_HPP

#include "decoder.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
TransformerDecoder<ActivationFunction, RegularizerType, InputDataType,
OutputDataType>::TransformerDecoder() :
    tgtSeqLen(0),
    srcSeqLen(0),
    dModel(0),
    numHeads(0),
    dimFFN(0),
    dropout(0)
{
  // Nothing to do here.
}

template<typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
TransformerDecoder<ActivationFunction, RegularizerType, InputDataType,
OutputDataType>::TransformerDecoder(
    const size_t numLayers,
    const size_t tgtSeqLen,
    const size_t srcSeqLen,
    const size_t dModel,
    const size_t numHeads,
    const size_t dimFFN,
    const double dropout,
    const InputDataType& attentionMask,
    const InputDataType& keyPaddingMask) :
    numLayers(numLayers),
    tgtSeqLen(tgtSeqLen),
    srcSeqLen(srcSeqLen),
    dModel(dModel),
    numHeads(numHeads),
    dimFFN(dimFFN),
    dropout(dropout),
    attentionMask(attentionMask),
    keyPaddingMask(keyPaddingMask)
{
  decoder = new Sequential<InputDataType, OutputDataType, false>();

  for (size_t N = 0; N < numLayers; ++N)
  {
    AttentionBlock();
    PositionWiseFFNBlock();
  }
}

template<typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
void TransformerDecoder<ActivationFunction, RegularizerType,
InputDataType, OutputDataType>::LoadModel(const std::string& filepath)
{
  data::Load(filepath, "TransformerDecoder", decoder);
  std::cout << "Loaded model" << std::endl;
}

template<typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
void TransformerDecoder<ActivationFunction, RegularizerType,
InputDataType, OutputDataType>::SaveModel(const std::string& filepath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filepath, "TransformerDecoder", decoder);
  std::cout << "Model saved in " << filepath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
