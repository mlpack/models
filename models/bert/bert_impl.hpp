/**
 * @file models/bert/bert_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the BERT (Bidirectional Encoder Representation for
 * Transformers).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_BERT_BERT_IMPL_HPP
#define MODELS_BERT_BERT_IMPL_HPP

#include "bert.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename OutputLayerType, typename InitType, typename InputDataType,
    typename OutputDataType>
BERT<OutputLayerType, InitType, InputDataType, OutputDataType>::BERT() :
    srcVocabSize(0),
    srcSeqLen(0),
    numEncoderLayers(0),
    dModel(0),
    numHeads(0),
    dimFFN(4 * dModel),
    dropout(0.0)
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitType, typename InputDataType,
    typename OutputDataType>
BERT<OutputLayerType, InitType, InputDataType, OutputDataType>::BERT(
    const size_t srcVocabSize,
    const size_t srcSeqLen,
    const size_t numEncoderLayers,
    const size_t dModel,
    const size_t numHeads,
    const double dropout,
    const InputDataType& attentionMask,
    const InputDataType& keyPaddingMask) :
    srcVocabSize(srcVocabSize),
    srcSeqLen(srcSeqLen),
    numEncoderLayers(numEncoderLayers),
    dModel(dModel),
    numHeads(numHeads),
    dimFFN(4 * dModel),
    dropout(dropout),
    attentionMask(attentionMask),
    keyPaddingMask(keyPaddingMask)
{
  AddMerge<>* embedding = new AddMerge<>();
  embedding->Add<Lookup<>>(vocabSize, dModel);
  embedding->Add<Lookup<>>(3, dModel);

  bert.Add(embedding);
  bert.Add<PositionalEncoding<>>(dModel, srcSeqLen);
  bert.Add<Dropout<>>(dropout);

  for (size_t i = 0; i < numLayers; ++i)
  {
    mlpack::ann::TransformerEncoder<> encoder(
      numEncoderLayers,
      srcSeqLen,
      dModel,
      numHeads,
      dimFFN,
      dropout,
      attentionMask,
      keyPaddingMask
    );

    bert.Add(encoder.Model());
  }
}

template<typename OutputLayerType, typename InitType, typename InputDataType,
    typename OutputDataType>
void BERT<OutputLayerType, InitType, InputDataType, OutputDataType>::LoadModel(
    const std::string& filepath)
{
  data::Load(filepath, "BERT", bert);
  std::cout << "Loaded model" << std::endl;
}

template<typename OutputLayerType, typename InitType, typename InputDataType,
    typename OutputDataType>
void BERT<OutputLayerType, InitType, InputDataType, OutputDataType>::SaveModel(
    const std::string& filepath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filepath, "BERT", bert);
  std::cout << "Model saved in " << filepath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
