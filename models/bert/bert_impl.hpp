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
    vocabSize(0),
    dModel(0),
    numHeads(0),
    dimFFN(4 * dModel),
    numLayers(0),
    dropout(0),
    maxSequenceLength(5000),
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitType, typename InputDataType,
    typename OutputDataType>
BERT<OutputLayerType, InitType, InputDataType, OutputDataType>::BERT(
    const size_t vocabSize,
    const size_t dModel,
    const size_t numHeads,
    const size_t numLayers,
    const double dropout,
    const size_t maxSequenceLength,
    const InputDataType& attentionMask,
    const InputDataType& keyPaddingMask) :
    vocabSize(vocabSize)
    dModel(dModel),
    numHeads(numHeads),
    dimFFN(4 * dModel),
    numLayers(numLayers),
    dropout(dropout),
    maxSequenceLength(maxSequenceLength),
    attentionMask(attentionMask),
    keyPaddingMask(keyPaddingMask)
{
  embedding = new AddMerge<>();
  embedding.Add<Lookup<>>(vocabSize, dModel);
  embedding.Add<Lookup<>>(3, dModel);

  bert.Add(embedding);
  bert.Add<PositionalEncoding<>>(dModel, maxSequenceLength);
  bert.Add<Dropout<>>(dropout);

  for (size_t i = 0; i < numLayers; ++i)
  {
    TransformerEncoder<> enc(dModel, numHeads, dimFFN, dropout);
    enc.AttentionMask() = attentionMask;
    enc.KeyPaddingMask() = keyPaddingMask;
    bert.Add(enc);
  }
}

template<typename OutputLayerType, typename InitType, typename InputDataType,
    typename OutputDataType>
void BERT<OutputLayerType, InitType, InputDataType, OutputDataType>::LoadModel(
    const std::string& filepath)
{
  data::Load(filepath, "BERT", network);
  std::cout << "Loaded model" << std::endl;
}

template<typename OutputLayerType, typename InitType, typename InputDataType,
    typename OutputDataType>
void BERT<OutputLayerType, InitType, InputDataType, OutputDataType>::SaveModel(
    const std::string& filepath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filepath, "BERT", network);
  std::cout << "Model saved in " << filepath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
