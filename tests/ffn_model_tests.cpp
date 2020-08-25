/**
 * @file model_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities and performance of models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BOOST_TEST_DYN_LINK
#include <utils/utils.hpp>
#include <ensmallen.hpp>
#include <dataloader/dataloader.hpp>
#include <models/darknet/darknet.hpp>
#include <models/transformer/encoder.hpp>
#include <models/transformer/decoder.hpp>
#include <models/transformer/transformer.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <boost/test/unit_test.hpp>

// Use namespaces for convenience.
using namespace mlpack;
using namespace mlpack::ann;
using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(FFNModelsTests);

/**
 * Simple test for Darknet model.
 */
BOOST_AUTO_TEST_CASE(DarknetModelTest)
{
  mlpack::ann::DarkNet<> darknetModel(3, 224, 224, 1000);
  arma::mat input(224 * 224 * 3, 1), output;
  input.ones();

  // Check output shape.
  darknetModel.GetModel().Predict(input, output);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  BOOST_REQUIRE_EQUAL(output.n_rows, 1000);

  // Repeat for DarkNet-53.
  mlpack::ann::DarkNet<> darknet53(3, 224, 224, 1000);
  darknet53.GetModel().Predict(input, output);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  BOOST_REQUIRE_EQUAL(output.n_rows, 1000);
}

/**
 * Simple Transformer Encoder test.
 */
BOOST_AUTO_TEST_CASE(TransformerEncoderTest)
{
  const size_t vocabSize = 20;
  const size_t numLayers = 2;
  const size_t srcSeqLen = 10;
  const size_t dModel = 16;
  const size_t numHeads = 2;
  const size_t dimFFN = 16;
  const double dropout = 0.3;

  arma::mat input = arma::randu(dModel * srcSeqLen, 1);
  arma::mat output;

  mlpack::ann::TransformerEncoder<> encoder(numLayers, srcSeqLen,
      dModel, numHeads, dimFFN, dropout);

  FFN<NegativeLogLikelihood<>, XavierInitialization> model;

  model.Add(encoder.Model());
  model.Add<Linear<>>(dModel * srcSeqLen, vocabSize);
  model.Add<LogSoftMax<>>();

  model.Predict(input, output);

  BOOST_REQUIRE_EQUAL(output.n_rows, vocabSize);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
}

/**
 * Simple Transformer Decoder test.
 */
BOOST_AUTO_TEST_CASE(TransformerDecoderTest)
{
  const size_t vocabSize = 20;
  const size_t numLayers = 2;
  const size_t tgtSeqLen = 10;
  const size_t srcSeqLen = 10;
  const size_t dModel = 16;
  const size_t numHeads = 2;
  const size_t dimFFN = 16;
  const double dropout = 0.3;

  arma::mat query = arma::randu(dModel * tgtSeqLen, 1);
  arma::mat memory = 0.73 * arma::randu(dModel * srcSeqLen, 1);

  arma::mat input = arma::join_cols(query, memory);
  arma::mat output;

  mlpack::ann::TransformerDecoder<> decoder(numLayers, tgtSeqLen, srcSeqLen,
      dModel, numHeads, dimFFN, dropout);

  FFN<NegativeLogLikelihood<>, XavierInitialization> model;

  model.Add(decoder.Model());
  model.Add<Linear<>>(dModel * tgtSeqLen, vocabSize);
  model.Add<LogSoftMax<>>();

  model.Predict(input, output);

  BOOST_REQUIRE_EQUAL(output.n_rows, vocabSize);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
}

/**
 * Transformer Model test.
 */
BOOST_AUTO_TEST_CASE(TransformerTest)
{
  const size_t tgtVocabSize = 20;
  const size_t srcVocabSize = 20;
  const size_t numLayers = 2;
  const size_t tgtSeqLen = 10;
  const size_t srcSeqLen = 10;
  const size_t dModel = 16;
  const size_t numHeads = 2;
  const size_t dimFFN = 16;
  const double dropout = 0.3;

  arma::mat srcLanguage(srcSeqLen, 1), tgtLanguage(tgtSeqLen, 1);

  for (size_t t = 0; t < srcSeqLen; ++t)
  {
    srcLanguage(t) = mlpack::math::RandInt(1, srcVocabSize);
  }

  for (size_t t = 0; t < tgtSeqLen; ++t)
  {
    tgtLanguage(t) = mlpack::math::RandInt(1, tgtVocabSize);
  }

  arma::mat input = arma::join_cols(srcLanguage, tgtLanguage);
  arma::mat output;

  mlpack::ann::Transformer<> transformer(numLayers, tgtSeqLen, srcSeqLen,
      tgtVocabSize, srcVocabSize, dModel, numHeads, dimFFN, dropout);

  FFN<NegativeLogLikelihood<>, XavierInitialization> model;

  model.Add(transformer.Model());
  model.Add<Linear<>>(dModel * tgtSeqLen, tgtVocabSize);
  model.Add<LogSoftMax<>>();

  model.Predict(input, output);

  BOOST_REQUIRE_EQUAL(output.n_rows, tgtVocabSize);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
}

BOOST_AUTO_TEST_SUITE_END();
