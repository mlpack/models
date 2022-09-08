/**
 * @file tests/alexnet_tests.cpp
 * @author Shubham Agrawal
 *
 * Tests the AlexNet model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "../models/alexnet/alexnet.hpp"

#include "./test_catch_tools.hpp"
#include "./catch.hpp"
#include "./serialization.hpp"

using namespace mlpack;

/**
 * Checks for the output dimensions of the model.
 *
 * @tparam ModelType Type of model to check.
 *
 * @param model The model to test.
 * @param input Input to pass to the model.
 * @param n_rows Output rows to check against.
 * @param n_cols Output columns to check against..
 */
template <typename ModelType>
void ModelDimTest(ModelType& model,
               arma::mat& input,
               const size_t n_rows = 1000,
               const size_t n_cols = 1)
{
  arma::mat output;
  model.Predict(input, output);
  REQUIRE(output.n_rows == n_rows);
  REQUIRE(output.n_cols == n_cols);
}

/**
 * Checks for the output sum of the model for a 
 *     single and multiple batch input.
 *
 * @tparam ModelType Type of model to check.
 *
 * @param model The model to test.
 * @param input Input to pass to the model.
 * @param singleBatchOutput Sum of the output of a single batch.
 * @param multipleBatchOutput Sum of the output of Multiple batches.
 * @param numBatches Number of batches to create for input.
 */
template <typename ModelType>
void PreTrainedModelTest(ModelType& model,
                         arma::mat& input,
                         const double singleBatchOutput,
                         const double multipleBatchOutput,
                         const size_t numBatches = 4)
{
  arma::mat multipleBatchInput(input.n_rows, numBatches), output;
  input.ones();
  multipleBatchInput.ones();

  // Run prediction for single batch.
  model.Predict(input, output);
  REQUIRE(arma::accu(output) == Approx(singleBatchOutput).epsilon(1e-2));

  // Run prediction for multiple batch.
  model.Predict(multipleBatchInput, output);
  REQUIRE(arma::accu(output) == Approx(multipleBatchOutput).epsilon(1e-2));
}

// General ANN serialization test.
template<typename LayerType>
void ModelSerializationTest(LayerType& layer)
{
  arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
  arma::mat output(1000, 10, arma::fill::randu);

  ann::FFN<> model;
  model.Add<LayerType>(layer);

	model.InputDimensions() = std::vector<size_t>({224, 224, 3});

  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  model.Train(input, output, opt);

  arma::mat originalOutput;
  model.Predict(input, originalOutput);

  // Now serialize the model.
  ann::FFN<> xmlModel, jsonModel, binaryModel;
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  // Ensure that predictions are the same.
  arma::mat modelOutput, xmlOutput, jsonOutput, binaryOutput;
  model.Predict(input, modelOutput);
  xmlModel.Predict(input, xmlOutput);
  jsonModel.Predict(input, jsonOutput);
  binaryModel.Predict(input, binaryOutput);

  CheckMatrices(originalOutput, modelOutput, 1e-5);
  CheckMatrices(originalOutput, xmlOutput, 1e-5);
  CheckMatrices(originalOutput, jsonOutput, 1e-5);
  CheckMatrices(originalOutput, binaryOutput, 1e-5);
}

TEST_CASE("AlexNetSerializationTest", "[AlexnetTests]")
{
  models::AlexNet model;
  ModelSerializationTest(model);
}

TEST_CASE("AlexNetTest", "[AlexnetTests]")
{
  arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
  arma::mat output;
  models::AlexNet alexLayer;
  ann::FFN<> model;
  model.InputDimensions() = std::vector<size_t>({224, 224, 3});
  model.Add<models::AlexNet>(alexLayer);
  ModelDimTest(model, input);
}

TEST_CASE("AlexNetMultiBatchTest", "[AlexnetTests]")
{
  arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
  arma::mat output;
  models::AlexNet alexLayer;
  ann::FFN<> model;
  model.InputDimensions() = std::vector<size_t>({224, 224, 3});
  model.Add<models::AlexNet>(alexLayer);
  ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("AlexNetCustomTest", "[AlexnetTests]")
{
  arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
  arma::mat output;
  models::AlexNet alexLayer(512);
  ann::FFN<> model;
  model.InputDimensions() = std::vector<size_t>({224, 224, 3});
  model.Add<models::AlexNet>(alexLayer);
  ModelDimTest(model, input, 512, 10);
}

TEST_CASE("AlexNetNoTopTest", "[AlexnetTests]")
{
  arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
  arma::mat output;
  models::AlexNet alexLayer(512, false);
  ann::FFN<> model;
  model.InputDimensions() = std::vector<size_t>({224, 224, 3});
  model.Add<models::AlexNet>(alexLayer);
  ModelDimTest(model, input, 9216, 10);
}
