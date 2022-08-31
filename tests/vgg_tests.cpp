/**
 * @file tests/vgg_tests.cpp
 * @author Shubham Agrawal
 *
 * Tests the SqueezeNet model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "../models/vgg/vgg.hpp"

#include "test_catch_tools.hpp"
#include "catch.hpp"
#include "serialization.hpp"

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

TEST_CASE("VGG11SerializationTest", "[VGGTests]")
{
	models::VGG11 model;
	ModelSerializationTest(model);
}

TEST_CASE("VGG13SerializationTest", "[VGGTests]")
{
	models::VGG13 model;
	ModelSerializationTest(model);
}

TEST_CASE("VGG16SerializationTest", "[VGGTests]")
{
	models::VGG16 model;
	ModelSerializationTest(model);
}

TEST_CASE("VGG19SerializationTest", "[VGGTests]")
{
	models::VGG19 model;
	ModelSerializationTest(model);
}

TEST_CASE("VGG11BNSerializationTest", "[VGGTests]")
{
	models::VGG11BN model;
	ModelSerializationTest(model);
}

TEST_CASE("VGG13BNSerializationTest", "[VGGTests]")
{
	models::VGG13BN model;
	ModelSerializationTest(model);
}

TEST_CASE("VGG16BNSerializationTest", "[VGGTests]")
{
	models::VGG16BN model;
	ModelSerializationTest(model);
}

TEST_CASE("VGG19BNSerializationTest", "[VGGTests]")
{
	models::VGG19BN model;
	ModelSerializationTest(model);
}

TEST_CASE("VGG11Test", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
	arma::mat output;
	models::VGG11 vggLayer11;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG11>(vggLayer11);
	ModelDimTest(model, input);
}

TEST_CASE("VGG13Test", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
	arma::mat output;
	models::VGG13 vggLayer13;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG13>(vggLayer13);
	ModelDimTest(model, input);
}

TEST_CASE("VGG16Test", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
	arma::mat output;
	models::VGG16 vggLayer16;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG16>(vggLayer16);
	ModelDimTest(model, input);
}

TEST_CASE("VGG19Test", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
	arma::mat output;
	models::VGG19 vggLayer19;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG19>(vggLayer19);
	ModelDimTest(model, input);
}

TEST_CASE("VGG11BNTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
	arma::mat output;
	models::VGG11BN vggbnLayer11;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG11BN>(vggbnLayer11);
	ModelDimTest(model, input);
}

TEST_CASE("VGG13BNTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
	arma::mat output;
	models::VGG13BN vggbnLayer13;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG13BN>(vggbnLayer13);
	ModelDimTest(model, input);
}

TEST_CASE("VGG16BNTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
	arma::mat output;
	models::VGG16BN vggbnLayer16;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG16BN>(vggbnLayer16);
	ModelDimTest(model, input);
}

TEST_CASE("VGG19BNTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
	arma::mat output;
	models::VGG19BN vggbnLayer19;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG19BN>(vggbnLayer19);
	ModelDimTest(model, input);
}

TEST_CASE("VGG11MultiBatchTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG11 vggLayer11;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG11>(vggLayer11);
	ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("VGG13MultiBatchTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG13 vggLayer13;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG13>(vggLayer13);
	ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("VGG16MultiBatchTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG16 vggLayer16;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG16>(vggLayer16);
	ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("VGG19MultiBatchTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG19 vggLayer19;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG19>(vggLayer19);
	ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("VGG11BNMultiBatchTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG11BN vggbnLayer11;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG11BN>(vggbnLayer11);
	ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("VGG13BNMultiBatchTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG13BN vggbnLayer13;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG13BN>(vggbnLayer13);
	ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("VGG16BNMultiBatchTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG16BN vggbnLayer16;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG16BN>(vggbnLayer16);
	ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("VGG19BNMultiBatchTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG19BN vggbnLayer19;
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG19BN>(vggbnLayer19);
	ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("VGG11CustomTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG11 vggLayer11(512);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG11>(vggLayer11);
	ModelDimTest(model, input, 512, 10);
}

TEST_CASE("VGG13CustomTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG13 vggLayer13(512);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG13>(vggLayer13);
	ModelDimTest(model, input, 512, 10);
}

TEST_CASE("VGG16CustomTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG16 vggLayer16(512);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG16>(vggLayer16);
	ModelDimTest(model, input, 512, 10);
}

TEST_CASE("VGG19CustomTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG19 vggLayer19(512);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG19>(vggLayer19);
	ModelDimTest(model, input, 512, 10);
}

TEST_CASE("VGG11BNCustomTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG11BN vggbnLayer11(512);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG11BN>(vggbnLayer11);
	ModelDimTest(model, input, 512, 10);
}

TEST_CASE("VGG13BNCustomTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG13BN vggbnLayer13(512);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG13BN>(vggbnLayer13);
	ModelDimTest(model, input, 512, 10);
}

TEST_CASE("VGG16BNCustomTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG16BN vggbnLayer16(512);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG16BN>(vggbnLayer16);
	ModelDimTest(model, input, 512, 10);
}

TEST_CASE("VGG19BNCustomTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG19BN vggbnLayer19(512);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG19BN>(vggbnLayer19);
	ModelDimTest(model, input, 512, 10);
}

TEST_CASE("VGG11NoTopTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG11 vggLayer11(512, false);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG11>(vggLayer11);
	ModelDimTest(model, input, 25088, 10);
}

TEST_CASE("VGG13NoTopTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG13 vggLayer13(512, false);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG13>(vggLayer13);
	ModelDimTest(model, input, 25088, 10);
}

TEST_CASE("VGG16NoTopTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG16 vggLayer16(512, false);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG16>(vggLayer16);
	ModelDimTest(model, input, 25088, 10);
}

TEST_CASE("VGG19NoTopTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG19 vggLayer19(512, false);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG19>(vggLayer19);
	ModelDimTest(model, input, 25088, 10);
}

TEST_CASE("VGG11BNNoTopTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG11BN vggbnLayer11(512, false);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG11BN>(vggbnLayer11);
	ModelDimTest(model, input, 25088, 10);
}

TEST_CASE("VGG13BNNoTopTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG13BN vggbnLayer13(512, false);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG13BN>(vggbnLayer13);
	ModelDimTest(model, input, 25088, 10);
}

TEST_CASE("VGG16BNNoTopTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG16BN vggbnLayer16(512, false);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG16BN>(vggbnLayer16);
	ModelDimTest(model, input, 25088, 10);
}

TEST_CASE("VGG19BNNoTopTest", "[VGGTests]")
{
	arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
	arma::mat output;
	models::VGG19BN vggbnLayer19(512, false);
	ann::FFN<> model;
	model.InputDimensions() = std::vector<size_t>({224, 224, 3});
	model.Add<models::VGG19BN>(vggbnLayer19);
	ModelDimTest(model, input, 25088, 10);
}
