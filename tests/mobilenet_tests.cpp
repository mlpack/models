#include "../models/mobilenet/mobilenet.hpp"

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
 * @param n_cols Output columns to check against.
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

  FFN<> model;
  model.Add<LayerType>(layer);

  model.InputDimensions() = std::vector<size_t>({224, 224, 3});

  // Takes only one pass over the input data.
  ens::StandardSGD opt(0.1, 1, 5, -100, false);
  model.Train(input, output, opt);

  arma::mat originalOutput;
  model.Predict(input, originalOutput);

  // Now serialize the model.
  FFN<> xmlModel, jsonModel, binaryModel;
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

TEST_CASE("MobileNetSerializationTest", "[MobileNetTests]")
{
  models::Mobilenet model;
  ModelSerializationTest(model);
}

TEST_CASE("MobileNetTest", "[MobileNetTests]")
{
  arma::mat input(224 * 224 * 3, 1, arma::fill::randu);
  arma::mat output;
  models::Mobilenet mobilenetLayer;
  FFN<> model;
  model.InputDimensions() = std::vector<size_t>({224, 224, 3});
  model.Add<models::Mobilenet>(mobilenetLayer);
  ModelDimTest(model, input);
}

TEST_CASE("MobileNetMultiBatchTest", "[MobileNetTests]")
{
  arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
  arma::mat output;
  models::Mobilenet mobilenetLayer;
  FFN<> model;
  model.InputDimensions() = std::vector<size_t>({224, 224, 3});
  model.Add<models::Mobilenet>(mobilenetLayer);
  ModelDimTest(model, input, 1000, 10);
}

TEST_CASE("MobileNetCustomTest", "[MobileNetTests]")
{
  arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
  arma::mat output;
  models::Mobilenet mobilenetLayer(512);
  FFN<> model;
  model.InputDimensions() = std::vector<size_t>({224, 224, 3});
  model.Add<models::Mobilenet>(mobilenetLayer);
  ModelDimTest(model, input, 512, 10);
}

TEST_CASE("MobileNetNoTopTest", "[MobileNetTests]")
{
  arma::mat input(224 * 224 * 3, 10, arma::fill::randu);
  arma::mat output;
  models::Mobilenet mobilenetLayer(512, false);
  FFN<> model;
  model.InputDimensions() = std::vector<size_t>({224, 224, 3});
  model.Add<models::Mobilenet>(mobilenetLayer);
  ModelDimTest(model, input, 9216, 10);
}
