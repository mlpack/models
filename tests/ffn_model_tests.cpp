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
#include <utils/utils.hpp>
#include <ensmallen.hpp>
#include <dataloader/dataloader.hpp>
#include <models/darknet/darknet.hpp>
#include <models/yolo/yolo.hpp>
#include <models/resnet/resnet.hpp>
#include "catch.hpp"

using namespace mlpack::models;

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
                         const size_t singleBatchOutput,
                         const size_t multipleBatchOutput,
                         const size_t numBatches = 4)
{
  arma::mat multipleBatchInput(input.n_rows, numBatches), output;
  input.ones();
  multipleBatchInput.ones();

  // Run prediction for single batch.
  model.Predict(input, output);
  REQUIRE(arma::accu(output) == singleBatchOutput);

  // Run prediction for multiple batch.
  model.Predict(multipleBatchInput, output);
  REQUIRE(arma::accu(output) == multipleBatchOutput);
}

/**
 * Simple test for Darknet model.
 */
TEST_CASE("DarknetModelTest", "[FFNModelsTests]")
{
  arma::mat input(224 * 224 * 3, 1);
  DarkNet19 darknet19(3, 224, 224, 1000);

  // Check output shape.
  ModelDimTest(darknet19.GetModel(), input);

  // Repeat for DarkNet-53.
  DarkNet53 darknet53(3, 224, 224, 1000);
  ModelDimTest(darknet53.GetModel(), input);
}

/**
 * Simple test for YOLOv1 model.
 */
TEST_CASE("YOLOV1ModelTest", "[FFNModelsTests]")
{
  arma::mat input(448 * 448 * 3, 1);
  YOLO<> yolo(3, 448, 448);

  // Check output shape.
  ModelDimTest(yolo.GetModel(), input, (7 * 7 * (5 * 2 + 20)), 1);
}

/**
 * Simple test for ResNet models.
 */
TEST_CASE("ResNetModelTest", "[FFNModelsTests]")
{
  ResNet18 resnet18(3, 224, 224);
  arma::mat input(224 * 224 * 3, 1);

  // Check output shape for resnet18.
  ModelDimTest(resnet18.GetModel(), input);

  // Check output shape for resnet34.
  ResNet34 resnet34(3, 224, 224);
  ModelDimTest(resnet34.GetModel(), input);

  // Check output shape for resnet50.
  ResNet50 resnet50(3, 224, 224);
  ModelDimTest(resnet50.GetModel(), input);

  // Check output shape for resnet101.
  ResNet101 resnet101(3, 224, 224);
  ModelDimTest(resnet101.GetModel(), input);

  // Check output shape for resnet152.
  ResNet152 resnet152(3, 224, 224);
  ModelDimTest(resnet152.GetModel(), input);
  
}

/**
 * Test for pre-trained ResNet models.
 */
TEST_CASE("PreTrainedResNetModelTest", "[FFNModelsTests]")
{
  ResNet18 resnet18(3, 224, 224, true, true);
  arma::mat input(224 * 224 * 3, 1);

  // Check output(referenced from PyTorch) for resnet18.
  PreTrainedModelTest(resnet18.GetModel(), input, 0.00618362, 0.02469635);

  // Check output(referenced from PyTorch) for resnet34.
  ResNet34 resnet34(3, 224, 224, true, true);
  PreTrainedModelTest(resnet34.GetModel(), input, 0.00664139, 0.02662659);

  // Check output(referenced from PyTorch) for resnet50.
  ResNet50 resnet50(3, 224, 224, true, true);
  PreTrainedModelTest(resnet50.GetModel(), input, 0.00266838, 0.01067352);

  // Check output(referenced from PyTorch) for resnet101.
  ResNet101 resnet101(3, 224, 224, true, true);
  PreTrainedModelTest(resnet101.GetModel(), input, 0.00168228, 0.00670624);

  // Check output for(referenced from PyTorch) resnet152.
  ResNet152 resnet152(3, 224, 224, true, true);
  PreTrainedModelTest(resnet152.GetModel(), input, 0.00199318, 0.00799561);
}
