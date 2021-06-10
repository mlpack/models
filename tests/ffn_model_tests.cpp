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
 * Simple test for Darknet model.
 */
TEST_CASE("DarknetModelTest", "[FFNModelsTests]")
{
  DarkNet<> darknetModel(3, 224, 224, 1000);
  arma::mat input(224 * 224 * 3, 1), output;
  input.ones();

  // Check output shape.
  darknetModel.GetModel().Predict(input, output);
  REQUIRE(output.n_cols == 1);
  REQUIRE(output.n_rows == 1000);

  // Repeat for DarkNet-53.
  DarkNet<> darknet53(3, 224, 224, 1000);
  darknet53.GetModel().Predict(input, output);
  REQUIRE(output.n_cols == 1);
  REQUIRE(output.n_rows == 1000);
}

/**
 * Simple test for YOLOv1 model.
 */
TEST_CASE("YOLOV1ModelTest", "[FFNModelsTests]")
{
  YOLO<> yolo(3, 448, 448);
  arma::mat input(448 * 448 * 3, 1), output;
  input.ones();

  // Check output shape.
  yolo.GetModel().Predict(input, output);
  REQUIRE(output.n_cols == 1);
  REQUIRE(output.n_rows == (7 * 7 * (5 * 2 + 20)));
}

TEST_CASE("ResNetModelTest", "[FFNModelsTests]")
{
  ResNet<> resnet(3, 224, 224);
}
