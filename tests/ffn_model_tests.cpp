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
#include <models/yolo/yolo.hpp>
#include <boost/test/unit_test.hpp>

// Use namespaces for convenience.
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
 * Simple test for YOLOv1 model.
 */
BOOST_AUTO_TEST_CASE(YOLOV1ModelTest)
{
  mlpack::ann::YOLO<> yolo(3, 448, 448);
  arma::mat input(448 * 448 * 3, 1), output;
  input.ones();

  // Check output shape.
  yolo.GetModel().Predict(input, output);
  BOOST_REQUIRE_EQUAL(output.n_cols, 1);
  BOOST_REQUIRE_EQUAL(output.n_rows, 7 * 7 * (5 * 2 + 20));
}

BOOST_AUTO_TEST_SUITE_END();
