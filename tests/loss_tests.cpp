/**
 * @file loss_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for model specific loss functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#define BOOST_TEST_DYN_LINK
#include <loss_functions/yolo_loss.hpp>
#include <boost/test/unit_test.hpp>

// Use namespaces for convenience.
using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(ModelLossTests);

/**
 * Simple test for YOLOv1 Loss function.
 */
BOOST_AUTO_TEST_CASE(YOLOLossFunctionTest)
{
  arma::mat input, output, delta;
  input = arma::mat(7 * 7 * 30, 2, arma::fill::ones);
  output = input;

  mlpack::ann::YOLOLoss<> yoloLoss(1);
  BOOST_REQUIRE_EQUAL(yoloLoss.Forward(input, output), 0.0);
  yoloLoss.Backward(input, output, delta);
  BOOST_REQUIRE_EQUAL(arma::accu(delta), 0.0);


  // Check objectness loss.
  input.ones();
  output.ones();

  arma::cube inputTemp(const_cast<arma::mat&>(input).colptr(0),
      7, 7, 30, false, false);
  arma::cube outputTemp(const_cast<arma::mat&>(output).colptr(0),
      7, 7, 30, false, false);

  // Set confidence to 0.6.
  inputTemp(0, 0, 4) = 0.6;
  outputTemp(0, 0, 4) = 1.0;
  inputTemp(0, 0, 9) = 0.6;
  outputTemp(0, 0, 9) = 1.0;

  // The bounding boxes are still the same.
  // Since bounding box coordinates are same, loss will be same as
  // numBoxes * (lambda_noobj * (1 - confidence) * (c_pred - c_target) +
  // (iou - confidence) * c_pred).
  double noObjectLoss = 2 * 0.5 * 0.4 * 0.16;
  BOOST_REQUIRE_CLOSE(yoloLoss.Forward(input, output), noObjectLoss /
      input.n_cols, 1e-3);

  // Include classification error.
  inputTemp(0, 0, 10) = 0.6;
  double classificationLoss = 0.16; // (1.0 - 0.6)^2.
  BOOST_REQUIRE_CLOSE(yoloLoss.Forward(input, output),
      (noObjectLoss + classificationLoss) / input.n_cols, 1e-3);

  // Including IoU error in total loss.
  // For simplicity, let's keep the same left coordinate.
  // For x1 = 1, y1 = 1, h1 = 0.5, w1 = 0.5 and x2 = 1, y2 = 1,
  // h2 = 0.6, w2 = 0.6.
  inputTemp(0, 0, 2) = 0.5;
  inputTemp(0, 0, 3) = 0.5;

  outputTemp(0, 0, 2) = 0.6;
  outputTemp(0, 0, 3) = 0.6;

  // Loss due to bounding box and iou.
  double boundingBoxError = 5.0 * 2 * std::pow(std::sqrt(0.5) -
      std::sqrt(0.6), 2) + 0.01467;
  BOOST_REQUIRE_CLOSE(yoloLoss.Forward(input, output),
      (noObjectLoss + classificationLoss + boundingBoxError) /
      input.n_cols, 1e-2);
}

BOOST_AUTO_TEST_SUITE_END();
