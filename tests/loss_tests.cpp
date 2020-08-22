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
  arma::mat input, output;
  input = arma::mat(7 * 7 * 30, 2, arma::fill::ones);
  output = input;

  mlpack::ann::YOLOLoss<> yoloLoss(1);
  BOOST_REQUIRE_EQUAL(yoloLoss.Forward(input, output), 0.0);
}

BOOST_AUTO_TEST_SUITE_END();
