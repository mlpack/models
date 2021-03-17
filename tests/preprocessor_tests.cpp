/**
 * @file preprocessor_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities of PreProcessor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <dataloader/preprocessor.hpp>
#include <dataloader/dataloader.hpp>
#include "catch.hpp"

TEST_CASE("YOLOPreProcessor", "[PreProcessorsTest]")
{
  arma::field<arma::vec> input;
  input.set_size(1, 1);

  arma::vec bBox(5);
  bBox << 2 << 84 << 48 << 493 << 387 << arma::endr;
  input(0, 0) = bBox;
  arma::mat output;

  // Single input check.
  PreProcessor<arma::mat, arma::field<arma::vec>>::YOLOPreProcessor(
      input, output, 1, 500, 387);
  REQUIRE(arma::accu(output) == Approx(8.3342).epsilon(1e-5));

  input.clear();
  input.set_size(1, 3);
  input(0, 0) = bBox;

  // Multiple bounding boxes check.
  bBox.clear();
  bBox.set_size(15);
  bBox << 8 << 341 << 217 << 487 << 375 << 8 << 114 << 209 << 183 <<
      298 << 19 << 237 << 110 << 320 << 176 << arma::endr;
  input(0, 1) = bBox;

  bBox.clear();
  bBox.set_size(5);
  bBox << 7 << 157 << 90 << 486 << 372 << arma::endr;
  input(0, 2) = bBox;

  PreProcessor<arma::mat, arma::field<arma::vec>>::YOLOPreProcessor(
      input, output, 1, 500, 387);

  arma::vec desiredSum(3);
  desiredSum << 8.3342 << 18.4093 << 7.13195 << arma::endr;
  for (size_t i = 0; i < output.n_cols; i++)
    REQUIRE(arma::accu(output.col(i)) == Approx(desiredSum(i)).epsilon(1e-5));

  desiredSum << 4.6671 << 10.70465 << 4.065975 << arma::endr;
  PreProcessor<arma::mat, arma::field<arma::vec>>::YOLOPreProcessor(
      input, output, 3, 500, 387);
  for (size_t i = 0; i < output.n_cols; i++)
    REQUIRE(arma::accu(output.col(i)) == Approx(desiredSum(i)).epsilon(1e-5));


  // For better unit testing, we create a very small output grid of size
  // numBoxes * 5 + numClasses, where numBoxes = 1, numClasses = 2.
  // The grid width and height will be 2 x 2. Hence, for
  // single input label, target map will be of size 1 x 2 x 2 x 7.
  input.clear();
  input.set_size(1, 1);
  bBox.clear();
  bBox.set_size(5);
  bBox << 0 << 157 << 90 << 486 << 300 << arma::endr;
  input(0, 0) = bBox;

  PreProcessor<arma::mat, arma::field<arma::vec>>::YOLOPreProcessor(
      input, output, 1, 500, 387, 2, 2, 1, 2);

  arma::mat desiredOutput(2 * 2 * 7, 1);
  desiredOutput.zeros();
  // To convert 4d Tensor to 1D array use tensor.numpy().ravel().
  desiredOutput(3) = 0.2860;
  desiredOutput(7) = 0.0078;
  desiredOutput(11) = 0.6580;
  desiredOutput(15) = 0.5426;
  desiredOutput(19) = 1.0;
  desiredOutput(23) = 1.0;

  // check for each value in matrix.
  double tolerance = 1e-1;
  for (size_t i = 0; i < output.n_elem; i++)
  {
    if (std::abs(output(i)) < tolerance / 2)
      BOOST_REQUIRE_SMALL(desiredOutput(i) == Approx(0.0).margin(tolerance / 2));
    else
      REQUIRE(desiredOutput(i) == Approx(output(i)).epsilon(1e-2));
  }
}
