/**
 * @file augmentation.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities of utils.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <augmentation/augmentation.hpp>
#include "catch.hpp"

TEST_CASE("ResizeAugmentationTest", "[AugmentationTest]")
{
  Augmentation augmentation(std::vector<std::string>(1,
      "resize (5, 4)"), 0.2);

  // Test on a square matrix.
  arma::mat input;
  size_t inputWidth = 2;
  size_t inputHeight = 2;
  size_t depth = 1;
  input.zeros(inputWidth * inputHeight * depth, 2);

  // Resize function called.
  augmentation.Transform(input, inputWidth, inputHeight, depth);

  // Check correctness of input.
  REQUIRE(input.n_cols == 2);
  REQUIRE(input.n_rows == 5 * 4);

  // Test on rectangular matrix.
  inputWidth = 5;
  inputHeight = 7;
  depth = 1;
  input.zeros(inputWidth * inputHeight * depth, 2);

  // Rectangular input to sqaure output.
  std::vector<std::string> augmentationVector = {"horizontal-flip",
      "resize : 8"};
  Augmentation augmentation2(augmentationVector, 0.2);

  // Resize function called.
  augmentation2.Transform(input, inputWidth, inputHeight, depth);

  // Check correctness of input.
  REQUIRE(input.n_cols == 2);
  REQUIRE(input.n_rows == 8 * 8);
}
