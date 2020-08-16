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

#define BOOST_TEST_DYN_LINK
#include <dataloader/preprocessor.hpp>
#include <dataloader/dataloader.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(PreProcessorsTest);

BOOST_AUTO_TEST_CASE(YOLOPreProcessor)
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
  BOOST_REQUIRE_CLOSE(arma::accu(output), 8.3342, 1e-3);

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
    BOOST_REQUIRE_CLOSE(arma::accu(output.col(i)), desiredSum(i), 1e-3);

  desiredSum << 4.6671 << 10.70465 << 4.065975 << arma::endr;
  PreProcessor<arma::mat, arma::field<arma::vec>>::YOLOPreProcessor(
      input, output, 3, 500, 387);
  for (size_t i = 0; i < output.n_cols; i++)
    BOOST_REQUIRE_CLOSE(arma::accu(output.col(i)), desiredSum(i), 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();
