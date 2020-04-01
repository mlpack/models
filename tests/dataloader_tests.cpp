/**
 * @file dataloader_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities of dataloader.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BOOST_TEST_DYN_LINK
#include <dataloader/dataloader.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(DataLoadersTest);

/**
 * Simple test for MNIST Dataloader.
 */
BOOST_AUTO_TEST_CASE(MNISTDataLoaderTest)
{
  DataLoader<> dataloader("mnist", true, 0.80);
  // Check for correct dimensions.
  BOOST_REQUIRE_EQUAL(dataloader.TrainX().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.TestX().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.ValidX().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.TrainX().n_rows, 33600);

  // Check for SHA256 Checksum.
  
}

BOOST_AUTO_TEST_SUITE_END();