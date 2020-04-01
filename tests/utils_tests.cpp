/**
 * @file dataloader_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities of utils.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BOOST_TEST_DYN_LINK
#include <utils/utils.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(UtilsTest);

/**
 * Simple test for Data Downloader.
 */
BOOST_AUTO_TEST_CASE(DownloadFileTest)
{
  Utils::DownloadFile("https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/mnist-dataset/mnist_train.csv", "./../data/mnist_train.csv");
  BOOST_REQUIRE_EQUAL((int)Utils::PathExists("./../data/mnist_train.csv"), 1);
  BOOST_REQUIRE_EQUAL((int)Utils::CompareSHA256("./../data/mninst_train.csv", "correct-checksum-here"), 1);

  Utils::DownloadFile("https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/mnist-dataset/mnist_test.csv", "./../data/mnist_test.csv");
  BOOST_REQUIRE_EQUAL((int)Utils::PathExists("./../data/mnist_test.csv"), 1);
  BOOST_REQUIRE_EQUAL((int)Utils::CompareSHA256("./../data/mninst_test.csv", "correct-checksum-here"), 1);

}

/**
 * Simple test for CompareSHA256.
 */
BOOST_AUTO_TEST_CASE(CheckSumTest)
{
  BOOST_REQUIRE_EQUAL((int)Utils::CompareSHA256("./../data/models/lenet.hpp", "correct-checksum-here"), 1);
  BOOST_REQUIRE_EQUAL((int)Utils::CompareSHA256("./.gitignore", "correct-checksum-here"), 1);
}

/**
 * Simple test for PathExists.
 */
BOOST_AUTO_TEST_CASE(PathExistsTest)
{
  BOOST_REQUIRE_EQUAL(int(Utils::PathExists("./../models/lenet.hpp")), 1);
  BOOST_REQUIRE_EQUAL(int(Utils::PathExists("./../models/lenet_impl.hpp")), 1);
}

BOOST_AUTO_TEST_SUITE_END();