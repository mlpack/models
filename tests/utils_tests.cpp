/**
 * @file utils_tests.cpp
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
#include <dataloader/datasets.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(UtilsTest);

/**
 * Simple test for Data Downloader.
 */
BOOST_AUTO_TEST_CASE(DownloadFileTest)
{
  // To check downloader, perform the following :
  // 1. Download the file.
  // 2. Check for it's existence.
  // 3. Match checksum for given file.
  Utils::DownloadFile("iris.csv", "./../data/iris.csv");
  BOOST_REQUIRE(Utils::PathExists("./../data/iris.csv"));
  BOOST_REQUIRE(Utils::CompareCRC32("./../data/iris.csv", 
      "152ec23b"));
  // Clean up.
  Utils::RemoveFile("./../data/iris.csv");
}

/**
 * Simple test for CompareCRC32.
 */
BOOST_AUTO_TEST_CASE(CheckSumTest)
{
  // Download the file and verify it's checksum.
  Utils::DownloadFile("iris_test.csv", "./../data/iris_test.csv");
  BOOST_REQUIRE(Utils::CompareCRC32("./../data/iris_test.csv",
      "c1d67a8f"));
  // Clean up.
  Utils::RemoveFile("./../data/iris.csv");
}

/**
 * Simple test for PathExists.
 */
BOOST_AUTO_TEST_CASE(PathExistsTest)
{
  // Check for files that exist.
  BOOST_REQUIRE(Utils::PathExists("./../models/CMakeLists.txt"));
  BOOST_REQUIRE(Utils::PathExists("./../CMakeLists.txt"));
}

BOOST_AUTO_TEST_SUITE_END();
