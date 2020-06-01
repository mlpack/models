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
#define BOOST_TEST_DYN_LINK
#include <boost/regex.hpp>
#include <boost/test/unit_test.hpp>
using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(AugmentationTest);

BOOST_AUTO_TEST_CASE(REGEXTest)
{
  // Some accepted formats.
  std::string s = " resize = {  19,    112 }, \
      resize : 133, 442, resize = [12 213]";
  boost::regex expr{"[0-9]+"};
  boost::smatch what;
  boost::sregex_token_iterator iter(s.begin(), s.end(), expr, 0);
  boost::sregex_token_iterator end;
}

BOOST_AUTO_TEST_SUITE_END();
