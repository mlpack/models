/**
 * @file utils.hpp
 * @author Kartik Dutt
 * 
 * Definition of Utils class.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_UTILS_HPP
#define MODELS_UTILS_HPP

#include <iostream>
#include <cstdlib>
#include <sys/stat.h>

class Utils
{
 public:
  /**
   * Determines whether a path exists.
   * 
   * @param path Global or relative path.
   * @return true if path exists else false.
   */
  static bool PathExists(std::string path)
  {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
  }

  /**
   * Downloads files using wget command.
   * 
   * @param url URL for file which is to be downloaded.
   * @returns 0 to determine success.
   */
  static int DownloadFile(std::string url)
  {
    std:: string command = "wget " + url;
    return system((const char *)command.c_str());
  }
};
#endif