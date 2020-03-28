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
#include <openssl/sha.h>

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
   * @param fileName output fileName.
   * @returns 0 to determine success.
   */
  static int DownloadFile(std::string url, std::string fileName, std::string name = "")
  {
    if (name.length())
    {
      std::cout << "Downloading "<< name << std::endl;
    }

    std:: string command = "curl " + url + " -o " + fileName + " --progress-bar";
    return system((const char *)command.c_str());
  }

  static bool CompareSHA256(std::string path, std::string hash)
  {
    return true; // Complete this function.
  }

  static std::string GetSHA256(std::string path)
  {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.good())
    {
      std::cout << "Cannot Open File." << std::endl;
    }
    // Complete This function.
    return std::string(1, 'a');
  }
};
#endif