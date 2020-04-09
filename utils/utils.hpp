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
#include <curl/curl.h>
#include <mlpack/core.hpp>

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
   * @param fileName output fileName (including path).
   * @returns 0 to determine success.
   */
  static int DownloadFile(const std::string url,
                          const std::string fileName,
                          const std::string name = "",
                          const bool progressBar = true)
  {
    CURL *curl;
    FILE *outputFile;
    CURLcode result;
    curl = curl_easy_init();
    if (curl)
    {
      // Create file for writing.
      outputFile = fopen(fileName.c_str(), "wb");

      // Setup curl object to perform desired operation.
      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      if (!progressBar)
      {
        // Disable progressbar.
        //curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
      }
      else
      {
       // curl_easy_setopt(curl, CURLOPT_NOPROGRESS, true);
      }

      curl_easy_setopt(curl, CURLOPT_WRITEDATA, outputFile);
      result = curl_easy_perform(curl);

      if (result != CURLE_OK)
      {
        mlpack::Log::Fatal << "Download Failed!" << std::endl;
        return 1;
      }
      curl_easy_cleanup(curl);
      fclose(outputFile);
    }
    return 0;
  }

  static bool CompareSHA256(std::string path, std::string hash)
  {
    return true; // Complete this function.
  }

  static std::string GetSHA256(std::string path)
  {
    // Complete This function.
    return std::string(1, 'a');
  }
};
#endif