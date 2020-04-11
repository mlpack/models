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
#include <openssl/sha.h>
#include <mlpack/core.hpp>

class Utils
{
  /**
   * Progress bar for libCurl CPP API.
   * 
   * @param currentProgress Pointer set with CURLOPT_PROGRESSDATA,
   *        passed along from the application to the callback.
   * @param totalDownload Total data to be downloaded.
   * @param currentDownload Data that has been downloaded.
   * @param totalUpload Total data to be uploaded.
   * @param currentUpload Data Currently uploaded.
   */
  static int ProgressBar(void* currentProgress, double totalDownload,
                  double currentDownload, double totalUpload,
                  double currntUpload)
  {
    int progressBarWidth = 40;
    double progress = currentDownload / (totalDownload + 1e-50);
    size_t downloaded = std::ceil(progress * progressBarWidth);
    std::string progressBar(progressBarWidth, '.');
    std::fill(progressBar.begin(), progressBar.begin() + downloaded, '=');
    progressBar = "[" + progressBar;
    progressBar += "] " + std::to_string(progress * 100);
    std::cout << progressBar << '\r' <<std::flush;

    return CURLE_OK;
  }

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
    CURL* curl;
    FILE* outputFile;
    CURLcode result;
    curl = curl_easy_init();
    if (progressBar)
    {
      std::cout << "Downloading " + name << std::endl;
    }

    if (curl)
    {
      // Create file for writing.
      outputFile = fopen(fileName.c_str(), "wb");

      // Setup curl object to perform desired operation.
      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, outputFile);
      if (progressBar)
      {
        // Disable internal progress bar.
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        // Enable progress bar.
        curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, ProgressBar);
      }

      result = curl_easy_perform(curl);

      if (result != CURLE_OK)
      {
        mlpack::Log::Fatal << "Download Failed!" << std::endl;
        return 1;
      }
      if (progressBar)
      {
        std::cout << "\n Download complete!" << std::endl;
      }

      curl_easy_cleanup(curl);
      fclose(outputFile);
    }
    return 0;
  }

  static bool CompareSHA256(std::string path, std::string hash)
  {
    return GetSHA256(path) == hash;
  }

  static std::string GetSHA256(std::string path)
  {
    FILE* inputFile = fopen(path.c_str(), "rb");
    if (!inputFile)
    {
      mlpack::Log::Fatal << "Cannot Open " + path + "." << std::endl;
      return "";
    }

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    const size_t bufferSize = 32768;
    char buffer[bufferSize];
    int bufferRead = 0;
    do
    {
      bufferRead = fread(buffer, 1, bufferSize, inputFile);
      SHA256_Update(&sha256, buffer, bufferRead);
    }while(bufferRead > 0);

    fclose(inputFile);
    SHA256_Final(hash, &sha256);
    return std::string(reinterpret_cast<char*>(hash));
  }
};
#endif
