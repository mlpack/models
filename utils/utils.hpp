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
#include <boost/asio.hpp>
#include <cstdlib>
#include <sys/stat.h>
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
   * Downloads files using boost asio.
   * 
   * @param url URL for file which is to be downloaded.
   * @param fileName output fileName (including path).
   * @returns 0 to determine success.
   */
  static int DownloadFile(const std::string url,
                          const std::string fileName,
                          const std::string name = "",
                          const bool progressBar = true,
                          const bool zipFile = true,
                          const std::string serverName = "https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/")
  {
    // IO functionality by boost core.
    boost::asio::io_service ioService;
    // Use TCP protocol by boost asio to make a connection to desired server.
    boost::asio::ip::tcp::resolver resolver(ioService);
    // Resolver will converts the the query object into list of end-points.
    boost::asio::ip::tcp::resolver::query query(serverName, "http");
    // The list of endpoints is returned using an iterator using resolve member function.
    boost::asio::ip::tcp::resolver::iterator endPoints  = resolver.resolve(query);
    boost::asio::ip::tcp::resolver::iterator end;

    // Establish a connection by trying to connect with each port.
    boost::asio::ip::tcp::socket socket(ioService);
    boost::system::error_code error = boost::asio::error::host_not_found;
    // Iterate over ports.
    while (error && endPoints != end)
    {
      socket.close();
      socket.connect(*endPoints++, error);
    }

    boost::asio::streambuf request;
    std::ostream requestStream(&request);
    requestStream << "GET " << url << " HTTP/1.0\r\n";
    requestStream << "Host: " << serverName << "\r\n";
    requestStream << "Accept: */*\r\n";
    requestStream << "Connection: close\r\n\r\n";

    // Sending the request.
    boost::asio::write(socket, request);

    // Read the response status line.
    boost::asio::streambuf response;
    boost::asio::read_until(socket, response, "\r\n");
    // Check that response is OK.
    std::istream responseStream(&response);
    std::string httpVer;
    responseStream >> httpVer;
    unsigned int statusCode;
    responseStream >> statusCode;
    if (statusCode != 200)
    {
      mlpack::Log::Fatal << "Connection returned with status " <<
          statusCode << ". Terminating Connection." << std::endl;
      return 1;
    }

    boost::asio::read_until(socket, response, "\r\n\r\n");
    // Read the response headers.
    std::string header;
    while (std::getline(responseStream, header) && header != "\r");
    // Write remaining data in response if any.
    std::ofstream outputFile(fileName.c_str(), std::ofstream::out | std::ofstream::binary);
    if (response.size() > 0)
    {
      outputFile << &response;
    }

    // Read the response and write to desired file.
    while (boost::asio::read(socket, response,
        boost::asio::transfer_at_least(1), error))
    {
      outputFile << &response;
    }

    outputFile.close();
    return 0;
  }

  static bool CompareSHA256(std::string path, std::string hash)
  {
    return GetSHA256(path) == hash;
  }

  static std::string GetSHA256(std::string path)
  {
    return "";
  }
};
#endif
