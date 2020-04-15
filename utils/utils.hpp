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
#include <boost/crc.hpp>
#include <mlpack/core.hpp>

/**
 * Utils class to provide utility functions.
 */
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
   * Downloads files using boost asio.
   * 
   * @param url URL for file which is to be downloaded.
   * @param fileName Output fileName (including path).
   * @param name Prints name of the file.
   * @param silent Boolean to display details of file being downloaded.
   * @param serverName Server to connect to, for downloading.
   * @returns 0 to determine success.
   */
  static int DownloadFile(const std::string url,
                          const std::string fileName,
                          const std::string name = "",
                          const bool silent = true,
                          const std::string serverName = "https://www.mlpack.org/datasets/")
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
    std::ofstream outputFile(fileName.c_str(), std::ofstream::out |
        std::ofstream::binary);
    if (response.size() > 0)
    {
      outputFile << &response;
    }

    // Read the response and write to desired file.
    do
    {
      outputFile << &response;
    } while (boost::asio::read(socket, response,
        boost::asio::transfer_at_least(1), error));

    if (error != boost::asio::error::eof)
    {
      mlpack::Log::Fatal << "Error in Downloading!" << std::endl;
    }
    outputFile.close();
    return 0;
  }

  /**
   * Compare CheckSum for provided file and hash.
   *
   * @param path Path to file for which hash to be calculated.
   * @param hash Desired value of hash for the given file.
   * @returns Boolean determining whether the hash matches or not.
   */
  static bool CompareCRC32(std::string path, std::string hash)
  {
    return GetCRC32(path) == hash;
  }

  /**
   * Calculates CRC32 checksum for given file.
   *
   * @param path Path for file whose checksum is to be calculated.
   * returns String of CRC32 checksum.
   */
  static std::string GetCRC32(std::string path)
  {
    boost::crc_32_type hash;
    std::ifstream inputFile(path.c_str(), std::ios::in | std::ios::binary);
    // Read File in chunks to prevent reading whole file into memory.
    std::vector<char> buffer(1024);
    while(inputFile.read(&buffer[0], buffer.size()))
    {
      hash.process_bytes(&buffer[0], inputFile.gcount());
    }
    std::stringstream hashString;
    hashString << std::hex << hash.checksum();
    std::cout << hashString.str() << std::endl;
    return hashString.str();
  }

  /**
   * Deletes the file whose path is given.
   *
   * @param path Path where file to be removed is stored.
   */
  static void RemoveFile(std::string path)
  {
    if (std::remove(path.c_str()) != 0)
    {
      mlpack::Log::Warn << "Error Deleting File." << std::endl;
    }
  }
};
#endif
