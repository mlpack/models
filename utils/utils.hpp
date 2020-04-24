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
#include <boost/filesystem.hpp>

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
   * @param absolutePath Boolean to determine if path is absolute or relative.
   * @return true if path exists else false. Defaults to false.
   */
  static bool PathExists(std::string path, bool absolutePath = false)
  {
    struct stat buffer;
    // Set correct path.
    std::string filePath = absolutePath ? path :
      boost::filesystem::current_path().string() + "/" + path;
    return (stat(filePath.c_str(), &buffer) == 0);
  }

  /**
   * Downloads files using boost asio.
   *
   * For more information on how to download using boost asio, refer to
   * the following boost_asio/example/cpp03 on boost asio examples.
   *
   * @param url URL for file which is to be downloaded.
   * @param downloadPath Output file path.
   * @param name Prints name of the file.
   * @param absolutePath Boolean to determine if path is absolute or relative.
   * @param silent Boolean to display details of file being downloaded.
   * @param serverName Server to connect to, for downloading.
   * @returns 0 to determine success.
   */
  static int DownloadFile(const std::string url,
                          const std::string downloadPath,
                          const std::string name = "",
                          const bool absolutePath = false,
                          const bool silent = true,
                          const std::string serverName =
                              "www.mlpack.org")
  {
    // IO functionality by boost core.
    boost::asio::io_service ioService;
    // Use TCP protocol by boost asio to make a connection to desired server.
    boost::asio::ip::tcp::resolver resolver(ioService);
    // Resolver will converts the the query object into list of end-points.
    boost::asio::ip::tcp::resolver::query query(serverName, "80",
        boost::asio::ip::resolver_query_base::numeric_service);
    // The list of endpoints is returned using an iterator.
    boost::asio::ip::tcp::resolver::iterator endPoint = resolver.resolve(query);
    boost::asio::ip::tcp::resolver::iterator end;

    // Establish a connection by trying to connect with each port.
    boost::asio::ip::tcp::socket socket(ioService);
    boost::system::error_code error = boost::asio::error::host_not_found;
    // Iterate over ports.
    while (error && endPoint != end)
    {
      socket.close();
      socket.connect(*endPoint++, error);
    }

    boost::asio::streambuf request;
    std::ostream requestStream(&request);
    requestStream << "GET " << url << " HTTP/1.1\r\n";
    requestStream << "Host: " << serverName << "\r\n";
    requestStream << "Accept: */*\r\n";
    requestStream << "Connection: close\r\n\r\n";

    if (!silent)
    {
      mlpack::Log::Info << "Connected to " << serverName <<
          ". Attempting download of "<< name << std::endl;
    }

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
    while (std::getline(responseStream, header) && header != "\r")
    {
      // Nothing to do here.
    }

    // Write remaining data in response if any.
    std::string filePath = absolutePath ? downloadPath :
      boost::filesystem::current_path().string() + "/" + downloadPath;

    std::ofstream outputFile(filePath.c_str(), std::ofstream::out |
        std::ofstream::binary);
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

    if (error != boost::asio::error::eof)
    {
      mlpack::Log::Fatal << "Download Failed!" << std::endl;
      return 1;
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
   * @param absolutePath Boolean to determine if path is absolute or relative.
   * @returns String of CRC32 checksum.
   */
  static std::string GetCRC32(const std::string path,
                              const bool absolutePath = false)
  {
    boost::crc_32_type hash;
    std::string filePath = absolutePath ? path :
      boost::filesystem::current_path().string() + "/" + path;
    std::ifstream inputFile(path.c_str(), std::ios::in | std::ios::binary);
    // Read File in chunks to prevent reading whole file into memory.
    std::vector<char> buffer(2048);
    while (inputFile.read(&buffer[0], buffer.size()))
    {
      hash.process_bytes(&buffer[0], inputFile.gcount());
    }

    std::stringstream hashString;
    hashString << std::hex << hash.checksum();
    return hashString.str();
  }

  /**
   * Deletes the file whose path is given.
   *
   * @param path Path where file to be removed is stored.
   * @param absolutePath Boolean to determine if path is absolute or relative.
   * @returns An integer, 0 if file was removed and 1 if it wasn't.
   */
  static int RemoveFile(const std::string path,
                        const bool absolutePath = false)
  {
    std::string filePath = absolutePath ? path :
      boost::filesystem::current_path().string() + "/" + path;
    std::remove(filePath.c_str());
    if (PathExists(path) != 0)
    {
      mlpack::Log::Warn << "Error Deleting File." << std::endl;
      return 1;
    }

    return 0;
  }
};
#endif
