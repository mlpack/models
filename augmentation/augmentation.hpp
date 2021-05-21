/**
 * @file augmentation.hpp
 * @author Kartik Dutt, Sirish
 * 
 * Definition of Augmentation class for augmenting data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/methods/ann/layer/bilinear_interpolation.hpp>
#include <mlpack/core/util/to_lower.hpp>
#include <boost/regex.hpp>

#ifndef MODELS_AUGMENTATION_HPP
#define MODELS_AUGMENTATION_HPP

/**
 * Augmentation class used to perform augmentations by transforming the data.
 * For the list of supported augmentation, take a look at our wiki page.
 *
 * @code
 * Augmentation augmentation({"horizontal-flip", "resize = (224, 224)"}, 0.2);
 * augmentation.Transform(dataloader.TrainFeatures);
 * @endcode
 */
class Augmentation
{
 public:
  //! Create the augmentation class object.
  Augmentation() :
      augmentations(std::vector<std::string>()),
      augmentationProbability(0.2)
  {
    // Nothing to do here.
  }

  /**
   * Constructor for augmentation class.
   *
   * @param augmentations List of strings containing one of the supported
   *                      augmentations.
   * @param augmentationProbability Probability of applying augmentation on
   *                                the dataset.
   *                                NOTE : This doesn't apply to augmentations
   *                                such as resize.
   */
  Augmentation(const std::vector<std::string>& augmentations,
               const double augmentationProbability) :
               augmentations(augmentations),
               augmentationProbability(augmentationProbability)
  {
    // Convert strings to lower case.
    for (size_t i = 0; i < augmentations.size(); i++)
      this->augmentations[i] = mlpack::util::ToLower(augmentations[i]);

    // Sort the vector to place resize parameter to the front of the string.
    // This prevents constant lookups for resize.
    sort(this->augmentations.begin(), this->augmentations.end(), [](
        std::string& str1, std::string& str2)
        {
          return str1.find("resize") != std::string::npos;
        });
  }

  /**
   * Applies augmentation to the passed dataset.
   *
   * @tparam DatasetType Datatype on which augmentation will be done.
   * 
   * @param dataset Dataset on which augmentation will be applied.
   * @param datapointWidth Width of a single data point i.e.
   *                       Since each column represents a seperate data
   *                       point.
   * @param datapointHeight Height of a single data point.
   * @param datapointDepth Depth of a single data point. For one 2-dimensional
   *                       data point, set it to 1. Defaults to 1.
   */
  template<typename DatasetType>
  void Transform(DatasetType& dataset,
                 const size_t datapointWidth,
                 const size_t datapointHeight,
                 const size_t datapointDepth = 1);

  /**
   * Applies resize transform to the entire dataset.
   *
   * @tparam DatasetType Datatype on which augmentation will be done.
   * 
   * @param dataset Dataset on which augmentation will be applied.
   * @param datapointWidth Width of a single data point i.e.
   *                       Since each column represents a seperate data
   *                       point.
   * @param datapointHeight Height of a single data point.
   * @param datapointDepth Depth of a single data point. For one 2-dimensional
   *                       data point, set it to 1. Defaults to 1.
   * @param augmentation String containing the transform.
   */
  template<typename DatasetType>
  void ResizeTransform(DatasetType& dataset,
                       const size_t datapointWidth,
                       const size_t datapointHeight,
                       const size_t datapointDepth,
                       const std::string& augmentation);


  /**
   * Applies gaussian blurring to the entire dataset.
   *
   * @tparam DatasetType Datatype on which augmentation will be done.
   * 
   * @param dataset Dataset on which augmentation will be applied.
   * @param datapointWidth Width of a single data point i.e.
   *                       Since each column represents a seperate data
   *                       point.
   * @param datapointHeight Height of a single data point.
   * @param datapointDepth Depth of a single data point. For one 2-dimensional
   *                       data point, set it to 1. Defaults to 1.
   * @param augmentation String containing the transform.
   */

  template<typename DatasetType>
  void GaussianBlurTransform(DatasetType& dataset,
    const size_t datapointWidth,
    const size_t datapointHeight,
    const size_t datapointDepth,
    const std::string& augmentation);

 private:
  /**
   * Function to determine if augmentation has Resize function.
   *
   * @param augmentation Optional argument to check if a string has
   *                     resize substring.
   */
  bool HasResizeParam(const std::string& augmentation = "")
  {
    if (augmentation.length())
      return augmentation.find("resize") != std::string::npos;


    // Search in augmentation vector.
    for(size_t i = 0; i < argumentation.size(); i++)
    {
      if (argumentation[i].find("resize") != std::string::npos)
        return true
    }
    return false
  }
  /*
  * Function to determine whether blurring is needed or not
    Will check if the string has blurring.
  */
  bool HasBlurring(const std::string& augmentation = "")
  {
    if (augmentation.length())
      return augmentation.find("gaussian-blur") != std::string::npos;

    for(size_t i = 0; i < argumentation.size(); i++)
    {
      if(argumentation[i].find("gaussian-blur") != std::string::npos)
        return true
    }
    return false
  }


  /**
   * Sets size of output width and output height of the new data.
   *
   * @param outWidth Output width of resized data point.
   * @param outHeight Output height of resized data point.
   * @param augmentation String from which output width and height
   *                     are extracted.
   */
  void GetResizeParam(size_t& outWidth,
                      size_t& outHeight,
                      const std::string& augmentation)
  {
    if (!HasResizeParam())
      return;

    outWidth = 0;
    outHeight = 0;

    // Use regex to find one or two numbers. If only one provided
    // set output width equal to output height.
    boost::regex regex{"[0-9]+"};

    // Create an iterator to find matches.
    boost::sregex_token_iterator matches(augmentation.begin(),
        augmentation.end(), regex, 0), end;

    size_t matchesCount = std::distance(matches, end);

    if (matchesCount == 0)
    {
      mlpack::Log::Fatal << "Invalid size / shape in " <<
          augmentation << std::endl;
    }

    if (matchesCount == 1)
    {
      outWidth = std::stoi(*matches);
      outHeight = outWidth;
    }
    else
    {
      outWidth = std::stoi(*matches);
      matches++;
      outHeight = std::stoi(*matches);
    }
  }
  /**
   * Sets size of radius/ sigma of the gaussian kernel.
   *
   * @param sigma is the radius of the gaussian kernel specified by user.
   */
  void GetBlurParam(size_t& sigma, const std::string& augmentation)
  {
    if (!HasBlurring(augmentation))
      return;

    sigma = 0;

    // Use regex to find one number.
    // Input should be of form sigma.
    boost::regex regex{"[0-9]+"};

    // Create an iterator to find matches.
    boost::sregex_token_iterator matches(augmentation.begin(),
        augmentation.end(), regex, 0), end;

    size_t matchesCount = std::distance(matches, end);

    if (matchesCount != 1)
    {
      mlpack::Log::Fatal << "Invalid sigma/ radius for gaussian blurring" <<
          augmentation << std::endl;
    }
    else
    {
      sigma = std::stoi(*matches);
    }
  }

  //! Locally held augmentations and transforms that need to be applied.
  std::vector<std::string> augmentations;

  //! Locally held value of augmentation probability.
  double augmentationProbability;

  // The dataloader class should have access to internal functions of
  // the augmentation class.
  template<typename DatasetX, typename DatasetY, class ScalerType>
  friend class DataLoader;
};

#include "augmentation_impl.hpp" // Include implementation.

#endif
