/**
 * @file augmentation.hpp
 * @author Kartik Dutt
 * 
 * Definition of Augmentation class for augmenting data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/methods/ann/layer/bilinear_interpolation.hpp>
#include <boost/regex.hpp>

#ifndef MODELS_AUGMENTATION_HPP
#define MODELS_AUGMENTATION_HPP

/**
 * Augmentation class used to perform augmentations / transform the data.
 * For the list of supported augmentation, take a look at our wiki page.
 *
 * @code
 * Augmentation<> augmentation({"horizontal-flip", "resize = (224, 224)"}, 0.2);
 * augmentation.Transform(dataloader.TrainFeatures);
 * @endcode
 * 
 * @tparam DatasetX Datatype on which augmentation will be done.
 */
class Augmentation
{
 public:
  //! Create the augmenation class object.
  Augmentation();

  /**
   * Constructor for augmentation class.
   *
   * @param augmentation List of strings containing one of the supported
   *                     augmentation.
   * @param augmentationProbability Probability of applying augmentation on
   *                                the dataset.
   *                                NOTE : This doesn't apply to augmentations
   *                                such as resize.
   */
  Augmentation(const std::vector<std::string>& augmentation,
               const double augmentationProbability);

  /**
   */
  template<typename DatasetType = arma::mat>
  void Transform(DatasetType& dataset);

  template<typename DatasetType = arma::mat>
  void ResizeTransform(DatasetType& dataset);

  template <typename DatasetType = arma::mat>
  void HorizontalFlipTransform(DatasetType &dataset);

  template<typename DatasetType = arma::mat>
  void VerticalFlipTransform(DatasetType& dataset);


 private:
  /**
   * Function to determine if augmentation has Resize function.
   */
  bool HasResizeParam()
  {
    return augmentations.size() <= 0 ? false :
        augmentations[0].find("resize") != std::string::npos ;
  }

  /**
   * Sets size of output width and output height of the new data.
   *
   * @param outWidth Output width of resized data point.
   * @param outHeight Output height of resized data point.
   */
  void GetResizeParam(size_t& outWidth, size_t& outHeight)
  {
    if (!HasResizeParam())
    {
      return;
    }

    outWidth = -1;
    outHeight = -1;

    // Use regex to find one / two  numbers. If only one provided
    // set output width equal to output height.
    boost::regex regex{"[0-9]+"};

    // Create an iterator to find matches.
    boost::sregex_token_iterator matches(augmentations[0].begin(),
        augmentations[0].end(), regex, 0), end;

    size_t matchesCount = std::distance(matches, end);

    if (matchesCount == 0)
    {
      mlpack::Log::Fatal << "Invalid size / shape in " <<
          augmentations[0] << std::endl;
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

  //! Locally held augmentations / transforms that need to be applied.
  std::vector<std::string> augmentations;

  //! Locally held value of augmentation probability.
  double augmentationProbability;
};

#include "augmentation_impl.hpp" // Include implementation.

#endif