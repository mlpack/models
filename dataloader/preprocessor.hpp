/**
 * @file dataloader.hpp
 * @author Kartik Dutt
 * 
 * Definition of PreProcessor class for popular datasets.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_PREPROCESSOR_HPP
#define MODELS_PREPROCESSOR_HPP

#include <mlpack/prereqs.hpp>

/**
 * Contains standatd pre-process functions for popular datasets.
 *
 * @tparam DatasetX Datatype for loading input features.
 * @tparam DatasetY Datatype for prediction features.
 */
template<
  typename DatasetX = arma::mat,
  typename DatasetY = arma::mat
>
class PreProcessor
{
 public:
  static void MNIST(DatasetX& /* trainX */,
                    DatasetY& trainY,
                    DatasetY& /* validX */,
                    DatasetY& validY,
                    DatasetX& /* testX */)
  {
    trainY = trainY + 1;
    validY = validY + 1;
  }

  static void PascalVOC(DatasetX& /* trainX */,
                        DatasetY& /* trainY */,
                        DatasetY& /* validX */,
                        DatasetY& /* validY */,
                        DatasetX& /* testX */)
  {
    // Nothing to do here. Added to match the rest of the codebase.
  }

  static void CIFAR10(DatasetX & /* trainX */,
                      DatasetY & /* trainY */,
                      DatasetY & /* validX */,
                      DatasetY & /* validY */,
                      DatasetX & /* testX */)
  {
    // Nothing to do here. Added to match the rest of the codebase.
  }

  /**
   * Converts image to channel first format used in PyTorch. Performs the same function
   * as torch.transforms.ToTensor().
   *
   * @param trainFeatures Input features that will be converted into channel first format.
   * @param imageWidth Width of the image in dataset.
   * @param imageHeight Height of the image in dataset.
   * @param imageDepth Depth / Number of channels of the image in dataset.
   */
  static void ChannelFirstImages(Dataset& trainFeatures,
                                 const size_t imageWidth,
                                 const size_t imageHeight,
                                 const size_t imageDepth,
                                 bool normalize = true)
  {
    for (size_t idx = 0; idx < trainFeatures.n_cols; idx++)
    {
      // Create a copy of the current image so that the image isn't affected.
      arma::cube inputTemp(trainFeatures.col(idx).memptr(), 3, 224, 224);

      size_t currentOffset = 0;
      for (size_t i = 0; i < inputTemp.n_slices; i++)
      {
          trainFeatures.col(idx)(arma::span(currentOffset, currentOffset +
              inputTemp.slice(i).n_elem - 1), arma::span()) =
              arma::vectorise(inputTemp.slice(i).t());
          currentOffset += inputTemp.slice(i).n_elem;
      }
    }

    if (normalize)
    {
      // Convert each element to uint8 first and then divide by 255.
      for (size_t i = 0; i < trainFeatures.n_elem; i++)
      {
        trainFeatures(i) = ((uint8_t) (trainFeatures(i)) / 255.0);
      }
    }
  }
};

#endif
