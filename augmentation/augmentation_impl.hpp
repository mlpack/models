/**
 * @file augmentation_impl.hpp
 * @author Kartik Dutt
 * 
 * Implementation of Augmentation class for augmenting data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
// Incase it has not been included already.
#include "augmentation.hpp"

#ifndef MODELS_AUGMENTATION_IMPL_HPP
#define MODELS_AUGMENTATION_IMPL_HPP

template<typename DatasetType>
Augmentation<DatasetType>::Augmentation() :
    augmentations(std::vector<std::string>()),
    augmentationProbability(0.2)
{
  // Nothing to do here.
}

template<typename DatasetType>
Augmentation<DatasetType>::Augmentation(
    const std::vector<std::string>& augmentations,
    const double augmentationProbability) :
    augmentations(augmentations),
    augmentationProbability(augmentationProbability)
{
  // Sort the vector to place resize parameter to the front of the string.
  // This prevents constant look ups for resize.
  sort(this->augmentations.begin(), this->augmentations.end(), [](
      std::string& str1, std::string& str2)
          {
            return str1.find("resize") != std::string::npos;
          });

  // Fill augmentation map with supported augmentations other than resize.
  InitializeAugmentationMap();
}

template<typename DatasetType>
void Augmentation<DatasetType>::Transform(DatasetType& dataset,
                                          const size_t datapointWidth,
                                          const size_t datapointHeight,
                                          const size_t datapointDepth)
{
  size_t i = 0;
  if (this->HasResizeParam())
  {
    this->ResizeTransform(dataset);
    i++;
  }

  for (; i < augmentations.size(); i++)
  {
    if (augmentationMap.count(augmentations[i]))
    {
      augmentationMap[augmentations[i]](dataset, datapointWidth,
        datapointHeight, datapointDepth, augmentations[i]);
    }
  }
}

template<typename DatasetType>
void Augmentation<DatasetType>::ResizeTransform(
    DatasetType& dataset,
    const size_t datapointWidth,
    const size_t datapointHeight,
    const size_t datapointDepth,
    const std::string& augmentation)
{
  size_t outputWidth = 0, outputHeight = 0;

  // Get output width and output height.
  GetResizeParam(outputWidth, outputHeight);

  // We will use mlpack's bilinear interpolation layer to
  // resize the input.
  mlpack::ann::BilinearInterpolation<DatasetType, DatasetType> resizeLayer(
      datapointWidth, datapointHeight, outputWidth, outputHeight,
      datapointDepth);

  // Not sure how to avoid a copy here.
  DatasetType output;
  resizeLayer.Forward(dataset, output);
  dataset = output;
}

template<typename DatasetType>
void Augmentation<DatasetType>::InitializeAugmentationMap()
{
  // Fill the map here.
}

#endif
