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
#ifndef MODELS_AUGMENTATION_AUGMENTATION_IMPL_HPP
#define MODELS_AUGMENTATION_AUGMENTATION_IMPL_HPP

// Incase it has not been included already.
#include "augmentation.hpp"

template<typename DatasetType>
void Augmentation::Transform(DatasetType& dataset,
                             const size_t datapointWidth,
                             const size_t datapointHeight,
                             const size_t datapointDepth)
{
  // Initialize the augmentation map.
  std::unordered_map<std::string, void(*)(DatasetType&,
       size_t, size_t, size_t, std::string&)> augmentationMap;

  for (size_t i = 0; i < augmentations.size(); i++)
  {
    if (augmentationMap.count(augmentations[i]))
    {
      augmentationMap[augmentations[i]](dataset, datapointWidth,
        datapointHeight, datapointDepth, augmentations[i]);
    }
    else if (this->HasResizeParam(augmentations[i]))
    {
      this->ResizeTransform(dataset, datapointWidth, datapointHeight,
        datapointDepth, augmentations[i]);
    }
    else
    {
      mlpack::Log::Warn << "Unknown augmentation : \'" <<
          augmentations[i] << "\' not found!" << std::endl;
    }
  }
}

template<typename DatasetType>
void Augmentation::ResizeTransform(
    DatasetType& dataset,
    const size_t datapointWidth,
    const size_t datapointHeight,
    const size_t datapointDepth,
    const std::string& augmentation)
{
  size_t outputWidth = 0, outputHeight = 0;

  // Get output width and output height.
  GetResizeParam(outputWidth, outputHeight, augmentation);

  // We will use mlpack's bilinear interpolation layer to
  // resize the input.
  mlpack::ann::BilinearInterpolation<DatasetType, DatasetType> resizeLayer(
      datapointWidth, datapointHeight, outputWidth, outputHeight,
      datapointDepth);

  DatasetType output;
  resizeLayer.Forward(dataset, output);
  dataset = std::move(output);
}

#endif
