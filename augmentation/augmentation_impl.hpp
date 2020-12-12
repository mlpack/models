/**Adding support for more data types to mlpack, it would be preferable to add the support upstream to Armadillo instead, so that may be a better direction to go first. Then very little code modification for mlpack will be necessary./**
 * @file augmentation_impl.hpp
 * @author Kartik Dutt, Sirish
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
    else if(this->HasBlurring(augmentations[i]))
    {
      this->GaussianBlurTransform(dataset, datapointWidth, datapointHeight,
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

template<typename DatasetType>
void Augmentation::GaussianBlurTransform(
    DatasetType& dataset,
    const size_t datapointWidth,
    const size_t datapointHeight,
    const size_t datapointDepth,
    const std::string& augmentation)
{
  //Implementing using http://blog.ivank.net/fastest-gaussian-blur.html
  size_t sigma = 0;
  GetBlurParam(sigma,augmentation);

  size_t rows, cols, depth;
  // Storing initial matrix dimensions
  rows = dataset.n_rows;
  cols = dataset.n_cols;
  depth = dataset.depth;

  // Creating empty object bImage
  DatasetType bImage(datapointHeight, datapointWidth, datapointDepth);

  // Reshaping the matrix for ease of calculation
  dataset = arma::resize(dataset,datapointHeight,datapointWidth,datapointDepth);

  //Significant radius
  size_t rs = arma::ceil(sigma * 2.57);

  for (size_t k = 0; k < datapointDepth; k++)
  {
    for (size_t i = 0; i < datapointHeight; i++)
    {
      for (size_t j = 0; j < datapointWidth; j++)
      {
        size_t val = 0;
        size_t wsum = 0;

        for (size_t iy = i - rs; iy <= i + rs; iy++)
        {
          for (size_t ix = j - rs; ix <= j + rs; ix++)
          {
            size_t x,y;
            x = arma::min(datapointWidth - 1, arma::max(0, ix))
            y = arma::min(datapointHeight - 1, arma::max(0, iy))
            // Calculating sqaured distance
            dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i)
            // Weight of gaussian kernel
            weight = arma::exp(-dsq / (2 * sigma * sigma)) / (datum::pi * 2 * sigma * sigma)
            // Summing all weighted contributions
            val += dataset(y,x,k)* weight
            wsum += weight 
        }
      }
      // Blurred image
      bImage(i,j,k) = arma::round(val / wsum)
    }
  }
  //Restoring the blurred image to original dimensions as that of input
  bImage = arma::resize(bImage,rows,cols,depth);
  dataset = std::move(bImage);
}

#endif
