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
};

#endif
