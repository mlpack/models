/**
 * @file utils.hpp
 * @author Kartik Dutt
 *
 * Definition of Ensmallen Utility functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_CALLBACKS_PRINT_METRIC_HPP
#define ENSMALLEN_CALLBACKS_PRINT_METRIC_HPP

#include <ensmallen.hpp>
#include <functional>

namespace ens {

/**
 * Prints metric on validation set.
 */
template<typename AnnType,
         class MetricType,
         typename InputType = arma::mat,
         typename OutputType = arma::mat
>
class PrintMetric
{
 public:
  /**
   * Constructor for PrintMetric class.
   */
  PrintMetric(AnnType& network,
              const InputType features,
              const std::string metricName = "metric",
              const bool trainData = false,
              std::ostream& output = arma::get_cout_stream()) :
              network(network),
              features(features),
              metricName(metricName),
              trainData(trainData),
              output(output)
  {
    // Nothing to do here.
  }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double /* objective */)
  {
    OutputType predictions;
    network.Predict(features, predictions);
    const double localObjective = MetricType::Evaluate(features, predictions);
    if (!std::isnan(localObjective))
    {
      std::string outputString = (trainData == true) ? "train " : "validation ";
      outputString = outputString + metricName + " : " +
          std::to_string(localObjective);
      output << outputString << std::endl;
    }
    return true;
  }

 private:
  // Reference to the model which will be used for evaluated using the metric.
  AnnType& network;

  // Dataset which will be used for evaluating the metric.
  InputType features;

  // Locally held string that depicts the name of the metric.
  std::string metricName;

  // Locally held boolean to determin whether evaluation is done on train data or
  // validation data.
  bool trainData;

  // The output stream that all data is to be sent to; example: std::cout.
  std::ostream& output;
};

}

#endif