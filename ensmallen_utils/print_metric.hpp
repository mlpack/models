/**
 * @file utils.hpp
 * @author Kartik Dutt
 *
 * Definition of PrintMetric class.
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
 * Prints metric on training / validation set.
 *
 * @tparam ANNType Type of model which will be used for evaluating metric.
 * @tparam MetricType Metric class which must have static `Evaluate` function
 *                    that will be called at the end of the epoch.
 * @tparam InputType Arma type of dataset features.
 * @tparam OutputType Arma type of dataset labels.
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
   * @param network Network type which will be saved periodically.
   * @param features Input features on which model will be evaluated.
   * @param responses Ground truth label for the mdoel.
   * @param metricName Metric name which will be printed after each epoch.
   * @param trainData Boolean to determine whether dataset corresponds to
   *                  training data or validation data.
   * @param output Outputstream where output will be directed.
   */
  PrintMetric(AnnType &network,
              const InputType &features,
              const OutputType &responses,
              const std::string metricName = "metric",
              const bool trainData = false,
              std::ostream &output = arma::get_cout_stream()) :
              network(network),
              features(features),
              responses(responses),
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
    const double localObjective = MetricType::Evaluate(predictions, responses);
    if (!std::isnan(localObjective))
    {
      std::string outputString = (trainData == true) ? "Train " : "Validation ";
      outputString = outputString + metricName + " : " +
          std::to_string(localObjective);
      output << outputString << std::endl;
    }
    return false;
  }

 private:
  // Reference to the model which will be used for evaluated using the metric.
  AnnType& network;

  // Dataset which will be used for evaluating the metric.
  InputType features;

  // Dataset labels / predictions that will be used for evaluating the dataset.
  OutputType responses;

  // Locally held string that depicts the name of the metric.
  std::string metricName;

  // Locally held boolean to determin whether evaluation is done on train data
  // or validation data.
  bool trainData;

  // The output stream that all data is to be sent to; example: std::cout.
  std::ostream& output;
};

} // namespace ens

#endif
