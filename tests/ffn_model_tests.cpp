/**
 * @file model_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities and performance of models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack.hpp>
#include <utils/utils.hpp>
#include <dataloader/dataloader.hpp>
#include <models/darknet/darknet.hpp>
#include <models/yolo/yolo.hpp>
#include <models/resnet/resnet.hpp>
#include <models/mobilenet/mobilenet_v1.hpp>
#include "catch.hpp"

using namespace mlpack::models;

/**
 * Checks for the output dimensions of the model.
 *
 * @tparam ModelType Type of model to check.
 *
 * @param model The model to test.
 * @param input Input to pass to the model.
 * @param n_rows Output rows to check against.
 * @param n_cols Output columns to check against..
 */
template <typename ModelType>
void ModelDimTest(ModelType& model,
               arma::mat& input,
               const size_t n_rows = 1000,
               const size_t n_cols = 1)
{
  arma::mat output;
  model.Predict(input, output);
  REQUIRE(output.n_rows == n_rows);
  REQUIRE(output.n_cols == n_cols);
}

/**
 * Checks for the output sum of the model for a 
 *     single and multiple batch input.
 *
 * @tparam ModelType Type of model to check.
 *
 * @param model The model to test.
 * @param input Input to pass to the model.
 * @param singleBatchOutput Sum of the output of a single batch.
 * @param multipleBatchOutput Sum of the output of Multiple batches.
 * @param numBatches Number of batches to create for input.
 */
template <typename ModelType>
void PreTrainedModelTest(ModelType& model,
                         arma::mat& input,
                         const double singleBatchOutput,
                         const double multipleBatchOutput,
                         const size_t numBatches = 4)
{
  arma::mat multipleBatchInput(input.n_rows, numBatches), output;
  input.ones();
  multipleBatchInput.ones();

  // Run prediction for single batch.
  model.Predict(input, output);
  REQUIRE(arma::accu(output) == Approx(singleBatchOutput).epsilon(1e-2));

  // Run prediction for multiple batch.
  model.Predict(multipleBatchInput, output);
  REQUIRE(arma::accu(output) == Approx(multipleBatchOutput).epsilon(1e-2));
}

/**
 * Simple test for Darknet model.
 */
TEST_CASE("DarknetModelTest", "[FFNModelsTests]")
{
  arma::mat input(224 * 224 * 3, 1);
  DarkNet19 darknet19(3, 224, 224, 1000);

  // Check output shape.
  ModelDimTest(darknet19.GetModel(), input);

  // Repeat for DarkNet-53.
  DarkNet53 darknet53(3, 224, 224, 1000);
  ModelDimTest(darknet53.GetModel(), input);
}

/**
 * Simple test for YOLOv1 model.
 */
TEST_CASE("YOLOV1ModelTest", "[FFNModelsTests]")
{
  arma::mat input(448 * 448 * 3, 1);
  YOLO<> yolo(3, 448, 448);

  // Check output shape.
  ModelDimTest(yolo.GetModel(), input, (7 * 7 * (5 * 2 + 20)), 1);
}

/**
 * Simple test for ResNet(18, 34, 50) models.
 */
TEST_CASE("ResNetModelTest", "[FFNModelsTests]")
{
  arma::mat input(224 * 224 * 3, 1);

  // Check output shape for resnet18.
  ResNet18 resnet18(3, 224, 224);
  ModelDimTest(resnet18.GetModel(), input);

  // Check output shape for resnet34.
  ResNet34 resnet34(3, 224, 224);
  ModelDimTest(resnet34.GetModel(), input);

  // Check output shape for resnet50.
  ResNet50 resnet50(3, 224, 224);
  ModelDimTest(resnet50.GetModel(), input);
}

/**
 * Simple test for ResNet101 models.
 * Have been split from the ResNetModelTests because of memory requirements.
 */
TEST_CASE("ResNet101ModelTest", "[FFNModelsTests]")
{
  arma::mat input(224 * 224 * 3, 1);

  // Check output shape for resnet101.
  ResNet101 resnet101(3, 224, 224);
  ModelDimTest(resnet101.GetModel(), input);
}

/**
 * Simple test for ResNet152 models.
 * Have been split from the ResNetModelTests because of memory requirements.
 * This test will not run on windows because of the memory requirements.
 */
#if !defined(WIN32)
  TEST_CASE("ResNet152ModelTest", "[FFNModelsTests]")
  {
    arma::mat input(224 * 224 * 3, 1);

    // Check output shape for resnet152.
    ResNet152 resnet152(3, 224, 224);
    ModelDimTest(resnet152.GetModel(), input);
  }
#endif
/**
 * Test for pre-trained ResNet(18, 34, 50) models.
 */
TEST_CASE("PreTrainedResNetModelTest", "[FFNModelsTests]")
{
  arma::mat input(224 * 224 * 3, 1);

  // Check output(referenced from PyTorch) for resnet18.
  ResNet18 resnet18(3, 224, 224, true, true);
  PreTrainedModelTest(resnet18.GetModel(), input, 0.00618362, 0.02469635);

  // Check output(referenced from PyTorch) for resnet34.
  ResNet34 resnet34(3, 224, 224, true, true);
  PreTrainedModelTest(resnet34.GetModel(), input, 0.00664139, 0.02662659);

  // Check output(referenced from PyTorch) for resnet50.
  ResNet50 resnet50(3, 224, 224, true, true);
  PreTrainedModelTest(resnet50.GetModel(), input, 0.00266838, 0.01067352);
}

/**
 * Test for pre-trained ResNet101 model.
 * Have been split from the PreTrainedResNetModelTests because of
 *     memory requirements.
 */
TEST_CASE("PreTrainedResNet101ModelTest", "[FFNModelsTests]")
{
  arma::mat input(224 * 224 * 3, 1);

  // Check output(referenced from PyTorch) for resnet101.
  ResNet101 resnet101(3, 224, 224, true, true);
  PreTrainedModelTest(resnet101.GetModel(), input, 0.00168228, 0.00670624);
}

/**
 * Test for pre-trained ResNet152 model.
 * Have been split from the PreTrainedResNetModelTests because of
 *     memory requirements.
 * This test will not run on windows because of the memory requirements.
 */
#if !defined(WIN32)
  TEST_CASE("PreTrainedResNetModel152Test", "[FFNModelsTests]")
  {
    arma::mat input(224 * 224 * 3, 1);

    // Check output for(referenced from PyTorch) resnet152.
    ResNet152 resnet152(3, 224, 224, true, true);
    PreTrainedModelTest(resnet152.GetModel(), input, 0.00199318, 0.00799561);
  }
#endif

/**
 * Simple test for MobileNetV1 model.
 */
TEST_CASE("MobileNetV1ModelTest", "[FFNModelsTests]")
{
  arma::mat input(224 * 224 * 3, 1);

  // Check output shape for resnet152.
  MobilenetV1 mobilenet(3, 224, 224);
  ModelDimTest(mobilenet.GetModel(), input);
}

/**
 * Test for all pre-trained MobileNetV1 models.
 */
TEST_CASE("PreTrainedMobileNetV1ModelTest", "[FFNModelsTests]")
{
  // Values taken from a PyTorch implementation based on
  // https://github.com/ZFTurbo/MobileNet-v1-Pytorch
  size_t counter = 0;

  // The first dimensions corresponds to the different configs of mobilenet
  // as can be figured out from the below loop and the values inside are
  // from the output of the model which are from index: 0, 500, 999.
  // The output values are obtained from the above mentioned PyTorch
  // implementation of MobileNetV1.
  double targets[16][3] = {{7.982727765920572e-06, 0.0008073403732851148,
      0.0009284192346967757},
      {9.541783219901845e-05, 7.927525439299643e-05, 0.0003265062696300447},
      {0.00010830028622876853, 0.00020112381025683135, 0.0009800317930057645},
      {6.33568488410674e-05, 0.00017718187882564962, 0.0021993769332766533},
      {7.146679126890376e-05, 0.00014385067333932966, 0.001759626786224544},
      {0.0003550674591679126, 0.0007125227712094784, 0.002989133121445775},
      {0.00018564300262369215, 0.0002874033816624433, 0.0027509047649800777},
      {7.508866838179529e-05, 0.0005556890973821282, 0.0033081816509366035},
      {3.287712388555519e-05, 0.00014808539708610624, 0.0028836114797741175},
      {0.00018852800712920725, 0.00014897453365847468, 0.0015567634254693985},
      {0.0001606910373084247, 0.0001062339506461285, 0.007338172290474176},
      {0.00013950835273135453, 0.00043900657328777015, 0.0018902374431490898},
      {0.00030765001429244876, 0.00036887291935272515, 0.004446627572178841},
      {0.00023077597143128514, 0.00023593794321641326, 0.0019488284597173333},
      {0.0001756725978339091, 0.00011693470878526568, 0.000924319785553962},
      {0.0003898103896062821, 0.0003618707705754787, 0.0009399897535331547}
  };
  arma::mat input, output;
  std::vector<double> alpha = {0.25, 0.5, 0.75, 1.0};
  std::vector<int> image_size = {128, 160, 192, 224};
  for (double alpha_val : alpha)
  {
    for (int image_size_val : image_size)
    {
      MobilenetV1 mobilenet(3, image_size_val, image_size_val, alpha_val, 1,
          true, true);
      input.set_size(image_size_val * image_size_val * 3, 1);
      input.fill(1);
      mobilenet.GetModel().Predict(input, output);
      REQUIRE(output[0] == Approx(targets[counter][0]).epsilon(1e-4));
      REQUIRE(output[500] == Approx(targets[counter][1]).epsilon(1e-4));
      REQUIRE(output[999] == Approx(targets[counter][2]).epsilon(1e-4));
      counter++;
    }
  }
}
