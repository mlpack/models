
.. _program_listing_file__home_aakash_models_models_mobilenet_mobilenet_v1.hpp:

Program Listing for File mobilenet_v1.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_models_models_mobilenet_mobilenet_v1.hpp>` (``/home/aakash/models/models/mobilenet/mobilenet_v1.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

    author = {Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
        Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam},
    title = {MobileNets: Efficient Convolutional Neural Networks for Mobile
        Vision Applications},
    year = {2017},
    url = {https://arxiv.org/pdf/1704.04861}
   }
   
   #ifndef MODELS_MODELS_MOBILENET_MOBILENET_V1_HPP
   #define MODELS_MODELS_MOBILENET_MOBILENET_V1_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/init_rules/random_init.hpp>
   #include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>
   
   #include "./../../utils/utils.hpp"
   
   namespace mlpack {
   namespace models {
   
   template<
     typename OutputLayerType = ann::CrossEntropyError<>,
     typename InitializationRuleType = ann::RandomInitialization
   >
   class MobileNetV1{
    public:
     MobileNetV1();
   
     MobileNetV1(const size_t inputChannel,
                 const size_t inputWidth,
                 const size_t inputHeight,
                 const float alpha = 1.0,
                 const size_t depthMultiplier = 1.0,
                 const bool includeTop = true,
                 const bool preTrained = false,
                 const size_t numClasses = 1000);
   
     MobileNetV1(std::tuple<size_t, size_t, size_t> inputShape,
                 const float alpha = 1.0,
                 const size_t depthMultiplier = 1.0,
                 const bool includeTop = true,
                 const bool preTrained = false,
                 const size_t numClasses = 1000);
   
     ann::FFN<OutputLayerType, InitializationRuleType>&
         GetModel() { return mobileNet; }
   
     void LoadModel(const std::string& filePath);
   
     void SaveModel(const std::string& filepath);
   
    private:
     void ReLU6Layer(ann::Sequential<>* baseLayer = NULL)
     {
       if (baseLayer != NULL)
       {
         baseLayer->Add(new ann::ReLU6<>);
         mlpack::Log::Info << "RelU6" << std::endl;
         return;
       }
   
       mobileNet.Add(new ann::ReLU6<>);
       mlpack::Log::Info << "RelU6" << std::endl;
     }
   
     template<typename SequentialType = ann::Sequential<>>
     void ConvolutionBlock(const size_t inSize,
                           const size_t outSize,
                           const size_t kernelWidth = 1,
                           const size_t kernelHeight = 1,
                           const size_t strideWidth = 1,
                           const size_t strideHeight = 1,
                           const size_t padL = 0,
                           const size_t padR = 0,
                           const size_t padT = 0,
                           const size_t padB = 0,
                           const std::string paddingType = "None",
                           SequentialType* baseLayer = NULL)
     {
       ann::Sequential<>* sequentialBlock = new ann::Sequential<>();
       sequentialBlock->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
           kernelHeight, strideWidth, strideHeight, std::make_tuple(padL, padR),
           std::make_tuple(padT, padB), inputWidth, inputHeight, paddingType));
   
       mlpack::Log::Info << "Convolution: " << "(" << inSize << ", " << inputWidth
           << ", " << inputHeight << ")" << " ---> (";
   
       if (paddingType != "same")
       {
         // Updating input dimesntions.
         inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padR);
         inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padB);
       }
   
       mlpack::Log::Info << outSize << ", " << inputWidth << ", " << inputHeight
           << ")" << std::endl;
   
       if (baseLayer != NULL)
       {
         baseLayer->Add(sequentialBlock);
         return;
       }
   
       mobileNet.Add(sequentialBlock);
     }
   
     size_t DepthWiseConvBlock(const size_t inSize,
                               const size_t outSize,
                               const float alpha,
                               const size_t depthMultiplier,
                               const size_t stride = 1)
   {
       paddingType = "same";
       size_t pointwiseOutSize = size_t(outSize * alpha);
       size_t depthMultipliedOutSize = size_t(inSize * depthMultiplier);
       ann::Sequential<>* sequentialBlock = new ann::Sequential<>();
   
       if (stride != 1)
       {
         sequentialBlock->Add(new ann::Padding<>(0, 1, 0, 1, inputWidth,
             inputHeight));
         mlpack::Log::Info << "Padding: " << "(" << inSize << ", " << inputWidth
             << ", " << inputWidth << " ---> (";
         inputWidth += 1;
         inputHeight += 1;
         mlpack::Log::Info << inSize << ", " << inputWidth << ", " << inputHeight
             << ")" << std::endl;
         paddingType = "valid";
       }
   
       sequentialBlock->Add(new ann::SeparableConvolution<>(inSize,
           depthMultipliedOutSize, 3, 3, stride, stride, 0, 0, inputWidth,
           inputHeight, inSize, paddingType));
       mlpack::Log::Info << "Separable convolution: " << "(" << inSize << ", " <<
           inputWidth << ", " << inputHeight << ")" << " ---> (";
   
       if (paddingType == "valid")
       {
         inputWidth = ConvOutSize(inputWidth, 3, stride, 0);
         inputHeight = ConvOutSize(inputHeight, 3, stride, 0);
       }
   
       mlpack::Log::Info << depthMultipliedOutSize << ", " << inputWidth << ", "
           << inputHeight << ")" << std::endl;
   
       sequentialBlock->Add(new ann::BatchNorm<>(depthMultipliedOutSize, 1e-3,
           true));
       mlpack::Log::Info << "BatchNorm: " << "(" << depthMultipliedOutSize << ")"
           << " ---> (" << depthMultipliedOutSize << ")" << std::endl;
       ReLU6Layer(sequentialBlock);
       ConvolutionBlock(depthMultipliedOutSize, pointwiseOutSize, 1, 1, 1, 1, 0,
           0, 0, 0, "same", sequentialBlock);
       sequentialBlock->Add(new ann::BatchNorm<>(pointwiseOutSize, 1e-3, true));
       mlpack::Log::Info << "BatchNorm: " << "(" << pointwiseOutSize << ")"
           << " ---> (" << pointwiseOutSize << ")" << std::endl;
       ReLU6Layer(sequentialBlock);
       mobileNet.Add(sequentialBlock);
   
       return pointwiseOutSize;
   }
   
     size_t ConvOutSize(const size_t size,
                        const size_t k,
                        const size_t s,
                        const size_t padding)
     {
       return std::floor((size - k + 2 * padding) / s) + 1;
     }
   
     ann::FFN<OutputLayerType, InitializationRuleType> mobileNet;
   
     size_t inputChannel;
   
     size_t inputWidth;
   
     size_t inputHeight;
   
     float alpha;
   
     float depthMultiplier;
   
     size_t numClasses;
   
     std::string paddingType;
   
     size_t outSize;
   
     std::map<size_t, size_t> mobileNetConfig = {
                                                   {128, 2},
                                                   {256, 2},
                                                   {512, 6},
                                                   {1024, 2},
                                                 };
   
     std::map<double, std::string> alphaToString = {
                                                     {0.25, "0.25"},
                                                     {0.5, "0.5"},
                                                     {0.75, "0.75"},
                                                     {1.0, "1"}
                                                   };
   
     std::map<size_t, std::string> imageSizeToString = {
                                                         {128, "128"},
                                                         {160, "160"},
                                                         {192, "192"},
                                                         {224, "224"}
                                                       };
   
     std::string preTrainedPath;
   }; // MobileNetV1 class
   
   // convenience typedef.
   typedef MobileNetV1<ann::CrossEntropyError<>, ann::RandomInitialization>
       MobilenetV1;
   
   } // namespace models
   } // namespace mlpack
   
   #include "mobilenet_v1_impl.hpp"
   
   #endif
