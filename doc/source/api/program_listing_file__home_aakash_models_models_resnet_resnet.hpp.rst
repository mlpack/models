
.. _program_listing_file__home_aakash_models_models_resnet_resnet.hpp:

Program Listing for File resnet.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_models_models_resnet_resnet.hpp>` (``/home/aakash/models/models/resnet/resnet.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

    author = {Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun},
    title = {Deep Residual Learning for Image Recognition},
    year = {2015},
    url = {https://arxiv.org/pdf/1512.03385.pdf}
   }
   
   #ifndef MODELS_MODELS_RESNET_RESNET_HPP
   #define MODELS_MODELS_RESNET_RESNET_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/init_rules/random_init.hpp>
   #include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>
   #include <mlpack/methods/ann/init_rules/he_init.hpp>
   
   namespace mlpack {
   namespace models {
   
   template<
     typename OutputLayerType = ann::CrossEntropyError<>,
     typename InitializationRuleType = ann::RandomInitialization,
     size_t ResNetVersion = 18
   >
   class ResNet{
    public:
     ResNet();
   
     ResNet(const size_t inputChannel,
            const size_t inputWidth,
            const size_t inputHeight,
            const bool includeTop = true,
            const bool preTrained = false,
            const size_t numClasses = 1000);
   
     ResNet(std::tuple<size_t, size_t, size_t> inputShape,
            const bool includeTop = true,
            const bool preTrained = false,
            const size_t numClasses = 1000);
   
     ann::FFN<OutputLayerType, InitializationRuleType>& GetModel()
     {
       return resNet;
     }
   
     void LoadModel(const std::string& filePath);
   
     void SaveModel(const std::string& filepath);
   
    private:
     template<typename SequentialType = ann::Sequential<>>
     void ConvolutionBlock3x3(SequentialType* baseLayer,
                              const size_t inSize,
                              const size_t outSize,
                              const size_t strideWidth = 1,
                              const size_t strideHeight = 1,
                              const size_t kernelWidth = 3,
                              const size_t kernelHeight = 3,
                              const size_t padW = 1,
                              const size_t padH = 1)
     {
       baseLayer->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
           kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
           inputHeight));
   
       mlpack::Log::Info << "Convolution: " << "(" << inSize << ", " << inputWidth
           << ", " << inputHeight << ")" << " ---> (";
   
       // Updating input dimesntions.
       inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
       inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
   
       mlpack::Log::Info << outSize << ", " << inputWidth << ", " << inputHeight
           << ")" << std::endl;
     }
   
     template<typename SequentialType = ann::Sequential<>>
     void ConvolutionBlock1x1(SequentialType* baseLayer,
                              const size_t inSize,
                              const size_t outSize,
                              const size_t downSampleInputWidth = 0,
                              const size_t downSampleInputHeight = 0,
                              const size_t strideWidth = 1,
                              const size_t strideHeight = 1,
                              const size_t kernelWidth = 1,
                              const size_t kernelHeight = 1,
                              const size_t padW = 0,
                              const size_t padH = 0,
                              const bool downSample = false)
     {
       if (downSample)
       {
         baseLayer->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
             kernelHeight, strideWidth, strideHeight, padW, padH,
             downSampleInputWidth, downSampleInputHeight));
   
         mlpack::Log::Info << "  Convolution: " << "(" << inSize << ", " <<
             downSampleInputWidth << ", " << downSampleInputHeight << ")" <<
             " ---> (" << outSize << ", " << downSampleInputWidth << ", " <<
             downSampleInputHeight << ")" << std::endl;
       }
       else
       {
         baseLayer->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
             kernelHeight, strideWidth, strideHeight, padW, padH,
             inputWidth, inputHeight));
   
         mlpack::Log::Info << "Convolution: " << "(" << inSize << ", " <<
             inputWidth << ", " << inputHeight << ")" << " ---> (";
   
         // Updating input dimesntions.
         inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
         inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
   
         mlpack::Log::Info << outSize << ", " << inputWidth << ", " << inputHeight
             << ")" << std::endl;
       }
     }
   
     template <typename AddMergeType = ann::AddMerge<>>
     void DownSample(AddMergeType* resBlock,
                     const size_t inSize,
                     const size_t outSize,
                     const size_t downSampleInputWidth,
                     const size_t downSampleInputHeight,
                     const size_t kernelWidth = 1,
                     const size_t kernelHeight = 1,
                     const size_t strideWidth = 2,
                     const size_t strideHeight = 2,
                     const size_t padW = 0,
                     const size_t padH = 0)
     {
       ann::Sequential<>* downSampleBlock = new ann::Sequential<>();
       ConvolutionBlock1x1(downSampleBlock, inSize, outSize, downSampleInputWidth,
           downSampleInputHeight, strideWidth, strideHeight, kernelWidth,
           kernelHeight, padW, padH, true);
   
       downSampleBlock->Add(new ann::BatchNorm<>(outSize, 1e-5));
       mlpack::Log::Info << "  BatchNorm: " << "(" << outSize << ")" << " ---> ("
           << outSize << ")" << std::endl;
       resBlock->Add(downSampleBlock);
     }
   
     void BasicBlock(const size_t inSize,
                     const size_t outSize,
                     const size_t strideWidth = 1,
                     const size_t strideHeight = 1,
                     const bool downSample = false)
     {
       downSampleInputWidth = inputWidth;
       downSampleInputHeight = inputHeight;
   
       ann::Sequential<>* basicBlock = new ann::Sequential<>();
       ann::AddMerge<>* resBlock = new ann::AddMerge<>(true, true);
       ann::Sequential<>* sequentialBlock = new ann::Sequential<>();
       ConvolutionBlock3x3(sequentialBlock, inSize, outSize, strideWidth,
           strideHeight);
       sequentialBlock->Add(new ann::BatchNorm<>(outSize, 1e-5));
       mlpack::Log::Info << "BatchNorm: " << "(" << outSize << ")" << " ---> ("
           << outSize << ")" << std::endl;
       sequentialBlock->Add(new ann::ReLULayer<>);
       mlpack::Log::Info << "Relu" << std::endl;
       ConvolutionBlock3x3(sequentialBlock, outSize, outSize);
       sequentialBlock->Add(new ann::BatchNorm<>(outSize, 1e-5));
       mlpack::Log::Info << "BatchNorm: " << "(" << outSize << ")" << " ---> ("
           << outSize << ")" << std::endl;
   
       resBlock->Add(sequentialBlock);
   
       if (downSample == true)
       {
         mlpack::Log::Info << "DownSample (" << std::endl;
         DownSample(resBlock, inSize, outSize, downSampleInputWidth,
             downSampleInputHeight);
         mlpack::Log::Info << ")" <<std::endl;
       }
       else
       {
         mlpack::Log::Info << "IdentityLayer" << std::endl;
         resBlock->Add(new ann::IdentityLayer<>);
       }
   
       basicBlock->Add(resBlock);
       basicBlock->Add(new ann::ReLULayer<>);
       mlpack::Log::Info << "Relu" << std::endl;
       resNet.Add(basicBlock);
     }
   
     void BottleNeck(const size_t inSize,
                     const size_t outSize,
                     const size_t strideWidth = 1,
                     const size_t strideHeight = 1,
                     const bool downSample = false,
                     const size_t baseWidth = 64,
                     const size_t groups = 1)
     {
       downSampleInputWidth = inputWidth;
       downSampleInputHeight = inputHeight;
   
       size_t width = int((baseWidth / 64.0) * outSize) * groups;
       ann::Sequential<>* basicBlock = new ann::Sequential<>();
       ann::AddMerge<>* resBlock = new ann::AddMerge<>(true, true);
       ann::Sequential<>* sequentialBlock = new ann::Sequential<>();
       ConvolutionBlock1x1(sequentialBlock, inSize, width);
       sequentialBlock->Add(new ann::BatchNorm<>(width, 1e-5));
       mlpack::Log::Info << "BatchNorm: " << "(" << width << ")" << " ---> ("
           << width << ")" << std::endl;
       sequentialBlock->Add(new ann::ReLULayer<>);
       mlpack::Log::Info << "Relu" << std::endl;
       ConvolutionBlock3x3(sequentialBlock, width, width, strideWidth,
           strideHeight);
       sequentialBlock->Add(new ann::BatchNorm<>(width, 1e-5));
       mlpack::Log::Info << "BatchNorm: " << "(" << width << ")" << " ---> ("
           << width << ")" << std::endl;
       sequentialBlock->Add(new ann::ReLULayer<>);
       mlpack::Log::Info << "Relu" << std::endl;
       ConvolutionBlock1x1(sequentialBlock, width, outSize * bottleNeckExpansion);
       sequentialBlock->Add(new ann::BatchNorm<>(outSize * bottleNeckExpansion,
           1e-5));
       mlpack::Log::Info << "BatchNorm: " << "(" << outSize * bottleNeckExpansion
       << ")" << " ---> (" << outSize * bottleNeckExpansion << ")" << std::endl;
   
       resBlock->Add(sequentialBlock);
   
       if (downSample == true)
       {
         mlpack::Log::Info << "DownSample (" << std::endl;
         DownSample(resBlock, inSize, outSize * bottleNeckExpansion,
             downSampleInputWidth, downSampleInputHeight, 1, 1, strideWidth,
             strideHeight);
         mlpack::Log::Info << ")" << std::endl;
       }
       else
       {
         mlpack::Log::Info << "IdentityLayer" << std::endl;
         resBlock->Add(new ann::IdentityLayer<>);
       }
   
       basicBlock->Add(resBlock);
       basicBlock->Add(new ann::ReLULayer<>);
       mlpack::Log::Info << "Relu" << std::endl;
       resNet.Add(basicBlock);
     }
   
     void MakeLayer(const std::string& block,
                    const size_t outSize,
                    const size_t numBlocks,
                    const size_t stride = 1)
     {
       bool downSample = false;
   
       if (block == "basicblock")
       {
         if (stride != 1 || downSampleInSize != outSize * basicBlockExpansion)
           downSample = true;
         BasicBlock(downSampleInSize, outSize * basicBlockExpansion, stride,
             stride, downSample);
         downSampleInSize = outSize * basicBlockExpansion;
         for (size_t i = 1; i != numBlocks; ++i)
           BasicBlock(downSampleInSize, outSize);
       }
   
       else if (block == "bottleneck")
       {
         if (stride != 1 || downSampleInSize != outSize * bottleNeckExpansion)
           downSample = true;
         BottleNeck(downSampleInSize, outSize, stride, stride, downSample);
         downSampleInSize = outSize * bottleNeckExpansion;
         for (size_t i = 1; i != numBlocks; ++i)
           BottleNeck(downSampleInSize, outSize);
       }
     }
   
     size_t ConvOutSize(const size_t size,
                        const size_t k,
                        const size_t s,
                        const size_t padding)
     {
       return std::floor((size - k + 2 * padding) / s) + 1;
     }
   
     ann::FFN<OutputLayerType, InitializationRuleType> resNet;
   
     size_t inputChannel;
   
     size_t inputWidth;
   
     size_t inputHeight;
   
     size_t numClasses;
   
     size_t downSampleInputWidth;
   
     size_t downSampleInputHeight;
   
     size_t basicBlockExpansion = 1;
   
     size_t bottleNeckExpansion = 4;
   
     size_t downSampleInSize = 64;
   
     std::map<size_t, std::map<std::string, std::array<size_t, 4>>> ResNetConfig =
         {
           {18, {{"basicblock", {2, 2, 2, 2}}}},
           {34, {{"basicblock", {3, 4, 6, 3}}}},
           {50, {{"bottleneck", {3, 4, 6, 3}}}},
           {101, {{"bottleneck", {3, 4, 23, 3}}}},
           {152, {{"bottleneck", {3, 8, 36, 3}}}}
         };
   
     std::array<size_t , 4> numBlockArray;
   
     std::string builderBlock;
   }; // ResNet class
   
   // convenience typedefs for different ResNet models.
   typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 18>
       ResNet18;
   typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 34>
       ResNet34;
   typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 50>
       ResNet50;
   typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 101>
       ResNet101;
   typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 152>
       ResNet152;
   
   } // namespace models
   } // namespace mlpack
   
   #include "resnet_impl.hpp"
   
   #endif
