
.. _program_listing_file__home_aakash_models_dataloader_datasets.hpp:

Program Listing for File datasets.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_models_dataloader_datasets.hpp>` (``/home/aakash/models/dataloader/datasets.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MODELS_DATALOADER_DATASETS_HPP
   #define MODELS_DATALOADER_DATASETS_HPP
   
   #include <dataloader/preprocessor.hpp>
   
   namespace mlpack {
   namespace models {
   
   template<
       typename DatasetX = arma::mat,
       typename DatasetY = arma::mat
   >
   struct DatasetDetails
   {
     std::string datasetName;
   
     std::string trainDownloadURL;
   
     std::string testDownloadURL;
   
     std::string trainHash;
   
     std::string testHash;
   
     std::string datasetType;
   
     std::string trainPath;
   
     std::string testPath;
   
     bool zipFile;
   
     std::string datasetURL;
   
     std::string datasetPath;
   
     std::string datasetHash;
   
     std::string serverName;
   
     // Pre-Process functor.
     std::function<void(DatasetX&, DatasetY&,
         DatasetX&, DatasetY&, DatasetX&)> PreProcess;
   
     // The following parameters are for CSVs only.
     size_t startTrainingInputFeatures;
     size_t endTrainingInputFeatures;
   
     size_t startTrainingPredictionFeatures;
     size_t endTrainingPredictionFeatures;
   
     size_t startTestingInputFeatures;
     size_t endTestingInputFeatures;
   
     bool dropHeader;
   
     // The following data members corresponds to image classification / detection
     // type of datasets.
     std::string trainingImagesPath;
   
     std::string testingImagesPath;
   
     std::string trainingAnnotationPath;
   
     std::vector<std::string> classes;
   
     size_t imageWidth;
   
     size_t imageHeight;
   
     size_t imageDepth;
   
     // Default constructor.
     DatasetDetails() :
         datasetName(""),
         trainDownloadURL(""),
         testDownloadURL(""),
         trainHash(""),
         testHash(""),
         datasetType("none"),
         trainPath(""),
         testPath(""),
         zipFile(false),
         datasetURL(""),
         datasetPath(""),
         datasetHash(""),
         serverName("www.mlpack.org"),
         startTrainingInputFeatures(0),
         endTrainingInputFeatures(0),
         startTrainingPredictionFeatures(0),
         endTrainingPredictionFeatures(0),
         startTestingInputFeatures(0),
         endTestingInputFeatures(0),
         dropHeader(false),
         trainingImagesPath(""),
         testingImagesPath(""),
         trainingAnnotationPath(""),
         classes(std::vector<std::string>()),
         imageWidth(0),
         imageHeight(0),
         imageDepth(0)
     {/* Nothing to do here. */}
   
     DatasetDetails(const std::string& datasetName,
                    const std::string& trainDownloadURL,
                    const std::string& testDownloadURL,
                    const std::string& trainHash,
                    const std::string& testHash,
                    const std::string& datasetType,
                    const std::string& trainPath,
                    const std::string& testPath) :
                    datasetName(datasetName),
                    trainDownloadURL(trainDownloadURL),
                    testDownloadURL(testDownloadURL),
                    trainHash(trainHash),
                    testHash(testHash),
                    datasetType(datasetType),
                    trainPath(trainPath),
                    testPath(testPath),
                    zipFile(false),
                    datasetURL(""),
                    datasetHash(""),
                    serverName("www.mlpack.org"),
                    startTrainingInputFeatures(0),
                    endTrainingInputFeatures(0),
                    startTrainingPredictionFeatures(0),
                    endTrainingPredictionFeatures(0),
                    startTestingInputFeatures(0),
                    endTestingInputFeatures(0),
                    dropHeader(false),
                    trainingImagesPath(""),
                    testingImagesPath(""),
                    trainingAnnotationPath(""),
                    classes(std::vector<std::string>()),
                    imageWidth(0),
                    imageHeight(0),
                    imageDepth(0)
     {
       // Nothing to do here.
     }
   
     DatasetDetails(const std::string& datasetName,
                    const bool zipFile,
                    const std::string& datasetURL,
                    const std::string& datasetPath,
                    const std::string& datasetHash,
                    const std::string& datasetType,
                    const std::string& trainPath = "",
                    const std::string& testPath = "") :
                    datasetName(datasetName),
                    trainDownloadURL(""),
                    testDownloadURL(""),
                    trainHash(""),
                    testHash(""),
                    datasetType(datasetType),
                    trainPath(trainPath),
                    testPath(testPath),
                    zipFile(zipFile),
                    datasetURL(datasetURL),
                    datasetPath(datasetPath),
                    datasetHash(datasetHash),
                    serverName("www.mlpack.org"),
                    startTrainingInputFeatures(0),
                    endTrainingInputFeatures(0),
                    startTrainingPredictionFeatures(0),
                    endTrainingPredictionFeatures(0),
                    startTestingInputFeatures(0),
                    endTestingInputFeatures(0),
                    dropHeader(false),
                    trainingImagesPath(""),
                    testingImagesPath(""),
                    trainingAnnotationPath(""),
                    classes(std::vector<std::string>()),
                    imageWidth(0),
                    imageHeight(0),
                    imageDepth(0)
     {
       // Nothing to do here.
     }
   };
   
   template<
       typename DatasetX = arma::mat,
       typename DatasetY = arma::mat
   >
   class Datasets
   {
    public:
     const static DatasetDetails<DatasetX, DatasetY> MNIST()
     {
       DatasetDetails<DatasetX, DatasetY> mnistDetails(
           "mnist",
           true,
           "/datasets/mnist.tar.gz",
           "./../data/mnist.tar.gz",
           "33470ca3",
           "csv",
           "./../data/mnist-dataset/mnist_train.csv",
           "./../data/mnist-dataset/mnist_test.csv");
   
       // Set the Pre-Processor Function.
       mnistDetails.PreProcess = PreProcessor<DatasetX, DatasetY>::MNIST;
   
       // Set Parameters for CSV file.
       mnistDetails.startTestingInputFeatures = 0;
       mnistDetails.endTestingInputFeatures = -1;
       mnistDetails.startTrainingInputFeatures = 1;
       mnistDetails.endTrainingInputFeatures = -1;
       mnistDetails.startTrainingPredictionFeatures = 0;
       mnistDetails.endTrainingPredictionFeatures = 0;
       mnistDetails.dropHeader = true;
       return mnistDetails;
     }
   
     const static DatasetDetails<DatasetX, DatasetY> VOCDetection()
     {
       DatasetDetails<DatasetX, DatasetY> VOCDetectionDetail(
           "voc-detection",
           true,
           "/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
           "./../data/VOCtrainval_11-May-2012.tar",
           "504b9278",
           "image-detection");
   
       VOCDetectionDetail.trainingImagesPath =
           "./../data/VOCdevkit/VOC2012/JPEGImages/";
       VOCDetectionDetail.trainingAnnotationPath =
         "./../data/VOCdevkit/VOC2012/Annotations/";
       VOCDetectionDetail.serverName = "http://host.robots.ox.ac.uk";
       VOCDetectionDetail.PreProcess = PreProcessor<DatasetX, DatasetY>::PascalVOC;
   
       // Set classes for dataset.
       VOCDetectionDetail.classes = {"background", "aeroplane", "bicycle",
         "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
         "sheep", "sofa", "train", "tvmonitor"};
   
       return VOCDetectionDetail;
     }
   
     const static DatasetDetails<DatasetX, DatasetY> CIFAR10()
     {
       DatasetDetails<DatasetX, DatasetY> CIFAR10Detail(
           "cifar10",
           true,
           "/datasets/cifar10.tar.gz",
           "./../data/cifar10.tar.gz",
           "4cd9757b",
           "image-classification");
   
       CIFAR10Detail.trainingImagesPath = "./../data/cifar10/train/";
       CIFAR10Detail.testingImagesPath = "./../data/cifar10/test/";
   
       CIFAR10Detail.serverName = "www.mlpack.org";
       CIFAR10Detail.PreProcess = PreProcessor<DatasetX, DatasetY>::CIFAR10;
   
       return CIFAR10Detail;
     }
   };
   
   } // namespace models
   } // namespace mlpack
   
   #endif
