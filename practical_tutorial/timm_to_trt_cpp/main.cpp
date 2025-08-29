#include "utils.hpp"

Logger logger;

// Constants for model configuration
static const int maxBatchSize = 1;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 1000;
static const int precision_mode = 16; // Precision modes: fp32 : 32, fp16 : 16

// Input and output tensor names (must match ONNX model)
std::string INPUT_NAME = "input";
std::string OUTPUT_NAME = "output";

// Paths for engine and model files
std::string engine_dir_path = "../engine";
std::string engineFileName = "resnet18";              // Model name
std::string onnx_file = "../onnx/resnet18_cuda.onnx"; // ONNX model file path
bool serialize = false;                               // Force serialization flag (true to recreate the engine file)
uint64_t iter_count = 10000;                          // Number of test iterations

int main()
{
    // Construct the engine file path
    std::string engine_file_path = engine_dir_path + "/" + engineFileName + "_" + std::to_string(precision_mode) + ".engine";
    std::cout << engine_file_path << std::endl;

    // Generate directory for engine files if it doesn't exist
    gen_dir(engine_dir_path);

    // Check if the engine file already exists
    bool exist_engine = (access(engine_file_path.c_str(), 0) != -1);

    // Create the engine file if it doesn't exist or if forced to serialize
    if (!(serialize == false && exist_engine == true))
    {
        std::cout << "=> 1) Create the engine file\n";

        // 1. Create a builder
        std::cout << "==> 1. Create a builder\n";
        IBuilder *builder = createInferBuilder(logger);
        if (!builder)
        {
            logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create builder");
            exit(EXIT_FAILURE);
        }

        // 2. Create a network
        std::cout << "==> 2. Create a network\n";
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
        if (!network)
        {
            logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create network");
            exit(EXIT_FAILURE);
        }

        // 3. Create an ONNX parser
        std::cout << "==> 3. Create an ONNX parser\n";
        auto parser = nvonnxparser::createParser(*network, logger);
        if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING)))
        {
            logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to parse ONNX file");
            exit(EXIT_FAILURE);
        }

        // Print parsing errors if any
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }

        // 4. Create a builder configuration
        std::cout << "==> 4. Create a builder configuration\n";
        IBuilderConfig *config = builder->createBuilderConfig();
        if (!config)
        {
            logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create configuration");
            exit(EXIT_FAILURE);
        }

        // Set memory pool limits
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
        config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);

        // Set precision mode
        if (precision_mode == 16)
        {
            std::cout << "==> Set precision: FP16\n";
            config->setFlag(BuilderFlag::kFP16);
        }
        else if (precision_mode == 32)
        {
            std::cout << "==> Set precision: FP32\n";
        }
        else
        {
            std::cout << "[TRT_WARNING] Wrong precision model value is entered(automatically set to FP32)\n";
        }

        // 5. Build engine
        std::cout << "==> 5. Build engine(please wait...)\n";
        IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);
        if (!serializedModel)
        {
            logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create engine");
            exit(EXIT_FAILURE);
        }

        // 6. Save the serialized model to a file
        std::cout << "==> 6. Save the serialized model to a file\n";
        std::ofstream p(engine_file_path, std::ios::binary);
        if (!p)
        {
            std::cerr << "[TRT_ERROR] Could not open plan output file\n";
            exit(EXIT_FAILURE);
        }
        p.write(reinterpret_cast<const char *>(serializedModel->data()), serializedModel->size());
        p.close();

        // Clean up resources
        delete parser;
        delete network;
        delete config;
        delete builder;
        delete serializedModel;
        std::cout << "==> 7. Engine file creation complete\n";
    }
    else
    {
        std::cout << "=> 1) the engine file already exists\n";
    }

    // 2) Load the engine file
    std::cout << "=> 2) Load the TensorRT engine file\n";

    char *modelData{nullptr};
    size_t modelSize{0};
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        modelSize = file.tellg();
        file.seekg(0, file.beg);
        modelData = new char[modelSize];
        file.read(modelData, modelSize);
        file.close();
    }
    else
    {
        std::cerr << "[TRT_ERROR] Engine file loading error\n";
        exit(EXIT_FAILURE);
    }

    // 3) Deserialize TensorRT engine from file
    std::cout << "=> 3) Deserialize TensorRT engine from file\n";

    IRuntime *runtime = createInferRuntime(logger);
    ICudaEngine *engine = runtime->deserializeCudaEngine(modelData, modelSize);
    IExecutionContext *context = engine->createExecutionContext();
    delete[] modelData; // Free model data after deserialization

    void *inputBuffer;
    void *outputBuffer;

    // Allocate GPU memory for input and output
    CHECK(cudaMalloc(&inputBuffer, maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&outputBuffer, maxBatchSize * OUTPUT_SIZE * sizeof(float)));

    // Set tensor addresses for input and output
    context->setTensorAddress(INPUT_NAME.c_str(), inputBuffer);
    context->setTensorAddress(OUTPUT_NAME.c_str(), outputBuffer);

    // 4) Prepare input data
    std::cout << "=> 4) Prepare input data\n";

    // Directory path for images
    std::string folderPath = "../data";
    std::vector<std::string> file_names;

    // Open directory pointer
    DIR *dir = opendir(folderPath.c_str());
    if (dir == nullptr)
    {
        std::cerr << "[TRT_ERROR] Cannot open directory: " << folderPath << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read directory entries
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (isImageFile(filename))
        {
            // std::cout << filename << std::endl;
            file_names.push_back(filename);
        }
    }

    // Close the directory
    closedir(dir);

    // Load the first sample input image
    std::string sample_input_path = "../data/" + file_names[0];
    std::cout << "==> Sample input data path : " << sample_input_path << std::endl;

    cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
    cv::Mat ori_img;
    std::vector<uint8_t> input_i8(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    std::vector<float> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    std::vector<float> outputs(OUTPUT_SIZE);

    // Read and preprocess input images
    for (int idx = 0; idx < maxBatchSize; idx++)
    {
        ori_img = cv::imread(sample_input_path);
        // Resize the image if needed
        // cv::resize(ori_img, img, img.size());
        memcpy(input_i8.data(), ori_img.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C); // mat -> 1d vector
        Preprocess(input, input_i8, maxBatchSize, INPUT_C, INPUT_H, INPUT_W);              // int8 BGR[NHWC](0, 255) -> float RGB[NCHW](0.f,1.f)
    }

    // 5) Inference
    std::cout << "=> 5) Inference\n";
    uint64_t dur_time = 0;

    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Warm-up
    CHECK(cudaMemcpyAsync(inputBuffer, input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV3(stream);
    CHECK(cudaMemcpyAsync(outputs.data(), outputBuffer, maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    for (uint64_t i = 0; i < iter_count; i++)
    {
        auto start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // Copy input data to GPU
        CHECK(cudaMemcpyAsync(inputBuffer, input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueueV3(stream); // Execute the inference
        CHECK(cudaMemcpyAsync(outputs.data(), outputBuffer, maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream); // Wait for the operation to finish

        // Calculate duration for this iteration
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
        dur_time += dur;
    }

    dur_time /= 1000.f; // Convert from microseconds to milliseconds

    // 6) Print Results
    std::cout << "=> 6) Print Results\n";
    std::cout << "==================================================\n";
    std::cout << "Model : " << engineFileName << ", Precision : " << precision_mode << "\n";
    std::cout << iter_count << " iterations completed.\n";
    std::cout << "Total duration time (with data transfer): " << dur_time << " [milliseconds]\n";
    std::cout << "Avg duration time (with data transfer): " << dur_time / (float)iter_count << " [milliseconds]\n";
    std::cout << "FPS : " << 1000.f / (dur_time / (float)iter_count) << " [frame/sec]\n";

    // Find and display the predicted class
    int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
    std::cout << "Index : " << max_index << ", Probability : " << outputs[max_index] << "\n";
    std::cout << "Class Name : " << class_names[max_index] << "\n";
    std::cout << "==================================================\n";

    // Release resources
    cudaStreamDestroy(stream);
    CHECK(cudaFree(inputBuffer));
    CHECK(cudaFree(outputBuffer));
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
