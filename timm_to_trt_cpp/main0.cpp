#include "utils.hpp"
Logger logger;

static const int maxBatchSize = 1;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 1000;
static const int precision_mode = 16;                 // fp32 : 32, fp16 : 16, int8(ptq) : 8
std::string INPUT_NAME = "input";                     // use same input name with onnx model
std::string OUTPUT_NAME = "output";                   // use same output name with onnx model
std::string engine_dir_path = "../engine";            // Engine directory path
std::string engineFileName = "resnet18";              // model name
std::string onnx_file = "../onnx/resnet18_cuda.onnx"; // onnx model file path
bool serialize = false;                               // force serialize flag (IF true, recreate the engine file unconditionally)
uint64_t iter_count = 10000;                          // the number of test iterations

int main()
{
    std::string engine_file_path = engine_dir_path + "/" + engineFileName + "/" + std::to_string(precision_mode) + ".engine";
    gen_dir(engine_dir_path);
    /*
    /! 1) Create engine file
    /! If force serialize flag is true, recreate unconditionally
    /! If force serialize flag is false, engine file is not created if engine file exist.
    /!                                   create the engine file if engine file doesn't exist.
    */
    bool exist_engine = false;
    if ((access(engine_file_path.c_str(), 0) != -1))
    {
        exist_engine = true;
    }

    if (!((serialize == false) /*Force Serialize flag*/ && (exist_engine == true) /*Whether the engine file exists*/))
    {
        std::cout << "===== Create Engine file =====\n";
        // 1. create a builder
        IBuilder *builder = createInferBuilder(logger);
        if (!builder)
        {
            std::string msg("failed to make builder");
            logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

        // 2. create a network
        std::cout << "==== model build start ====\n";
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
        if (!network)
        {
            std::string msg("failed to make network");
            logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

        // 3. create an ONNX parser
        auto parser = nvonnxparser::createParser(*network, logger);
        if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING)))
        {
            std::string msg("failed to parse onnx file");
            logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }

        // 4. create a build configuratio
        IBuilderConfig *config = builder->createBuilderConfig();
        if (!config)
        {
            std::string msg("failed to make config");
            logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
        config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);

        if (precision_mode == 16)
        {
            std::cout << "==== precision f16 ====\n";
            config->setFlag(BuilderFlag::kFP16);
        }
        else if (precision_mode == 8)
        {
            std::cout << "==== precision int8 ====\n";
            std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
            assert(builder->platformHasFastInt8());
        }
        else
        {
            std::cout << "==== precision f32 ====\n";
        }

        // Build engine
        std::cout << "Building engine, please wait for a while..." << std::endl;
        IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);
        if (!serializedModel)
        {
            std::string msg("failed to make engine");
            logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }
        std::cout << "==== model build done ====\n";
        std::cout << "==== model selialize start ====\n";
        std::ofstream p(engine_file_path, std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file\n";
        }
        p.write(reinterpret_cast<const char *>(serializedModel->data()), serializedModel->size());
        std::cout << "==== model selialize done ====\n";
        p.close();

        delete parser;
        delete network;
        delete config;
        delete builder;
        delete serializedModel;

        std::cout << "===== Create Engine file =====\n";
    }

    // 2) load engine file
    char *modelData{nullptr};
    size_t modelSize{0};
    std::cout << "===== Engine file load =====\n";
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
        std::cout << "[ERROR] Engine file load error\n";
    }

    // 3) deserialize TensorRT Engine from file
    std::cout << "===== Engine file deserialize =====\n";
    IRuntime *runtime = createInferRuntime(logger);
    ICudaEngine *engine = runtime->deserializeCudaEngine(modelData, modelSize);
    IExecutionContext *context = engine->createExecutionContext();
    delete[] modelData;

    void *inputBuffer;
    void *outputBuffer;

    // Allocate GPU memory space for input and output
    CHECK(cudaMalloc(&inputBuffer, maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&outputBuffer, maxBatchSize * OUTPUT_SIZE * sizeof(float)));

    context->setTensorAddress(INPUT_NAME.c_str(), inputBuffer);
    context->setTensorAddress(OUTPUT_NAME.c_str(), outputBuffer);

    // 4) prepare input data
    std::string img_dir = "../../data/";
    std::vector<std::string> file_names;
    // 읽을 디렉토리 경로
    std::string folderPath = "../data"; // 현재 디렉토리를 나타냄

    // 디렉토리 포인터
    DIR *dir = opendir(folderPath.c_str());
    if (dir == nullptr)
    {
        std::cerr << "디렉토리를 열 수 없습니다: " << folderPath << std::endl;
        return 1;
    }

    // 디렉토리 엔트리 포인터
    struct dirent *entry;

    // 디렉토리 내용을 읽음
    std::cout << "파일 리스트:" << std::endl;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (isImageFile(filename))
        {
            std::cout << filename << std::endl;
            file_names.push_back(filename);
        }
    }

    // 디렉토리 닫기
    closedir(dir);

    std::string sample_input_path = "../data/" + file_names[0];
    std::cout << sample_input_path << std::endl;

    cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
    cv::Mat ori_img;
    std::vector<uint8_t> input_i8(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    std::vector<float> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    std::vector<float> outputs(OUTPUT_SIZE);
    for (int idx = 0; idx < maxBatchSize; idx++)
    { // mat -> vector<uint8_t>
        cv::Mat ori_img = cv::imread(sample_input_path);
        // cv::resize(ori_img, img, img.size());
        memcpy(input_i8.data(), ori_img.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
        Preprocess(input, input_i8, maxBatchSize, INPUT_C, INPUT_H, INPUT_W);
    }
    std::cout << "===== input load done =====\n";

    uint64_t dur_time = 0;

    // CUDA stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Warm-up
    CHECK(cudaMemcpyAsync(inputBuffer, input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV3(stream);
    CHECK(cudaMemcpyAsync(outputs.data(), outputBuffer, maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // 5) Inference
    for (uint64_t i = 0; i < iter_count; i++)
    {
        auto start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        CHECK(cudaMemcpyAsync(inputBuffer, input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueueV3(stream);
        CHECK(cudaMemcpyAsync(outputs.data(), outputBuffer, maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
        dur_time += dur;
        // std::cout << dur << " milliseconds" << std::endl;
    }
    dur_time /= 1000.f; // microseconds -> milliseconds

    // 6) Print Results
    std::cout << "==================================================" << std::endl;
    std::cout << "Model : " << engineFileName << ", Precision : " << precision_mode << std::endl;
    std::cout << iter_count << " th Iteration" << std::endl;
    std::cout << "Total duration time with data transfer : " << dur_time << " [milliseconds]" << std::endl;
    std::cout << "Avg duration time with data transfer : " << dur_time / (float)iter_count << " [milliseconds]" << std::endl;
    std::cout << "FPS : " << 1000.f / (dur_time / (float)iter_count) << " [frame/sec]" << std::endl;
    int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
    std::cout << "Index : " << max_index << ", Probability : " << outputs[max_index] << std::endl;
    std::cout << "Class Name : " << class_names[max_index] << std::endl;
    std::cout << "==================================================" << std::endl;

    // Release stream and buffers ...
    cudaStreamDestroy(stream);
    CHECK(cudaFree(inputBuffer));
    CHECK(cudaFree(outputBuffer));
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
