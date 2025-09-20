#include "utils.hpp"
#include <filesystem>

Logger logger;

// Constants for model configuration
static const int maxBatchSize = 1;
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int INPUT_C = 3;
static const int INPUT_SIZE0 = maxBatchSize * INPUT_C * INPUT_H * INPUT_W;
static const int OUTPUT_SIZE0 = maxBatchSize * 1;
static const int OUTPUT_SIZE1 = maxBatchSize * 300 * 4;
static const int OUTPUT_SIZE2 = maxBatchSize * 300;
static const int OUTPUT_SIZE3 = maxBatchSize * 300;
static const int precision_mode = 16; // Precision modes: fp32 : 32, fp16 : 16

// Input and output tensor names (must match ONNX model)
std::string INPUT_NAME0 = "input";
std::string OUTPUT_NAME0 = "num_dets";
std::string OUTPUT_NAME1 = "det_boxes";
std::string OUTPUT_NAME2 = "det_scores";
std::string OUTPUT_NAME3 = "det_classes";

// Paths for engine and model files
std::string engine_dir_path = "../../engine";
std::string onnx_file_path = "../../onnx/yolox-s_640x640_sim_w_nms.onnx"; // ONNX model file path
bool serialize = false;                               // Force serialization flag (true to recreate the engine file)
uint64_t iter_count = 1000;                          // Number of test iterations

int main()
{
    initLibNvInferPlugins(&logger, "");

    // Construct the engine file path
    std::filesystem::path p0(onnx_file_path);
    std::string onnx_file_name = p0.stem().string();
    std::string engine_file_path = engine_dir_path + "/" + onnx_file_name + "_fp" + std::to_string(precision_mode) + "_cpp.engine";
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
        if (!parser->parseFromFile(onnx_file_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING)))
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
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
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

    // Allocate GPU memory space for input and output
    void *inputBuffer0;
    void *outputBuffer0;
    void *outputBuffer1;
    void *outputBuffer2;
    void *outputBuffer3;
    CHECK(cudaMalloc(&inputBuffer0, INPUT_SIZE0 * sizeof(float)));
    CHECK(cudaMalloc(&outputBuffer0, OUTPUT_SIZE0 * sizeof(int)));
    CHECK(cudaMalloc(&outputBuffer1, OUTPUT_SIZE1 * sizeof(float)));
    CHECK(cudaMalloc(&outputBuffer2, OUTPUT_SIZE2 * sizeof(float)));
    CHECK(cudaMalloc(&outputBuffer3, OUTPUT_SIZE3 * sizeof(int)));

    context->setTensorAddress(INPUT_NAME0.c_str(), inputBuffer0);
    context->setTensorAddress(OUTPUT_NAME0.c_str(), outputBuffer0);
    context->setTensorAddress(OUTPUT_NAME1.c_str(), outputBuffer1);
    context->setTensorAddress(OUTPUT_NAME2.c_str(), outputBuffer2);
    context->setTensorAddress(OUTPUT_NAME3.c_str(), outputBuffer3);

    // 4) Prepare input data
    std::cout << "=> 4) Prepare input data\n";

    // Directory path for images
    std::string folderPath = "../../data";
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

    cv::Mat img;
    std::vector<uint8_t> input_i8(INPUT_SIZE0);
    std::vector<float> input(INPUT_SIZE0);
    std::vector<int> outputs0(OUTPUT_SIZE0);
    std::vector<float> outputs1(OUTPUT_SIZE1);
    std::vector<float> outputs2(OUTPUT_SIZE2);
    std::vector<int> outputs3(OUTPUT_SIZE3);

    // Load the first sample input image
    std::string sample_input_path = folderPath + "/" + file_names[2];
    std::cout << "==> Sample input data path : " << sample_input_path << std::endl;
    img = cv::imread(sample_input_path);

    // preprocess input images
    float ratio = std::min(static_cast<float>(INPUT_W) / img.cols, static_cast<float>(INPUT_H) / img.rows);
    // resize
    cv::Mat resized_img(static_cast<int>(ratio * img.rows), static_cast<int>(ratio * img.cols), CV_8UC3);
    cv::resize(img, resized_img, resized_img.size());
    // pad
    cv::Mat padded_img(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized_img.copyTo(padded_img(cv::Rect(0, 0, resized_img.cols, resized_img.rows)));
    
    memcpy(input_i8.data(), padded_img.data, INPUT_SIZE0); // mat -> 1d vector
    Preprocess(input, input_i8, maxBatchSize, INPUT_C, INPUT_H, INPUT_W); // int8 BGR[NHWC](0, 255) -> float BGR[NCHW](0, 255)
    
    // 5) Inference
    std::cout << "=> 5) Inference\n";
    uint64_t dur_time = 0;

    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Warm-up
    for (uint64_t i = 0; i < 30; i++)
    {
        CHECK(cudaMemcpyAsync(inputBuffer0, input.data(), INPUT_SIZE0 * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueueV3(stream);
        CHECK(cudaMemcpyAsync(outputs0.data(), outputBuffer0, OUTPUT_SIZE0 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(outputs1.data(), outputBuffer1, OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(outputs2.data(), outputBuffer2, OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(outputs3.data(), outputBuffer3, OUTPUT_SIZE3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);

    for (uint64_t i = 0; i < iter_count; i++)
    {
        auto start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // Copy input data to GPU
        CHECK(cudaMemcpyAsync(inputBuffer0, input.data(), INPUT_SIZE0 * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueueV3(stream); // Execute the inference
        CHECK(cudaMemcpyAsync(outputs0.data(), outputBuffer0, OUTPUT_SIZE0 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(outputs1.data(), outputBuffer1, OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(outputs2.data(), outputBuffer2, OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(outputs3.data(), outputBuffer3, OUTPUT_SIZE3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream); // Wait for the operation to finish

        // Calculate duration for this iteration
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
        dur_time += dur;
    }

    dur_time /= 1000.f; // Convert from microseconds to milliseconds

    // 6) Print Results
    std::cout << "=> 6) Print Results\n";
    std::cout << "==================================================\n";
    std::cout << "Model : " << onnx_file_name << ", Precision : " << precision_mode << "\n";
    std::cout << iter_count << " iterations completed.\n";
    std::cout << "Total duration time (with data transfer): " << dur_time << " [milliseconds]\n";
    std::cout << "Avg duration time (with data transfer): " << dur_time / static_cast<float>(iter_count) << " [milliseconds]\n";
    std::cout << "FPS : " << 1000.f / (dur_time / static_cast<float>(iter_count)) << " [frame/sec]\n";

    // Find and display the predicted class
    std::filesystem::path p(sample_input_path);
    std::string img_name = p.stem().string();
    int img_width = static_cast<int>(img.cols);
    int img_hight = static_cast<int>(img.rows);
    int x, y, x1, y1, w, h, cls_id;
    float conf;
    float conf_thre = 0.45;
    for (int g_idx = 0; g_idx < outputs0[0]; g_idx++)
    {
        conf = outputs2[g_idx];
        if (conf < conf_thre) continue;
        x = static_cast<int>(outputs1[g_idx * 4 + 0]/ratio);
        y = static_cast<int>(outputs1[g_idx * 4 + 1]/ratio);
        x1 = static_cast<int>(outputs1[g_idx * 4 + 2]/ratio);
        y1 = static_cast<int>(outputs1[g_idx * 4 + 3]/ratio);
        cls_id = outputs3[g_idx];
        w = x1 - x;
        h = y1 - y;

        // bbox
        cv::Rect rect(x, y, w, h);
        auto color_type = color_table[cls_id % color_table.size()];
        auto color = cv::Scalar(color_type[0], color_type[1], color_type[2]);
        rectangle(img, rect, color, 2, 8, 0);

        // text box
        std::string text = std::to_string(g_idx) + " " + defect_names[cls_id] + " " + std::to_string(conf);
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        int text_box_x = (x >= (img_width - 250)) ? x - 250 : x;
        int text_box_y = (y < 50) ? y + 50 : y;
        cv::Rect text_box(text_box_x, text_box_y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(img, text_box, color, cv::FILLED);
        cv::putText(img, text, cv::Point(text_box_x, text_box_y - 3), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        // text_print(img_name, g_idx, x, y, w, h, cls_id, conf, defect_names[cls_id]);

        std::string consol_text = "[" + std::to_string(g_idx) + "] " + img_name + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + ", " +
        std::to_string(cls_id) + ", " + std::to_string(conf) + ", " + defect_names[cls_id];
        std::cout << consol_text << std::endl;
    }

    // show
    cv::resize(img, img, cv::Size(static_cast<int>(img_width), static_cast<int>(img_hight)));
    cv::namedWindow(img_name);
    cv::moveWindow(img_name, 30, 30);
    cv::imshow(img_name, img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    std::string save_dir_path = "../../results";
    gen_dir(save_dir_path);
    std::string save_file_path = save_dir_path + "/" + img_name + "_trt_cpp.jpg";
    std::cout << save_file_path << std::endl;
    cv::imwrite(save_file_path, img); 

    if (!cv::imwrite(save_file_path, img)) {
        std::cerr << "Failed to save file" << std::endl;
        return -1;
    }

    // Release resources
    cudaStreamDestroy(stream);
    CHECK(cudaFree(inputBuffer0));
    CHECK(cudaFree(outputBuffer0));
    CHECK(cudaFree(outputBuffer1));
    CHECK(cudaFree(outputBuffer2));
    CHECK(cudaFree(outputBuffer3));
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
