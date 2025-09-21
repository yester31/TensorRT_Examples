#include "yolox_opti_trt.hpp"

yolox_opti_trt::yolox_opti_trt(
    int batch_size_, int input_h_, int input_w_, int input_c_, int class_count_, int precision_mode_, bool serialize_, int gpu_device_,
    std::string engine_dir_path_, std::string engine_file_name_, std::string weight_file_path_)
    : base_opti_trt(batch_size_, input_h_, input_w_, input_c_, class_count_, precision_mode_, serialize_, gpu_device_, engine_dir_path_, engine_file_name_, weight_file_path_)
      
{
    initLibNvInferPlugins(&trtLogger, "");

    // 1) generate main logger
    std::cout << ("==================================================================") << std::endl;
    std::cout << ("      Starting the work to accelerate inference of the YOLOX      ") << std::endl;
    std::cout << ("==================================================================") << std::endl;

    // 3) check gpu device
    check_device();

    // 4) set gpu device
    set_device(gpu_device);

    // 5) create CUDA stream & trt default logger
    CHECK(cudaStreamCreate(&stream));

    // 6) define engine file name & check engine directory
    engine_file_path = engine_dir_path + "/" + engine_file_name + ".engine";

    if (std::filesystem::is_directory(engine_dir_path))
    {
        trtLogger.log(Logger::Severity::kINFO, ("The folder already exists. : " + engine_dir_path).c_str());
    }
    else
    {
        std::filesystem::create_directories(engine_dir_path);
        trtLogger.log(Logger::Severity::kINFO, ("The folder was created. : " + engine_dir_path).c_str());
    }

    // 7) check engine file
    auto exist_engine = std::filesystem::exists(engine_file_path);
    if (exist_engine)
    {
        trtLogger.log(Logger::Severity::kINFO, ("The engine file exists. : " + engine_file_path).c_str());
    }
    else
    {
        trtLogger.log(Logger::Severity::kWARNING, ("The engine file does not exist. : " + engine_file_path).c_str());
    }

    // 8) check weight file
    auto exist_weight = std::filesystem::exists(weight_file_path);
    if (exist_weight)
    {
        trtLogger.log(Logger::Severity::kINFO, ("The weight file exists. : " + weight_file_path).c_str());
    }
    else
    {
        // In case of no exist both the engine file and the wieght file.
        // In case of no exist the wieght file when the serialize variable is true.
        if (!exist_engine || serialize)
        {
            trtLogger.log(Logger::Severity::kERROR, ("The weight file does not exist. : " + weight_file_path).c_str());
            exit(EXIT_FAILURE);
        }
        else
        {
            trtLogger.log(Logger::Severity::kWARNING, ("The weight file does not exist. : " + weight_file_path).c_str());
        }
    }

    // 9) Create engine file
    // If force serialize flag is true, recreate unconditionally
    // If force serialize flag is false, engine file is not created if engine file exist.
    //                                   create the engine file if engine file doesn't exist.

    if (!((serialize == false) /*Force Serialize flag*/ && (exist_engine == true) /*Whether the resnet18.engine file exists*/))
    {
        trtLogger.log(Logger::Severity::kINFO, "create engine (This process takes from a few minutes to several tens of minutes depending on the device.)");

        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(trtLogger));
        if (!builder)
        {
            trtLogger.log(Logger::Severity::kERROR, "Unable to create builder object.");
            exit(EXIT_FAILURE);
        }

        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config)
        {
            trtLogger.log(Logger::Severity::kERROR, "Unable to create config object.");
            exit(EXIT_FAILURE);
        }

        // *** create tensorrt model using by TensorRT API ***
        createEngineFromOnnx(builder, config);

        trtLogger.log(Logger::Severity::kINFO, "create engine done");
    }

    // 10) load engine file
    std::vector<char> trt_model_stream;
    size_t trt_model_stream_size{ 0 };
    trtLogger.log(Logger::Severity::kINFO, "engine file load");

    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        trt_model_stream_size = file.tellg();
        file.seekg(0, file.beg);
        trt_model_stream.resize(trt_model_stream_size);
        file.read(trt_model_stream.data(), trt_model_stream_size);
        file.close();
    }
    else
    {
        trtLogger.log(Logger::Severity::kERROR, "Engine file load error");
        exit(EXIT_FAILURE);
    }

    // 11) deserialize TensorRT Engine from file
    trtLogger.log(Logger::Severity::kINFO, "engine file deserialize start");

    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtLogger));
    if (!runtime)
    {
        trtLogger.log(Logger::Severity::kERROR, "Unable to create runtime");
        exit(EXIT_FAILURE);
    }

    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trt_model_stream.data(), trt_model_stream_size));
    if (!engine)
    {
        trtLogger.log(Logger::Severity::kERROR, "Unable to build cuda engine.");
        exit(EXIT_FAILURE);
    }
    trtLogger.log(Logger::Severity::kINFO, "engine file deserialize done");

    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        trtLogger.log(Logger::Severity::kERROR, "Unable to create context");
        exit(EXIT_FAILURE);
    }

    // 13) Allocate GPU memory space for input and output
    buffers.resize(5);
    INPUT_SIZE0 = input_c * input_h * input_w;
    CHECK(cudaMalloc(&buffers[0], batch_size * INPUT_SIZE0 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], batch_size * OUTPUT_SIZE0 * sizeof(int)));
    CHECK(cudaMalloc(&buffers[2], batch_size * OUTPUT_SIZE1 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[3], batch_size * OUTPUT_SIZE2 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[4], batch_size * OUTPUT_SIZE3 * sizeof(int)));

    context->setTensorAddress(INPUT_NAME0.c_str(), buffers[0]);
    context->setTensorAddress(OUTPUT_NAME0.c_str(), buffers[1]);
    context->setTensorAddress(OUTPUT_NAME1.c_str(), buffers[2]);
    context->setTensorAddress(OUTPUT_NAME2.c_str(), buffers[3]);
    context->setTensorAddress(OUTPUT_NAME3.c_str(), buffers[4]);

    // 12) warm-up
    uint64_t iter_count = 30; // the number of test iterations
    for (size_t  i = 0; i < iter_count; i++)
    {
        context->enqueueV3(stream);
    }
    cudaStreamSynchronize(stream);
    trtLogger.log(Logger::Severity::kINFO, "warmup complete");

    // 13) prepare input & output data
    output_post0.resize(batch_size * OUTPUT_SIZE0);
    output_post1.resize(batch_size * OUTPUT_SIZE1);
    output_post2.resize(batch_size * OUTPUT_SIZE2);
    output_post3.resize(batch_size * OUTPUT_SIZE3);

    trtLogger.log(Logger::Severity::kINFO, "inference preparation complete");
}

yolox_opti_trt::~yolox_opti_trt() {}

// feed input data & run preprocess
void yolox_opti_trt::input_data(const void* inputs)
{
    // cpu -> gpu memory copy
    CHECK(cudaMemcpyAsync(static_cast<float*>(buffers[0]), static_cast<const float*>(inputs), batch_size * INPUT_SIZE0 * sizeof(float), cudaMemcpyHostToDevice, stream));
}

// inference
void yolox_opti_trt::run_model()
{
    context->enqueueV3(stream);
}

// get output result
void yolox_opti_trt::output_data(void* outputs)
{
    CHECK(cudaMemcpyAsync(output_post0.data(), buffers[1], batch_size * OUTPUT_SIZE0 * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_post1.data(), buffers[2], batch_size * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_post2.data(), buffers[3], batch_size * OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_post3.data(), buffers[4], batch_size * OUTPUT_SIZE3 * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // b * (1 + 300 * 6)
    for (int b_idx = 0; b_idx < batch_size; b_idx++)
    {
        ((float*)outputs)[b_idx * (1 + (OUTPUT_SIZE1+OUTPUT_SIZE2+OUTPUT_SIZE3))] = static_cast<float>(output_post0[b_idx]); // num_dets
        for (int d_idx = 0; d_idx < output_post0[b_idx]; d_idx++){
            int box_offset = b_idx * OUTPUT_SIZE1 + d_idx * 4;
            int score_offset = b_idx * OUTPUT_SIZE2 + d_idx;   
            int class_offset = b_idx * OUTPUT_SIZE3 + d_idx;
            int output_offset = b_idx * (1 + (OUTPUT_SIZE1+OUTPUT_SIZE2+OUTPUT_SIZE3)) + 1 + d_idx * 6;
            ((float*)outputs)[output_offset + 0] = output_post1[box_offset + 0]; // bbox x0
            ((float*)outputs)[output_offset + 1] = output_post1[box_offset + 1]; // bbox y0
            ((float*)outputs)[output_offset + 2] = output_post1[box_offset + 2]; // bbox x1
            ((float*)outputs)[output_offset + 3] = output_post1[box_offset + 3]; // bbox y1
            ((float*)outputs)[output_offset + 4] = output_post2[score_offset]; // score
            ((float*)outputs)[output_offset + 5] = static_cast<float>(output_post3[class_offset]); // class id
        }
    }
    output_post0.clear();
    output_post1.clear();
    output_post2.clear();
    output_post3.clear();
}
// Creat the engine using onnx.
void yolox_opti_trt::createEngineFromOnnx(std::unique_ptr<nvinfer1::IBuilder>& builder, std::unique_ptr<nvinfer1::IBuilderConfig>& config)
{
    trtLogger.log(Logger::Severity::kINFO,"make network start");

    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));

    if (!network)
    {
        trtLogger.log(Logger::Severity::kERROR,"Unable to create network object.");
        exit(EXIT_FAILURE);
    }

    auto parser = nvonnxparser::createParser(*network, trtLogger);
    if (!parser->parseFromFile(weight_file_path.data(), (int)nvinfer1::ILogger::Severity::kINFO))
    {
        trtLogger.log(Logger::Severity::kERROR,"failed to parse onnx file.");
        exit(EXIT_FAILURE);
    }

    // Print parsing errors if any
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // Set memory pool limits
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);

    if (precision_mode == 16)
    {
        if (builder->platformHasFastFp16()){
            trtLogger.log(Logger::Severity::kINFO,"precision f16");
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        } else {
            trtLogger.log(Logger::Severity::kWARNING,"fp16 is not supported, use fp32");
        }
    }
    else if (precision_mode == 8)
    {
        trtLogger.log(Logger::Severity::kINFO,"precision int8 is not work");
    }
    else
    {
        trtLogger.log(Logger::Severity::kINFO,"precision f32");
    }

    trtLogger.log(Logger::Severity::kINFO,"make network done");
    trtLogger.log(Logger::Severity::kINFO,"build engine, please wait for a while...");

    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        trtLogger.log(Logger::Severity::kERROR,"Unable to build serialized plan.");
        exit(EXIT_FAILURE);
    }

    trtLogger.log(Logger::Severity::kINFO,"build engine done");
    trtLogger.log(Logger::Severity::kINFO,"engine file generation start");

    std::ofstream p(engine_file_path, std::ios::binary);
    if (!p)
    {
        trtLogger.log(Logger::Severity::kERROR,"could not open engine file");
        exit(EXIT_FAILURE);
    }
    p.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    p.close();

    trtLogger.log(Logger::Severity::kINFO,"engine file generation done");
}
