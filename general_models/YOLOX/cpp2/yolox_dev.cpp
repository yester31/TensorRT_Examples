#include "yolox_opti_trt.hpp"

#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

// Do data pre-processing
void Preprocess(std::vector<float> &output, std::vector<uint8_t> &input, int BatchSize, int channels, int height, int width)
{
    /*
        INPUT  = BGR[NHWC](0, 255)
        OUTPUT = BGR[NCHW]
        - Shuffle form HWC to CHW
    */
    int offset = channels * height * width;
    int b_off, c_off, h_off, h_off_o;
    for (int b = 0; b < BatchSize; b++)
    {
        b_off = b * offset;
        for (int c = 0; c < channels; c++)
        {
            c_off = c * height * width + b_off;
            for (int h = 0; h < height; h++)
            {
                h_off = h * width + c_off;
                h_off_o = h * width * channels + b_off;
                for (int w = 0; w < width; w++)
                {
                    int dstIdx = h_off + w;
                    int srcIdx = h_off_o + w * channels;
                    output[dstIdx] = (static_cast<const float>(input[srcIdx]));
                }
            }
        }
    }
};

void gen_dir(std::string &engine_dir_path)
{
    if (mkdir(engine_dir_path.c_str(), 0777) == 0)
    {
        std::cout << "generated directory :: " << engine_dir_path << std::endl;
    }
    else
    {
        std::cerr << "already exist" << std::endl;
    }
}

std::vector<std::vector<int>> color_table = {
    {0,   114, 189},
    {217,  83,  25},
    {237, 176,  32},
    {126,  47, 142},
    {119, 172,  48},
    { 77, 190, 238},
    {162,  20,  47},
    { 77,  77,  77},
    {153, 153, 153},
    {255,   0,   0},
    {255, 128,   0},
    {191, 191,   0},
    {  0, 255,   0},
    {  0,   0, 255},
    {170,   0, 255},
    { 85,  85,   0},
    { 85, 170,   0},
    { 85, 255,   0},
    {170,  85,   0},
    {170, 170,   0},
    {170, 255,   0},
    {255,  85,   0},
    {255, 170,   0},
    {255, 255,   0},
    {  0,  85, 128},
    {  0, 170, 128},
    {  0, 255, 128},
    { 85,   0, 128},
    { 85,  85, 128},
    { 85, 170, 128},
    { 85, 255, 128},
    {170,   0, 128},
    {170,  85, 128},
    {170, 170, 128},
    {170, 255, 128},
    {255,   0, 128},
    {255,  85, 128},
    {255, 170, 128},
    {255, 255, 128},
    {  0,  85, 255},
    {  0, 170, 255},
    {  0, 255, 255},
    { 85,   0, 255},
    { 85,  85, 255},
    { 85, 170, 255},
    { 85, 255, 255},
    {170,   0, 255},
    {170,  85, 255},
    {170, 170, 255},
    {170, 255, 255},
    {255,   0, 255},
    {255,  85, 255},
    {255, 170, 255},
    { 85,   0,   0},
    {128,   0,   0},
    {170,   0,   0},
    {213,   0,   0},
    {255,   0,   0},
    {  0,  43,   0},
    {  0,  85,   0},
    {  0, 128,   0},
    {  0, 170,   0},
    {  0, 213,   0},
    {  0, 255,   0},
    {  0,   0,  43},
    {  0,   0,  85},
    {  0,   0, 128},
    {  0,   0, 170},
    {  0,   0, 213},
    {  0,   0, 255},
    {  0,   0,   0},
    { 36,  36,  36},
    { 73,  73,  73},
    {109, 109, 109},
    {146, 146, 146},
    {182, 182, 182},
    {219, 219, 219},
    {  0, 114, 189},
    { 80, 183, 189},
    {128, 128,   0} // (0.50, 0.5, 0) → (128,128,0)
};

std::vector<std::string> defect_names{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

int main()
{
    // 1) parameter setting
    const int BATCH_SIZE{ 1 };
    const int INPUT_H{ 640 };
    const int INPUT_W{ 640 };
    const int INPUT_C{ 3 };
    const int CLASS_COUNT{ 80 };
    const int precision_mode{ 16 }; // fp32 mode : 32, fp16 mode : 16
    int gpu_device{ 0 };            // gpu device index (default = 0)
    bool serialize{ false };        // force serialize flag (IF true, recreate the engine file unconditionally)
    std::string engine_file_name{ "yolox-s" };  // engine file name (engine file will be generated uisng this name)
    std::string engine_dir_path{ "../../engine" }; // engine directory path (engine file will be generated in this location)
    std::string weight_file_path{ "../../onnx/yolox-s_640x640_sim_w_nms.onnx" }; // weight file path

    yolox_opti_trt yolox_trt = yolox_opti_trt(BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C, CLASS_COUNT, precision_mode, serialize, gpu_device, engine_dir_path, engine_file_name, weight_file_path);

    // 2) prepare input data
    std::vector<std::string> img_paths{"../../data/dog.jpg","../../data/test1.jpg","../../data/test2.jpg"};
    int num_test_imgs = static_cast<int>(img_paths.size());
    std::cout << "num_test_imgs : "<< num_test_imgs << std::endl;

    // 3) Inference results check
    std::vector<cv::Mat> imgs; // temporary save for visualization
    std::vector<float> ratios; // temporary save for visualization
    int input_size = INPUT_H * INPUT_W * INPUT_C;
    int output_size = (1 + 6 * 300);
    std::vector<uint8_t> inputs0(BATCH_SIZE * input_size); // [640, 640, 3]
    std::vector<float> inputs(BATCH_SIZE * input_size);
    std::vector<float> outputs(BATCH_SIZE * output_size); // [the number of detection,  {bbox[x,y,w,h], score, cls_id} * 300]

    for (int i = 0; i < static_cast<int>(ceil(static_cast<float>(num_test_imgs) / BATCH_SIZE)); i++) // batch unit loop
    {
        // load image
        for (int b_idx = 0; b_idx < BATCH_SIZE; b_idx++)
        {
            int imd_idx = (i * BATCH_SIZE + b_idx < num_test_imgs) ? i * BATCH_SIZE + b_idx : num_test_imgs - 1;
            cv::Mat ori_img = cv::imread(img_paths[imd_idx]);
            imgs.push_back(ori_img);
            if (!ori_img.data)
            {
                std::cerr << "[ERROR] Data load error (Check image path)" << std::endl;
            }

            // preprocess input images
            float ratio = std::min((float)INPUT_W / (ori_img.cols), (float)INPUT_H / (ori_img.rows));
            ratios.push_back(ratio);
            int unpad_w = ratio * ori_img.cols;
            int unpad_h = ratio * ori_img.rows;
            // resize
            cv::Mat resized_img(unpad_h, unpad_w, CV_8UC3);
            cv::resize(ori_img, resized_img, resized_img.size());
            // pad
            cv::Mat padded_img(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
            resized_img.copyTo(padded_img(cv::Rect(0, 0, resized_img.cols, resized_img.rows)));
            
            memcpy(inputs0.data() + b_idx * input_size, padded_img.data, input_size);
        }
        Preprocess(inputs, inputs0, BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W); // int8 BGR[NHWC](0, 255) -> float BGR[NCHW](0, 255)

        yolox_trt.input_data(inputs.data());
        yolox_trt.run_model();
        yolox_trt.output_data(outputs.data());

        // draw results (with tuning)
        for (int b_idx = 0; b_idx < BATCH_SIZE; b_idx++)
        {
            int imd_idx = (i * BATCH_SIZE + b_idx < num_test_imgs) ? i * BATCH_SIZE + b_idx : num_test_imgs - 1;
            //draw_bbox_text(outputs.data() + b_idx * output_size, imgs[b_idx], img_paths[imd_idx], color_table, defect_names, 0.8f);
            int num_dets = static_cast<int>(outputs[b_idx * output_size + 0]);  // number of detections
            std::cout << num_dets << std::endl;

            std::filesystem::path p(img_paths[imd_idx]);
            std::string img_name = p.stem().string();
            cv::Mat img = imgs[imd_idx];
            float ratio = ratios[imd_idx];
            int img_width = img.cols;
            int img_hight = img.rows;
            int x, y, x1, y1, w, h, cls_id;
            float conf;
            float conf_thre = 0.5;
            float* detection_ptr = outputs.data() + b_idx * output_size + 1;
            for (int d_idx = 0; d_idx < num_dets; d_idx++)
            {   
                int g_idx = d_idx * 6;
                conf = detection_ptr[g_idx + 4];
                if (conf < conf_thre) continue;
                x = static_cast<int>(detection_ptr[g_idx]/ratio);
                y = static_cast<int>(detection_ptr[g_idx + 1]/ratio);
                x1 = static_cast<int>(detection_ptr[g_idx + 2]/ratio);
                y1 = static_cast<int>(detection_ptr[g_idx + 3]/ratio);
                cls_id = static_cast<int>(detection_ptr[g_idx + 5]);

                w = x1 - x;
                h = y1 - y;

                // bbox
                cv::Rect rect(x, y, w, h);
                auto color_type = color_table[cls_id % color_table.size()];
                auto color = cv::Scalar(color_type[0], color_type[1], color_type[2]);
                rectangle(img, rect, color, 2, 8, 0);

                // text box
                std::string text = std::to_string(d_idx) + " " + defect_names[cls_id] + " " + std::to_string(conf);
                cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                int text_box_x = (x >= (img_width - 250)) ? x - 250 : x;
                int text_box_y = (y < 50) ? y + 50 : y;
                cv::Rect text_box(text_box_x, text_box_y - 30, text_size.width + 10, text_size.height + 15);
                cv::rectangle(img, text_box, color, cv::FILLED);
                cv::putText(img, text, cv::Point(text_box_x, text_box_y - 3), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
                std::string consol_text = "[" + std::to_string(d_idx) + "] " + img_name + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + ", " +
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
            std::string save_file_path = save_dir_path + "/" + img_name + "_trt_cpp2.jpg";
            std::cout << save_file_path << std::endl;
            cv::imwrite(save_file_path, img); 

            if (!cv::imwrite(save_file_path, img)) {
                std::cerr << "Failed to save file!" << std::endl;
                return -1;
            }

        }
        std::cout << "==========================================================================" << std::endl;
    }
    imgs.clear();

    return 0;
}