#include <fstream>

#include <opencv2/opencv.hpp>

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net &net, bool is_cuda)
{
    auto result = cv::dnn::readNet("config_files/yolov5n.onnx");
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

const int netWidth = 640;
const int netHeight = 640;
const float nmsThreshold = 0.45;
const float boxThreshold = 0.25;
const float netStride[3] = {8, 16.0, 32};
const float netAnchors[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0}, {116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};
const float classThreshold = 0.25;

struct Detection
{
    int id;
    float confidence;
    cv::Rect box;
};

void detect(cv::Mat &SrcImg, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className)
{
    cv::Mat blob;
    int col = SrcImg.cols;
    int row = SrcImg.rows;
    int maxLen = MAX(col, row);
    cv::Mat netInputImg = SrcImg.clone();
    if (maxLen > 1.2 * col || maxLen > 1.2 * row)
    {
        cv::Mat resizeImg = cv::Mat::zeros(maxLen, maxLen, CV_8UC3);
        SrcImg.copyTo(resizeImg(cv::Rect(0, 0, col, row)));
        netInputImg = resizeImg;
    }
    cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> netOutputImg;
    net.forward(netOutputImg, net.getUnconnectedOutLayersNames());

    int nDims = netOutputImg[0].dims;
    auto size = netOutputImg[0].size();

    //std::cout << size.height << " " << size.width << " " << nDims << " " << netOutputImg[0].channels() << "\n";

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float ratio_h = (float)netInputImg.rows / netHeight;
    float ratio_w = (float)netInputImg.cols / netWidth;
    int net_width = className.size() + 5;
    float *pdata = (float *)netOutputImg[0].data;
    for (int stride = 0; stride < 3; stride++)
    {
        int grid_x = (int)(netWidth / netStride[stride]);
        int grid_y = (int)(netHeight / netStride[stride]);
        for (int anchor = 0; anchor < 3; anchor++)
        {
            const float anchor_w = netAnchors[stride][anchor * 2];
            const float anchor_h = netAnchors[stride][anchor * 2 + 1];
            for (int i = 0; i < grid_y; i++)
            {
                for (int j = 0; j < grid_x; j++)
                {
                    float box_score = pdata[4];
                    if (box_score > boxThreshold)
                    {
                        cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
                        cv::Point classIdPoint;
                        double max_class_socre;
                        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                        max_class_socre = (float)max_class_socre;
                        if (max_class_socre > classThreshold)
                        {
                            float x = pdata[0];
                            float y = pdata[1];
                            float w = pdata[2];
                            float h = pdata[3];
                            int left = (x - 0.5 * w) * ratio_w;
                            int top = (y - 0.5 * h) * ratio_h;
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back(max_class_socre * box_score);
                            boxes.push_back(cv::Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
                        }
                    }
                    pdata += net_width;
                }
            }
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.id = classIds[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

int main(int argc, char **argv)
{

    std::vector<std::string> class_list = load_class_list();

    cv::Mat frame;
    cv::VideoCapture capture("sample.mp4");
    if (!capture.isOpened())
    {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    cv::dnn::Net net;
    load_net(net, is_cuda);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            std::cout << "End of stream\n";
            break;
        }

        std::vector<Detection> output;
        detect(frame, net, output, class_list);

        frame_count++;
        total_frames++;

        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(frame, box, color, 3);

            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        if (frame_count >= 30)
        {

            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("output", frame);

        if (cv::waitKey(1) != -1)
        {
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
    }

    std::cout << "Total frames: " << total_frames << "\n";

    return 0;
}