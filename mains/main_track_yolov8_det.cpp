#include "main_track_yolov8_det.h"

void yolov8_det_track_trt_inference(ai::arg_parsing::Settings *s)
{
    // yolov8 det初始化
    tensorrt_infer::yolov8_infer::yolov8_detector yolov8_det_obj;
    yolov8_det_obj.initParameters(s->model_path, s->score_thr);

    ai::utils::Timer timer; // 初始化计时器

    // 这里的image_path 实际指的是输入的视频路径
    if (!ai::utils::file_exist(s->image_path))
    {
        INFO("Error: video path is not exist!!!");
        exit(0);
    }
    
    cv::VideoCapture cap;
    cap.open(s->image_path);
    int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS); // 追踪类初始化时要用到
    cv::VideoWriter writer("res/demo_res.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(img_w, img_h));
    if (!cap.isOpened())
    {
        INFO("input video is destoryed,please check your video!");
        exit(RETURN_FAIL);
    }

    // 追踪类初始化,第二个参数是目标丢失后，track_id最长保留的帧数，超过该帧数，track_id+1
    ai::ByteTrack::BYTETracker tracker(fps, 30); // 这里最长保留30帧
    cv::Mat img;

    while (true)
    {
        cap >> img;
        if (img.empty())
            break;

        ai::cvUtil::BoxArray res_boxes = yolov8_det_obj.forward(ai::cvUtil::cvimg_trans_func(img));
        timer.start();
        std::vector<ai::ByteTrack::STrack> output_stracks = tracker.update(res_boxes);
        timer.stop("track time[ms]:");

        // 为了可以直接删除utils/tracker文件夹，不影响其他项目使用，就再这里直接画框了
        for (int i = 0; i < output_stracks.size(); i++)
        {
            vector<float> tlwh = output_stracks[i].tlwh;
            cv::Scalar color = tracker.get_color(output_stracks[i].class_label);
            cv::rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), color, 2);
            auto name = s->classlabels[output_stracks[i].class_label];
            auto caption = cv::format("%s %.2f", name.c_str(), output_stracks[i].score);
            auto track_caption = cv::format("id:%d", output_stracks[i].track_id);
            int txt_width = cv::getTextSize(caption, 1, 0.8, 0.6, nullptr).width;
            int width = (txt_width > tlwh[2]) ? txt_width : tlwh[2];
            cv::rectangle(img, cv::Point(tlwh[0] - 3, tlwh[1] - 33), cv::Point(tlwh[0] + width, tlwh[1]), color, -1);
            cv::putText(img, caption, cv::Point(tlwh[0], tlwh[1] - 5), 1, 0.8, cv::Scalar::all(0), 0.6, 1);
            cv::putText(img, track_caption, cv::Point(tlwh[0], tlwh[1] - 20), 1, 1, cv::Scalar::all(0), 0.6, 1);
        }

        // cv::resize(img, img, cv::Size(), 0.5, 0.5);
        // cv::imshow("win", img);
        // cv::waitKey(1);
        writer.write(img);
    }
    cap.release();
}