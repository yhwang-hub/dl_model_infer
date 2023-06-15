#include "main_yolov8_det.h"

void yolov8_trt_inference(ai::arg_parsing::Settings *s)
{
    ai::utils::Timer timer; // 创建
    tensorrt_infer::yolov8_infer::yolov8_detector yolov8_obj;
    yolov8_obj.initParameters(s->model_path, s->score_thr);

    // 判断图片路径是否存在
    if (!ai::utils::file_exist(s->image_path))
    {
        INFO("Error: image path is not exist!!!");
        exit(0);
    }

    // 加载要推理的数据
    std::vector<cv::Mat> images;
    for (int i = 0; i < s->batch_size; i++)
    {
        images.push_back(cv::imread(s->image_path));
    }
    std::vector<ai::cvUtil::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), ai::cvUtil::cvimg_trans_func);

    // 模型预热，如果要单张推理，请调用yolov8_obj.forward
    for (int i = 0; i < s->number_of_warmup_runs; ++i)
    {
        auto warmup_batched_result = yolov8_obj.forwards(yoloimages);
    }

    ai::cvUtil::BatchBoxArray batched_result;
    // 模型推理
    timer.start();
    for (int i = 0; i < s->loop_count; ++i)
    {
        batched_result = yolov8_obj.forwards(yoloimages);
    }
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        ai::cvUtil::draw_batch_rectangle(images, batched_result, s->output_dir, s->classlabels);
    }
}

void yolov8_trt_inference_perf(ai::arg_parsing::Settings *s)
{
    if (!s->output_dir.empty())
    {
        rmtree(s->output_dir);
    }

    int max_infer_batch = 16;
    int batch = 16;
    std::vector<cv::Mat> images{
        cv::imread("res/bus.jpg"),
        cv::imread("res/dog.jpg"),
        cv::imread("res/zidane.jpg")
    };

    for (int i = images.size(); i < batch; i++)
    {
        images.push_back(images[i % 3]);
    }

    cpm::Instance<BoxArray, Image, yolov8_detector> cpmi;
    bool ok = cpmi.start(
        [&s] { return load(s->model_path, DetectorType::V8); },
        max_infer_batch
    );

    if (!ok) return;

    std::vector<Image> yoloimages(images.size());
    std::transform(
        images.begin(),
        images.end(),
        yoloimages.begin(),
        cvimg_trans_func
    );

    Timer timer;
    for (int i = 0; i < 5; i++)
    {
        timer.start();
        cpmi.commits(yoloimages).back().get();
        timer.stop("BATCH16");
        
        auto result_futures = cpmi.commits(yoloimages);
        for (int ib = 0; ib < result_futures.size(); ++ib)
        {
            auto objs = result_futures[ib].get();
            auto image = images[ib].clone();
            for (auto& obj : objs)
            {
                uint8_t b, g, r;
                tie(b, g, r) = random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                                cv::Scalar(b, g, r), 2);
                auto name = s->classlabels[obj.class_label];
                auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                                cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                            16);
            }
            if (mkdirs(s->output_dir))
            {
                std::string save_path = path_join("%s/Infer_%d.jpg", s->output_dir.c_str(), ib);
                cv::imwrite(save_path, image);
            }
        }
    }

    for (int i = 0; i < 5; i++)
    {
        timer.start();
        cpmi.commit(yoloimages[0]).get();
        timer.stop("BATCH1");
    }
}