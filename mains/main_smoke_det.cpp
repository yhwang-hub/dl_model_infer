#include "main_smoke_det.h"

void smoke_det_trt_inference(ai::arg_parsing::Settings *s)
{
    ai::utils::Timer timer; // 创建
    tensorrt_infer::smoke_det_infer::smoke_detector smoke_det_obj;
    smoke_det_obj.initParameters(s->model_path, s->score_thr);

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

    // 模型预热，如果要单张推理，请调用smoke_det_obj.forward
    for (int i = 0; i < s->number_of_warmup_runs; ++i)
    {
        auto warmup_batched_result = smoke_det_obj.forwards(yoloimages);
    }

    ai::cvUtil::BatchCubeArray batched_result;
    // 模型推理
    timer.start();
    for (int i = 0; i < s->loop_count; ++i)
    {
        batched_result = smoke_det_obj.forwards(yoloimages);
    }
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        for (int ib = 0; ib < (int)batched_result.size(); ib++)
        {
            auto &objs = batched_result[ib];
            auto &image = images[ib];
            int count = 0;
            for (auto &obj : objs)
            {
                count++;
                for (int i = 0; i < 4; i++)
                {
                    const auto& p1 = obj.cube_point[i];
                    const auto& p2 = obj.cube_point[(i + 1) % 4];
                    const auto& p3 = obj.cube_point[i + 4];
                    const auto& p4 = obj.cube_point[(i + 1) % 4 + 4];
                    cv::line(image, p1, p2, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
                    cv::line(image, p3, p4, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
                    cv::line(image, p1, p3, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
                }
            }
            if (ai::utils::mkdirs(s->output_dir))
            {
                std::string save_path = ai::utils::path_join("%s/Infer_%d.jpg", s->output_dir.c_str(), ib);
                cv::imwrite(save_path, image);
            }
        }
    }
}