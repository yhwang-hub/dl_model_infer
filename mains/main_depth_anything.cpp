#include "main_depth_anything.h"

void depth_anything_trt_inference(ai::arg_parsing::Settings *s)
{
    ai::utils::Timer timer; // 创建
    tensorrt_infer::depth_anything_infer::depth_anything_detector depth_anything_obj;
    depth_anything_obj.initParameters(s->model_path);

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

    // 模型预热，如果要单张推理，请调用yolov7_det_obj.forward
    for (int i = 0; i < s->number_of_warmup_runs; ++i)
    {
        auto warmup_batched_result = depth_anything_obj.forwards(yoloimages);
    }

    std::vector<cv::Mat> batched_result;
    // 模型推理
    timer.start();
    for (int i = 0; i < s->loop_count; ++i)
    {
        batched_result = depth_anything_obj.forwards(yoloimages);
    }
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        for (int ib = 0; ib < (int)batched_result.size(); ++ib)
        {
            auto& mask_mat = batched_result[ib];
            auto& image = images[ib];
            int img_h = image.rows;
            int img_w = image.cols;
            cv::Mat show_frame;
            image.copyTo(show_frame);

            cv::normalize(mask_mat, mask_mat, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::Mat colormap;
            cv::applyColorMap(mask_mat, colormap, cv::COLORMAP_RAINBOW);
            cv::resize(colormap, colormap, cv::Size(img_w, img_h));
            addWeighted(show_frame, 0.7, colormap, 0.3, 0.0, show_frame);

            cv::Mat result;
            cv::hconcat(colormap, show_frame, result);
            cv::resize(result, result, cv::Size(1080, 720));

            if (ai::utils::mkdirs(s->output_dir))
            {
                std::string save_path = ai::utils::path_join("%s/Infer_%d.jpg", s->output_dir.c_str(), ib);
                cv::imwrite(save_path, result);
            }
        }
    }
}