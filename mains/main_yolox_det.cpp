#include "main_yolox_det.h"

void yolox_det_trt_inference(ai::arg_parsing::Settings *s)
{
    ai::utils::Timer timer;
    tensorrt_infer::yolox_infer::yolox_detector yolox_det_obj;
    yolox_det_obj.initParameters(s->model_path, s->score_thr);

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

    // 模型预热，如果要单张推理，请调用yyolox_det_obj.forward
    for (int i = 0; i < s->number_of_warmup_runs; ++i)
    {
        auto warmup_batched_result = yolox_det_obj.forwards(yoloimages);
    }

    ai::cvUtil::BatchBoxArray batched_result;
    // 模型推理
    timer.start();
    for (int i = 0; i < s->loop_count; ++i)
    {
        batched_result = yolox_det_obj.forwards(yoloimages);
    }
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        ai::cvUtil::draw_batch_rectangle(images, batched_result, s->output_dir, s->classlabels);
    }
}