#include "main_detr_det.h"

void detr_trt_inference(ai::arg_parsing::Settings *s)
{
    ai::utils::Timer timer;
    tensorrt_infer::detr_infer::detr_detector detr_obj;
    detr_obj.initParameters(s->model_path, s->score_thr);

    if (!ai::utils::file_exist(s->image_path))
    {
        INFO("Error: image path is not exist!!!");
        exit(0);
    }

    std::vector<cv::Mat> images;
    for (int i = 0; i < s->batch_size; i++)
        images.push_back(cv::imread(s->image_path));
    std::vector<ai::cvUtil::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), ai::cvUtil::cvimg_trans_func);

    for (int i = 0; i < s->number_of_warmup_runs; ++i)
        auto warmup_batched_result = detr_obj.forwards(yoloimages);

    ai::cvUtil::BatchBoxArray batched_result;

    timer.start();
    for (int i = 0; i < s->loop_count; ++i)
        batched_result = detr_obj.forwards(yoloimages);
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        ai::cvUtil::draw_batch_rectangle(images, batched_result, s->output_dir, s->classlabels);
    }
}