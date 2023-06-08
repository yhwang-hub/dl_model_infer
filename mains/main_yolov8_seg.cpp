#include "main_yolov8_seg.h"

void yolov8_seg_trt_inference(ai::arg_parsing::Settings* s)
{
    ai::utils::Timer timer;
    tensorrt_infer::yolov8_infer::yolov8_seg_detector yolov8_seg_obj;
    yolov8_seg_obj.initParameters(s->model_path, s->score_thr);

    if (!ai::utils::file_exist(s->image_path))
    {
        INFO("Error: image path is not exist!!!");
        exit(0);
    }

    std::vector<cv::Mat> images;
    for (int i = 0; i < s->batch_size; i++)
    {
        images.push_back(cv::imread(s->image_path));
    }
    std::vector<ai::cvUtil::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), ai::cvUtil::cvimg_trans_func);
    
    for (int i = 0; i < s->number_of_warmup_runs; i++)
    {
        auto warmup_batched_result = yolov8_seg_obj.forwards(yoloimages);
    }

    ai::cvUtil::BatchSegBoxArray batched_result;
    timer.start();
    for (int i = 0; i < s->loop_count; i++)
    {
        batched_result = yolov8_seg_obj.forwards(yoloimages);
    }
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        ai::cvUtil::draw_batch_segment(images, batched_result, s->output_dir, s->classlabels);
    }
}