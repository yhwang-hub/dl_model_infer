#include "main_yolop_det.h"

void yolop_det_trt_inference(ai::arg_parsing::Settings *s)
{
    ai::utils::Timer timer;
    tensorrt_infer::yolop_infer::yolop_detector yolop_det_obj;
    if(s->infer_task == "yolopv1")
    {
        yolop_det_obj.initParameters(s->model_path, ai::cvUtil::DetectorType::YOLOPV1, s->score_thr);
    }
    else if (s->infer_task == "yolopv2")
    {
        yolop_det_obj.initParameters(s->model_path, ai::cvUtil::DetectorType::YOLOPV2, s->score_thr);/* code */
    }

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

    for (int i = 0; i < s->number_of_warmup_runs; ++i)
    {
        auto warmup_batched_result = yolop_det_obj.forwards(yoloimages);
    }

    ai::cvUtil::BatchPTMM batched_result;
    timer.start();
    for (int i = 0; i < s->loop_count; ++i)
    {
        batched_result = yolop_det_obj.forwards(yoloimages);
    }
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        for (int ib = 0; ib < batched_result.size(); ib++)
        {
            auto &image = images[ib];
            auto res = batched_result[ib];
            ai::cvUtil::BoxArray& boxes = get<0>(res);
            cv::Mat& drive_lane         = get<1>(res);
            cv::Mat& drive_mask         = get<2>(res);
            cv::Mat& lane_mask          = get<3>(res);

            for(auto& ibox : boxes)
            {
                cv::rectangle(drive_lane, cv::Point(ibox.left, ibox.top),
                        cv::Point(ibox.right, ibox.bottom),
                        {0, 0, 255}, 2);
            }

            if (ai::utils::mkdirs(s->output_dir))
            {
                std::string drive_lane_save_path = ai::utils::path_join("%s/drive_lane_Infer_%d.jpg", s->output_dir.c_str(), ib);
                std::string drive_mask_save_path = ai::utils::path_join("%s/drive_mask_Infer_%d.jpg", s->output_dir.c_str(), ib);
                std::string lane_mask_save_path = ai::utils::path_join("%s/lane_mask_Infer_%d.jpg", s->output_dir.c_str(), ib);
                cv::imwrite(drive_lane_save_path, drive_lane);
                cv::imwrite(drive_mask_save_path, drive_mask);
                cv::imwrite(lane_mask_save_path, lane_mask);
            }
        }
    }
}