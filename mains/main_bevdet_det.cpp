#include "main_bevdet_det.h"

void TestNuscenes(YAML::Node &config)
{
    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
    std::string data_info_path = config["dataset_info"].as<std::string>();
    std::string model_config = config["ModelConfig"].as<std::string>();
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    std::string output_dir = config["OutputDir"].as<std::string>();
    std::vector<std::string> cams_name = config["cams"].as<std::vector<std::string>>();

    ai::cvUtil::DataLoader nuscenes(img_N, img_h, img_w, data_info_path, cams_name);
    tensorrt_infer::bevdet_det_infer::bevdet_detector bevdet(model_config, img_N,
                                                             nuscenes.get_cams_intrin(), 
                                                             nuscenes.get_cams2ego_rot(),
                                                             nuscenes.get_cams2ego_trans(),
                                                             imgstage_file, bevstage_file);
    std::vector<ai::utils::bevBox> ego_boxes;
    double sum_time = 0;
    int  cnt = 0;
    for(int i = 0; i < nuscenes.size(); i++)
    {
        ego_boxes.clear();
        float time = 0.f;
        bevdet.forward(nuscenes.data(i), ego_boxes, time, i);
        if(i != 0){
            sum_time += time;
            cnt++;
        }
        ai::utils::Boxes2Txt(ego_boxes, output_dir + "/bevdet_egoboxes_" + std::to_string(i) + ".txt", true);
    }
    printf("Infer mean cost time : %.5lf ms\n", sum_time / cnt);
}

void TestSample(YAML::Node &config){
    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
    std::string model_config = config["ModelConfig"].as<std::string>();
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    YAML::Node camconfig = YAML::LoadFile(config["CamConfig"].as<std::string>()); 
    std::string output_lidarbox = config["OutputLidarBox"].as<std::string>();
    YAML::Node sample = config["sample"];

    std::vector<std::string> imgs_file;
    std::vector<std::string> imgs_name;
    for(auto file : sample)
    {
        imgs_file.push_back(file.second.as<std::string>());
        imgs_name.push_back(file.first.as<std::string>()); 
    }

    ai::cvUtil::camsData sampleData;
    sampleData.param = ai::cvUtil::camParams(camconfig, img_N, imgs_name);

    tensorrt_infer::bevdet_det_infer::bevdet_detector bevdet(model_config, img_N, sampleData.param.cams_intrin, 
                                                             sampleData.param.cams2ego_rot, sampleData.param.cams2ego_trans, 
                                                             imgstage_file, bevstage_file);
    std::vector<std::vector<char>> imgs_data;
    ai::cvUtil::read_sample(imgs_file, imgs_data);

    uchar* imgs_dev = nullptr;
    CHECK(cudaMalloc((void**)&imgs_dev, img_N * 3 * img_w * img_h * sizeof(uchar)));
    decode_cpu(imgs_data, imgs_dev, img_w, img_h);
    sampleData.imgs_dev = imgs_dev;

    std::vector<ai::utils::bevBox> ego_boxes;
    ego_boxes.clear();
    float time = 0.f;
    bevdet.forward(sampleData, ego_boxes, time);
    std::vector<ai::utils::bevBox> lidar_boxes;
    ai::utils::Egobox2Lidarbox(ego_boxes, lidar_boxes, sampleData.param.lidar2ego_rot, 
                    sampleData.param.lidar2ego_trans);
    ai::utils::Boxes2Txt(lidar_boxes, output_lidarbox, false);
    ego_boxes.clear();
    bevdet.forward(sampleData, ego_boxes, time); // only for inference time
}

void bevdet_trt_inference()
{
    std::string config_file = "/home/uisee/dl_model_infer/application/bevdet_app/configure.yaml";
    YAML::Node config = YAML::LoadFile(config_file);
    printf("Successful load config : %s!\n", config_file.c_str());
    bool testNuscenes = config["TestNuscenes"].as<bool>();

    if(testNuscenes)
    {
        TestNuscenes(config);
    }
    else
    {
        TestSample(config);
    }
}