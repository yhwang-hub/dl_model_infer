from mmdet3d.apis import init_model, inference_mono_3d_detector, show_result_meshlab
import glob
import mmcv
from os import path as osp

# 这里是最好用绝对路劲，相对路劲有时候行不通，我也不知道是为什么
# config_file代表的是mmdetection3d的文件夹下ocnfigs的模型配置json文件。这可以根据自己的需求修改
config_file = '/home/wyh/mmdetection3d/configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py' 
# ann_file是关于create_data.py后生成的data里面的mono3d.json文件，也是同上可以自行修改。
ann_file = '/home/wyh/mmdetection3d/data/kitti/kitti_infos_test_mono3d.coco.json'
# image_path是关于自己下载数据集的图片所在文件夹的路径，这个要根据个人的需求进行该别：注意：必行要跟上面的ann_file相匹配。
# image_path = r'/home/wyh/mmdetection3d/data/kitti/testing/image_2/'
image_path = r'/home/wyh/mmdetection3d/demo/data/kitti/'
# checkpoints_file是关于自己训练模型的绝对路径，修改同上。
checkpoint_file = '/home/wyh/mmdetection3d/checkpoints/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.pth'
# images_name是为了获得文件夹下的所在的所有图片的绝对路径。
images_name = glob.glob(image_path + "*.png")
print("images_name:{}".format(images_name))
for image_name in images_name:
	# 这一步是为了有些数据集如kitti下有很多有不同相机在同一时间和统一位置照的图片，
	#有下角标_，这下面的几步就是为了将其剔除，只留下真正的不含_的图片。
    # if osp.basename(image_name).partition('_')[0] == osp.basename(image_name):
    #     continue
    image_name = osp.basename(image_name).partition('_')[0]
    # image = image_path + image_name + '.png'
    image = image_path + image_name
    print("image_name:{}".format(image_name))

    # data_infos = mmcv.load(ann_file)
    # # find the info corresponding to this image
    # for x in data_infos['images']:
    #     # print('x:',x)
    #     # print('image_name:', osp.basename(image))
    #     # print("x['file_name']:", osp.basename(x['file_name']))
    #     if osp.basename(x['file_name']) != osp.basename(image):
    #         continue
    #     img_info = x
    #     # print('img_info:', img_info)
    #     break
	# 这步是为了读取网络模型。
    print("start inference................")
    model = init_model(config_file, checkpoint_file, device='cuda:0')
	# 这一步是为了的到结果和数据
    result, data = inference_mono_3d_detector(model, image, ann_file)
    print("end inference................")
	
    print("start save result................")
	# 这一步是为了将最终的结果以单目3d检测的方式储存并保存到你想要的路径下
    out_dir = './outputs/result'
    show_result_meshlab(data, result, out_dir, task='mono-det')
    print("end save result................")