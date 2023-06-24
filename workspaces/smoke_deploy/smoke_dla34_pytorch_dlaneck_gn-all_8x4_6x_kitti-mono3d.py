dataset_type = 'KittiMonoDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomShiftScale', shift_scale=(0.2, 0.4), aug_prob=0.3),
    dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car']),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'centers2d', 'depths'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 384),
        flip=False,
        transforms=[
            dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car'],
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car'],
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='KittiMonoDataset',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_train_mono3d.coco.json',
        info_file='data/kitti/kitti_infos_train.pkl',
        img_prefix='data/kitti/',
        classes=['Pedestrian', 'Cyclist', 'Car'],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_bbox=True,
                with_label=True,
                with_attr_label=False,
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox_depth=True),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='RandomShiftScale', shift_scale=(0.2, 0.4), aug_prob=0.3),
            dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car']),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                    'gt_labels_3d', 'centers2d', 'depths'
                ])
        ],
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        box_type_3d='Camera'),
    val=dict(
        type='KittiMonoDataset',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_val_mono3d.coco.json',
        info_file='data/kitti/kitti_infos_val.pkl',
        img_prefix='data/kitti/',
        classes=['Pedestrian', 'Cyclist', 'Car'],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 384),
                flip=False,
                transforms=[
                    dict(
                        type='AffineResize',
                        img_scale=(1280, 384),
                        down_ratio=4),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian', 'Cyclist', 'Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['img'])
                ])
        ],
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type='KittiMonoDataset',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_val_mono3d.coco.json',
        info_file='data/kitti/kitti_infos_val.pkl',
        img_prefix='data/kitti/',
        classes=['Pedestrian', 'Cyclist', 'Car'],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 384),
                flip=False,
                transforms=[
                    dict(
                        type='AffineResize',
                        img_scale=(1280, 384),
                        down_ratio=4),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian', 'Cyclist', 'Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['img'])
                ])
        ],
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        box_type_3d='Camera'))
evaluation = dict(interval=2)
model = dict(
    type='SMOKEMono3D',
    backbone=dict(
        type='DLANet',
        depth=34,
        in_channels=3,
        norm_cfg=dict(type='GN', num_groups=32),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
        )),
    neck=dict(
        type='DLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5,
        norm_cfg=dict(type='GN', num_groups=32)),
    bbox_head=dict(
        type='SMOKEMono3DHead',
        num_classes=3,
        in_channels=64,
        dim_channel=[3, 4, 5],
        ori_channel=[6, 7],
        stacked_convs=0,
        feat_channels=64,
        use_direction_classifier=False,
        diff_rad_by_sin=False,
        pred_attrs=False,
        pred_velo=False,
        dir_offset=0,
        strides=None,
        group_reg_dims=(8, ),
        cls_branch=(256, ),
        reg_branch=((256, ), ),
        num_attrs=0,
        bbox_code_size=7,
        dir_branch=(),
        attr_branch=(),
        bbox_coder=dict(
            type='SMOKECoder',
            base_depth=(28.01, 16.32),
            base_dims=((0.88, 1.73, 0.67), (1.78, 1.7, 0.58), (3.88, 1.63,
                                                               1.53)),
            code_size=7),
        loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss', reduction='sum', loss_weight=0.0033333333333333335),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=None,
        conv_bias=True,
        dcn_on_last_conv=False),
    train_cfg=None,
    test_cfg=dict(topK=100, local_maximum_kernel=3, max_per_img=100))
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
optimizer = dict(type='Adam', lr=0.00025)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup=None, step=[50])
runner = dict(type='EpochBasedRunner', max_epochs=72)
find_unused_parameters = True

