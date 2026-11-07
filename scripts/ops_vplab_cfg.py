crop_size = (
    640,
    640,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        122.7709,
        116.746,
        104.0937,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=640,
    std=[
        68.5005,
        66.6322,
        70.3232,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=20,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, interval=1, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/ops-vit-b16_vplab-640x640/iter_10500.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    asymetric_input=True,
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            122.7709,
            116.746,
            104.0937,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=640,
        std=[
            68.5005,
            66.6322,
            70.3232,
        ],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        deep_supervision_idxs=[
            7,
        ],
        loss_decode=[
            dict(
                class_weight=None,
                loss_name='loss_cls_ce',
                loss_weight=2.0,
                type='CrossEntropyLoss'),
            dict(
                loss_name='loss_mask_ce',
                loss_weight=5.0,
                type='CrossEntropyLoss',
                use_sigmoid=True),
            dict(
                eps=1,
                ignore_index=None,
                loss_name='loss_mask_dice',
                loss_weight=5.0,
                naive_dice=True,
                type='DiceLoss'),
        ],
        maskgen_cfg=dict(
            act_cfg=dict(type='QuickGELU'),
            cross_attn=False,
            embed_dims=768,
            final_norm=True,
            frozen_exclude=[],
            mlp_ratio=4,
            norm_cfg=dict(eps=1e-05, type='LN'),
            num_heads=12,
            num_layers=3,
            out_dims=512,
            qkv_bias=True,
            sos_token_format='cls_token',
            sos_token_num=100),
        num_classes=21,
        ops_cfg=dict(
            cfg_dao=dict(
                act_layer='GELU',
                center_feature_scale=True,
                channels=240,
                dilation=1,
                dw_kernel_size=3,
                group=12,
                kernel_size=3,
                norm_layer='LN',
                offset_scale=1.0,
                pa_kernel_size=3,
                pa_norm_layer='Softmax',
                pad=1,
                stride=1),
            cfg_decoder=dict(
                embed_channels=256,
                mlp_channels=256,
                num_heads=12,
                num_layers=1,
                num_mlp=3,
                rescale=True),
            cfg_encoder=dict(mlp_ratio=4, num_encode_layer=8, num_heads=6),
            clip_channels=768,
            dao_after_fusion=False,
            embed_dims=240,
            fuse_type='conv',
            fusion_index=[
                0,
                1,
                2,
                3,
            ],
            in_channels=3,
            norm_cfg=dict(eps=1e-06, type='LN'),
            num_queries=100,
            patch_bias=True,
            patch_size=16),
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='ClassificationCost', weight=2.0),
                    dict(
                        type='CrossEntropyLossCost',
                        use_sigmoid=True,
                        weight=5.0),
                    dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
                ],
                type='HungarianAssigner'),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0),
        type='OPSCLIPHead'),
    encoder_resolution=0.5,
    image_encoder=dict(
        act_cfg=dict(type='QuickGELU'),
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_rate=0.0,
        embed_dims=768,
        frozen_exclude=[
            'pos_embed',
        ],
        img_size=(
            224,
            224,
        ),
        in_channels=3,
        interpolate_mode='bicubic',
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-05, type='LN'),
        norm_eval=False,
        num_heads=12,
        num_layers=9,
        out_indices=(
            2,
            5,
            8,
        ),
        out_origin=True,
        output_cls_token=True,
        patch_bias=False,
        patch_pad=0,
        patch_size=16,
        pre_norm=True,
        qkv_bias=True,
        type='VisionTransformer',
        with_cls_token=True),
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth',
    test_cfg=dict(crop_size=(
        640,
        640,
    ), mode='slide', stride=(
        320,
        320,
    )),
    text_encoder=dict(
        cache_feature=True,
        cat_bg=True,
        dataset_name=None,
        embed_dims=512,
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-05, type='LN'),
        num_heads=8,
        num_layers=12,
        output_dims=512,
        templates='vild',
        type='CLIPTextEncoder',
        vocabulary=[
            'roof',
            'bed',
            'obstructions',
            'refrigerator',
            'phone',
            'wall',
            'furniture',
            'electronics',
            'floor',
            'monitor',
            'cabinet',
            'door',
            'curtain',
            'pillow',
            'window',
            'misc',
            'chair',
            'table',
            'boxes',
            'bag',
            'bottles',
        ]),
    train_cfg=dict(),
    type='MultimodalEncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 21
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=1e-05, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            cls_token=dict(decay_mult=0.0),
            img_encoder=dict(decay_mult=1.0, lr_mult=0.1),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=60000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix=dict(img_path='val/img', seg_map_path='val/masks_mmseg'),
        data_root='Panorama Segmentation DataSet',
        ignore_index=255,
        img_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(max_size=2560, scale=(
                640,
                640,
            ), type='ResizeShortestEdge'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=False,
        seg_map_suffix='.png',
        type='VPLABDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(max_size=2560, scale=(
        640,
        640,
    ), type='ResizeShortestEdge'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=30000,
    type='IterBasedTrainLoop',
    val_begin=100,
    val_interval=500)
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/masks_mmseg'),
        data_root='Panorama Segmentation DataSet',
        ignore_index=255,
        img_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                max_size=2560,
                resize_type='ResizeShortestEdge',
                scales=[
                    320,
                    384,
                    448,
                    512,
                    576,
                    640,
                    704,
                    768,
                    832,
                    896,
                    960,
                ],
                type='RandomChoiceResize'),
            dict(cat_max_ratio=1.0, crop_size=(
                640,
                640,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=False,
        seg_map_suffix='.png',
        type='VPLABDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        max_size=2560,
        resize_type='ResizeShortestEdge',
        scales=[
            320,
            384,
            448,
            512,
            576,
            640,
            704,
            768,
            832,
            896,
            960,
        ],
        type='RandomChoiceResize'),
    dict(cat_max_ratio=1.0, crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(type='PhotoMetricDistortion'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix=dict(img_path='val/img', seg_map_path='val/masks_mmseg'),
        data_root='Panorama Segmentation DataSet',
        ignore_index=255,
        img_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(max_size=2560, scale=(
                640,
                640,
            ), type='ResizeShortestEdge'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=False,
        seg_map_suffix='.png',
        type='VPLABDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='outputs',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/ops-vit-b16_vplab-640x640'
