import os

# base options
dist_params = dict(backend="nccl")  # Parameters to setup distributed training, the port can also be set.
log_level = "INFO"  # The level of logging.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the iteration when the checkpoint's is saved.
cudnn_benchmark = True  # Whether use cudnn_benchmark to speed up, which is fast for fixed input size.

custom_imports = dict(imports=["geospatial_fm"])


### Configs
# Data
data_root = "/kaggle/input/s1s2-water-dataset/split_data/"

dataset_type = "GeospatialDataset"
num_classes = 2
num_frames = 1
img_size = 224
num_workers = 2  # Worker to pre-fetch data for each single GPU
samples_per_gpu = 14  # Batch size of a single GPU
CLASSES = (0, 1)

### Computed on training and validation sets of parts 1, 2 and 5 of the dataset
img_norm_cfg=dict(
    means= [0.10735853, 0.09544952, 0.08811411, 0.19080341, 0.18524158, 0.11630596],
    stds= [0.02595188, 0.02899879, 0.04512926, 0.08647086, 0.09900333, 0.07598658]
)

bands = [0, 1, 2, 3, 4, 5]
tile_size = img_size
orig_nsize = 512
crop_size = (tile_size, tile_size)

train_img_dir = "train/img"
train_ann_dir = "train/msk"
val_img_dir = "val/img"
val_ann_dir = "val/msk"
test_img_dir = "test/img"
test_ann_dir = "test/msk"
img_suffix = "_img.tif"
seg_map_suffix = "_msk.tif"


# ignore_index = 2  # ?
# label_nodata = -1  # Data is already preprocessed to remove tiles with missing data
# image_nodata = -9999
# image_nodata_replace = 0

# Model
pretrained_weights_path = "./backbones/prithvi/Prithvi_100M.pt"
num_layers = 12  # Left to default
patch_size = 16  # Left to default
embed_dim = 768  # Left to default
num_heads = 12  # Left to default
tubelet_size = 1  # Left to default

# TRAINING
epochs = 2
eval_epoch_interval = 2

# TO BE DEFINED BY USER: Save directory
experiment = "s1s2_water_large_dataset_pretrained_head_sen1floods11_nofreeze"
project_dir = "experiments"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

# Pipelines
train_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",  # Loads a tiff image. Returns in channels last format.
        to_float32=False,
        # nodata=image_nodata,
        # nodata_replace=image_nodata_replace,
    ),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        # nodata=label_nodata,
        # nodata_replace=ignore_index,
    ),
    dict(type="BandsExtract", bands=bands),
    # dict(type="ConstantMultiply", constant=constant),
    dict(type="RandomFlip", prob=0),  # No data augmentation for now
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    # dict(type="TorchRandomCrop", crop_size=crop_size),  # No data augmentation for now
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]


test_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=False,
        # nodata=image_nodata,
        # nodata_replace=image_nodata_replace,
    ),
    dict(type="BandsExtract", bands=bands),
    # dict(type="ConstantMultiply", constant=constant),
    dict(type="ToTensor", keys=["img"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, -1, -1),
        look_up={"2": 1, "3": 2},  # TODO: why is this needed?
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]

# Dataset
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        # ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir=val_img_dir,
        ann_dir=val_ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        # ignore_index=ignore_index,
        # gt_seg_map_loader_cfg=dict(nodata=label_nodata, nodata_replace=ignore_index),
    ),
    test=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir=test_img_dir,
        ann_dir=test_ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        # ignore_index=ignore_index,
        # gt_seg_map_loader_cfg=dict(nodata=label_nodata, nodata_replace=ignore_index),
    ),
)

# Training
# Unless stated otherwise, here we use the parameters of the sen1floods11 article
optimizer = dict(
    type="AdamW",
    lr=1.5e-5,
    weight_decay=0.05,
    betas=(0.9, 0.999),
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=500,  # Reduced this number of iterations as we do not do as many epochs as for sen1flodds11
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        dict(type="TensorboardLoggerHook", by_epoch=True),
    ],
)

checkpoint_config = dict(by_epoch=True, interval=2, out_dir=save_path)  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.

evaluation = dict(  # The config to build the evaluation hook. Please refer to mmseg/core/evaluation/eval_hook.py for details
    interval=eval_epoch_interval,
    metric=["mIoU", "mFscore"],
    pre_eval=True,
    save_best="mIoU",
    by_epoch=True,
)

runner = dict(type="EpochBasedRunner", max_epochs=epochs)

workflow = [("train", 1), ("val", 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.

norm_cfg = dict(type="BN", requires_grad=True)  # The configuration of norm layer

ce_weights = [0.3, 0.7]

model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=False,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=num_layers,
        num_heads=num_heads,
        mlp_ratio=4.0,  # Ratio of mlp hidden dim to embedding dim
        norm_pix_loss=False,  # Whether or not normalize target
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=num_frames * embed_dim,
        output_embed_dim=embed_dim,
        drop_cls_token=True,
        Hp=img_size // patch_size,
        Wp=img_size // patch_size,
    ),
    decode_head=dict(
        num_classes=num_classes,
        in_channels=embed_dim,
        type="FCNHead",
        in_index=-1,  # The index of feature map to select
        # ignore_index=ignore_index,
        channels=256,
        num_convs=1,
        concat_input=False,  # Whether concat output of convs with input before classification layer
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,  # The align_corners argument for resize in decoding
        loss_decode=dict(  # Config of loss function for the decode_head
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,  # Loss weight of decode head (vs auxiliary head)
            class_weight=ce_weights,
            # avg_non_ignore=True,
        ),
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        in_channels=embed_dim,
        # ignore_index=ignore_index,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=ce_weights,
            # avg_non_ignore=True,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(  
        mode="slide",  # The test mode, options are 'whole' and 'sliding'. 'whole': whole image fully-convolutional test. 'sliding': sliding crop window on the image.
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
