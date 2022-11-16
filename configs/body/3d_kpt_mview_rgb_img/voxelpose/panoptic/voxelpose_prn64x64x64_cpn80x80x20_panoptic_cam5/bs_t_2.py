_base_ = [
    '../voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5.py'
]

data = dict(
    samples_per_gpu=2
)