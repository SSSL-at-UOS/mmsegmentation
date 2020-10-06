_base_ = './deeplabv3plus_r50-d8_769x769_40k_concrete_damage_cs.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))

work_dir = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_x101-d8_769x769_40k_concrete_damage_cs'
