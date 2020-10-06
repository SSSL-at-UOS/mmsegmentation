_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/concrete_damage_cs.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(
        align_corners=True,
        num_classes=5),
    auxiliary_head=dict(
        align_corners=True, 
        num_classes=5))

test_cfg = dict(mode='slide', crop_size=(1025, 1025), stride=(513, 513))

work_dir = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r50-d8_769x769_40k_concrete_damage_cs'
