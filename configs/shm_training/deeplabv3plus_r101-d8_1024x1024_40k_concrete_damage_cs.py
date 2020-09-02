_base_ = './deeplabv3plus_r50-d8_1024x1024_40k_concrete_damage_cs.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
