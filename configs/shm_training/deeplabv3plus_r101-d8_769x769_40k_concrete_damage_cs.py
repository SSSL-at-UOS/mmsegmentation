_base_ = './deeplabv3plus_r50-d8_769x769_40k_concrete_damage_cs.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

work_dir = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/2020.09.02_deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs'
