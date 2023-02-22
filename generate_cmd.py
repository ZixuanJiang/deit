path_to_imagenet = "/home/usr1/zixuan/ImageNet/data"
job_root_dir = "/home/usr1/zixuan/deit/experiments/"  # please use absolute path here
model = "b"

seed = 0

name_dict = {
    's': 'deit_small_patch16_LS',
    'b': 'deit_base_patch16_LS',
    'l': 'deit_large_patch16_LS',
    'h': 'deit_huge_patch14_LS', }

setting_dict = {
    's': '--batch 256 --nodes 1 --ngpus 8 --lr 4e-3 --input-size 224 --drop-path 0.05 ',
    'b': '--batch 256 --nodes 1 --ngpus 8 --lr 3e-3 --input-size 192 --drop-path 0.2  ',
    'l': '--batch 64  --nodes 4 --ngpus 8 --lr 3e-3 --input-size 192 --drop-path 0.45 ',
    'h': '--batch 64  --nodes 4 --ngpus 8 --lr 3e-3 --input-size 160 --drop-path 0.6  ',
}


def generate_one_command(q_pre_flag=True, k_pre_flag=True, v_pre_flag=True, q_post_flag=False, k_post_flag=False, v_post_flag=False, job_dir=None):
    res = "python run_with_submitit.py "
    res += f"--model {name_dict[model]} "
    res += f"--data-path {path_to_imagenet} --job_dir {job_dir} "
    res += setting_dict[model]
    res += f'--epochs 800 --weight-decay 0.05 --sched cosine --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed {seed} --opt fusedlamb --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment '

    if q_pre_flag:
        res += '--q-pre-flag '
    if k_pre_flag:
        res += '--k-pre-flag '
    if v_pre_flag:
        res += '--v-pre-flag '
    if q_post_flag:
        res += '--q-post-flag '
    if k_post_flag:
        res += '--k-post-flag '
    if v_post_flag:
        res += '--v-post-flag '

    res += '\n'

    return res


print(generate_one_command(False, False, False, False, False, False, job_root_dir + 'none-none'))

print(generate_one_command(True, False, False, False, False, False, job_root_dir + 'q-none'))
print(generate_one_command(False, False, False, True, False, False, job_root_dir + 'none-q'))

print(generate_one_command(False, True, False, False, False, False, job_root_dir + 'k-none'))
print(generate_one_command(False, False, False, False, True, False, job_root_dir + 'none-k'))

print(generate_one_command(False, False, True, False, False, False, job_root_dir + 'v-none'))
print(generate_one_command(False, False, False, False, False, True, job_root_dir + 'none-v'))

print(generate_one_command(True, True, False, False, False, False, job_root_dir + 'qk-none'))
print(generate_one_command(False, False, False, True, True, False, job_root_dir + 'none-qk'))

print(generate_one_command(True, True, True, False, False, False, job_root_dir + 'qkv-none'))
print(generate_one_command(False, False, False, True, True, True, job_root_dir + 'none-qkv'))
