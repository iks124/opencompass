from mmengine.config import read_base
from opencompass.models import FLAMemoryTransformer

with read_base():
    from ..opencompass.configs.datasets.demo.demo_gsm8k_base_gen import gsm8k_datasets

models = [
    dict(
        type=FLAMemoryTransformer,
        abbr='memory-transformer-660m',
        path='/home/shihoukun/checkpoints/memory_transformer_660m_init',
        max_seq_len=4096,
        max_out_len=512,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(device_map='cuda:0'),
    )
]

datasets = gsm8k_datasets
