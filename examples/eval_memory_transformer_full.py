from mmengine.config import read_base
from opencompass.models import FLAMemoryTransformer

with read_base():
    # GSM8K (baseline)
    from ..opencompass.configs.datasets.demo.demo_gsm8k_base_gen import gsm8k_datasets

    # LongBench v2
    from ..opencompass.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets

    # MMLU-Pro (0-shot COT)
    from ..opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets

    # RULER 4k (fits max_seq_len=4096)
    from ..opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets

    # NeedleBench 4k (fits max_seq_len=4096)
    from ..opencompass.configs.datasets.needlebench.needlebench_4k.needlebench_4k import needlebench_datasets

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

datasets = sum([
    gsm8k_datasets,
    LongBenchv2_datasets,
    mmlu_pro_datasets,
    ruler_datasets,
    needlebench_datasets,
], [])
