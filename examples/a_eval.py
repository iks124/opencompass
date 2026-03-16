from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    # longbenchv2
    from ..opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets

    # ceval
    from ..opencompass.configs.datasets.ceval.ceval_gen import \
        ceval_datasets

    # LV eval
    from ..opencompass.configs.datasets.lveval.lvevalcmrc_mixup.lveval_cmrc_mixup_gen import \
        LVEval_cmrc_mixup_datasets
    from ..opencompass.configs.datasets.lveval.lvevaldureader_mixup.lveval_dureader_mixup_gen import \
        LVEval_dureader_mixup_datasets
    from ..opencompass.configs.datasets.lveval.lvevalloogle_SD_mixup.lveval_loogle_SD_mixup_gen import \
        LVEval_loogle_SD_mixup_datasets

    # InfiniteBench
    from ..opencompass.configs.datasets.infinitebench.infinitebenchenqa.infinitebench_enqa_gen import \
        InfiniteBench_enqa_datasets
    from ..opencompass.configs.datasets.infinitebench.infinitebenchzhqa.infinitebench_zhqa_gen import \
        InfiniteBench_zhqa_datasets
"""from opencompass.configs.datasets.needlebench_v2.needlebench_v2_4k.needlebench_v2_4k import \
    needlebench_datasets"""
"""from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import \
    models as hf_qwen2_1_5b_instruct_models"""

datasets = [*InfiniteBench_enqa_datasets, *InfiniteBench_zhqa_datasets, *LVEval_cmrc_mixup_datasets, *LVEval_dureader_mixup_datasets,*LVEval_loogle_SD_mixup_datasets]
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='Qwen3.5-4B',
        path='/home/shihoukun/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        model_kwargs=dict(gpu_memory_utilization=0.85),
        tokenizer_path='/home/shihoukun/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        batch_size=4,
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=200000,
        max_out_len=10000,
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]



# python run.py examples/a_eval.py -w outputs/demo -a vllm
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py examples/a_eval.py -w outputs/demo -a vllm