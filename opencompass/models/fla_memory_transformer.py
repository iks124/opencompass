from typing import Optional

from opencompass.models.huggingface_above_v4_33 import HuggingFaceBaseModel
from opencompass.registry import MODELS


@MODELS.register_module()
class FLAMemoryTransformer(HuggingFaceBaseModel):
    """HuggingFaceBaseModel wrapper for FLA MemoryTransformerForCausalLM.

    Imports fla.models.memory_transformer inside _load_model so that
    MemoryTransformerForCausalLM is registered with AutoModelForCausalLM
    before the checkpoint is loaded (mmengine lazy-parses config files, so
    a top-level import in the config file never actually executes).
    """

    def _load_model(self,
                    path: str,
                    kwargs: dict,
                    peft_path: Optional[str] = None,
                    peft_kwargs: dict = dict()):
        import fla.models.memory_transformer  # noqa: F401
        super()._load_model(path, kwargs, peft_path, peft_kwargs)
