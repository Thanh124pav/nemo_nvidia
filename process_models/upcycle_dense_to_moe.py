import os
os.environ['LOCAL_RANK'] = '0'

r"""
Conversion script to convert NeMo Mistral-7B checkpoints into HuggingFace checkpoint.
  Example to run this conversion script:
torchrun --nproc_per_node=1 process_models/upcycle_dense_to_moe.py \
        --model models/tiny_gpt.nemo \
        --num-experts 32 \
        --moe-ffn-dim 64 \
        --moe-router-topk 8 \
        --output-path models/tiny_gpt_moe.nemo
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn
from megatron.core import parallel_state
import torch.distributed as dist 
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils import logging
from nemo.lightning import MegatronStrategy, Trainer, _strategy_lib
import tempfile
from pathlib import Path
from upcycling_utils_upgrade import upcycle_state_dict

# Try importing distributed checkpoint utilities with fallbacks
try:
    from nemo.lightning.io.pl import TrainerContext, ckpt_to_context_subdir, ckpt_to_weights_subdir
except ImportError:
    # Fallback for older versions
    TrainerContext = None
    ckpt_to_context_subdir = None
    ckpt_to_weights_subdir = None

try:
    from nemo.lightning.pytorch.plugins.mixed_precision import bf16_mixed
except ImportError:
    from nemo.lightning.pytorch.plugins import MegatronMixedPrecision
    def bf16_mixed():
        return MegatronMixedPrecision(precision="bf16-mixed")

try:
    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject
except ImportError:
    # Fallback implementations
    def dict_list_map_inplace(func, obj):
        pass
    LocalNonpersistentObject = None
    ShardedObject = None


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None, required=True, help="Path to NeMo checkpoint")
    parser.add_argument(
        "--output-path", type=str, default='', required=False, help="Path to NeMo save upcycled checkpoint"
    )
    parser.add_argument(
        "--moe-ffn-dim", type = int, default=1024, required=True, help= "hidden dim of MoE FFN"
    )
    parser.add_argument(
        "--num-experts", type=int, default=8, required=True, help="Number of experts to use in upcycled model."
    )
    parser.add_argument(
        "--moe-router-pre-softmax", type=bool, default=True, help="Use softmax then topK or topK then softmax"
    )
    parser.add_argument(
        "--moe-router-topk", type=int, default=2, help="Number of experts chosen each inference step"
    )
    parser.add_argument(
        "--moe_shared_expert_intermediate_size", type=int, default=None, help = "hidden dim of shared experts"
    )

    args = parser.parse_args()
    assert isinstance(args.num_experts, int)
    assert args.num_experts > 1, "Expected --num-experts to be greater-than 1."
    if args.output_path == '':
        args.output_path = args.model + f'_e{args.num_experts}.nemo'
    return args


def make_moe_config_from_dense(config, args):
    """
    Các tham số có thể điều chỉnh:
        num_local_experts = mlp.num_local_experts
        num_experts = mlp.config.num_moe_experts
        gated_linear_unit = mlp.config.gated_linear_unit
        moe_router_topk = mlp.config.moe_router_topk
        moe_router_pre_softmax = mlp.config.moe_router_pre_softmax
        moe_ffn_hidden_size = mlp.config.moe_ffn_hidden_size
        moe_shared_expert_intermediate_size = mlp.config.moe_shared_expert_intermediate_size
    """
    from copy import deepcopy

    moe_config = deepcopy(config)
    moe_config['num_moe_experts'] = args.num_experts
    moe_config['moe_ffn_hidden_size'] = args.moe_ffn_dim
    moe_config['moe_router_pre_softmax'] = args.moe_router_pre_softmax
    moe_config['moe_router_topk'] = args.moe_router_topk
    moe_config['moe_shared_expert_intermediate_size'] = args.moe_shared_expert_intermediate_size

    return moe_config


def unwrap(model):
    tmp = model
    while hasattr(tmp, 'module'):
        tmp = tmp.module
    return tmp


def upcycle(args, cpu_only=True) -> None:
    """
    Upcycle dense checkpoint to MoE.
    """
    in_file = args.model
    logging.info(f'Loading NeMo checkpoint from: {in_file}')

    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())

    # Load dense model
    model_config = MegatronGPTModel.restore_from(in_file, trainer=dummy_trainer, return_config=True)
    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = 1
    model_config.sequence_parallel = False
    if cpu_only:
        map_location = torch.device('cpu')
        model_config.use_cpu_initialization = True
    else:
        map_location = None
    model_config.perform_initialization = False
    dense_model = MegatronGPTModel.restore_from(
        in_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )

    # Make upcycled config
    moe_config = make_moe_config_from_dense(model_config, args )
    print(moe_config)
    # quit()
    dummy_trainer2 = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    moe_model = MegatronGPTModel(moe_config, trainer=dummy_trainer2)

    moe_state_dict = upcycle_state_dict([unwrap(moe_model.model)], [unwrap(dense_model.model)])
    #print(moe_state_dict)
    # Add module attribute to MoE model if it doesn't exist (compatibility fix)
    if not hasattr(moe_model, 'module'):
        moe_model.module = moe_model.model
    
    moe_model.model.load_state_dict(moe_state_dict['model'])  # Fixed indexing
    with open("moe_state_dict.txt", "w", encoding="utf-8") as f: 
        f.write(str(moe_state_dict))
    
    return moe_model, moe_state_dict


def save_distributed_checkpoint(model, output_path):
    with open("moe_weights_after_upcycled.txt", "w", encoding='utf-8') as f:
        for name, parameters in model.named_parameters():
            # if 'linear_fc1.weight' in name or 'linear_fc2.weight' in name:
                # print(f"module {name} - shape: {parameters.shape} - value:\n: {parameters.data}")
            f.write(f"module {name} - shape: {parameters.shape} - value:\n: {parameters.data}")
    model._save_restore_connector = NLPSaveRestoreConnector()
    if not output_path.endswith('.nemo'):
        output_path = output_path + '.nemo'
    model.save_to(output_path)


if __name__ == '__main__':
    args = get_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    logging.info("Starting dense to MoE upcycling...")
    if not parallel_state.is_initialized():
        dist.init_process_group(backend='nccl')
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size = 1,
            pipeline_model_parallel_size = 1,
            expert_model_parallel_size = 1,
        )
    # Upcycle the model
    moe_model, state_dict = upcycle(args)
    
    # Save as distributed checkpoint
    save_distributed_checkpoint(moe_model, args.output_path)

    logging.info(f'Upcycling complete! Output saved to: {args.output_path}')

