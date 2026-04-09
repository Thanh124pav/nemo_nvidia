import copy 
from enum import Enum 
import random

import torch 

from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import BaseMoELayer 

ExpertsType = Enum('ExpertsType', ('SequentialMLP', 'TEGroupedMLP'))
ActivationFuncName = Enum('AcivationFuncName', ('gelu', 'silu', 'squared_relu'))
'''Note: 
1. All modules having `config` attribute 
2. All functions having '__name__' attribute
'''

def _get_keys_endswith(model, suffix):
    '''
    Get keys of model having specified suffix
    '''
    return [k  for k in model if k.endswith(suffix)]

def _find_submodule(model, submodule_name):
    '''
    Return submodule of model
    '''
    for name, submodule in model.named_modules():
        if name.endswith('.' + submodule_name) or name == submodule_name:
            return submodule

def _get_config(moe_model, dense_model): 
    '''
    get model config
    '''
    # MLP
    mlp  = _find_submodule(moe_model, 'mlp')
    assert mlp is not None, f"Can not find mlp module in {moe_model}"
    assert isinstance(mlp, BaseMoELayer), (
        f"This module is not inherited from BaseMoELayer"
    )
    num_local_experts = mlp.config.num_local_experts if hasattr(mlp.config, "num_local_experts") else mlp.config.num_moe_experts
    print(f"num local experts: {num_local_experts}")
    num_experts = mlp.config.num_moe_experts 
    gated_linear_unit = mlp.config.gated_linear_unit 
    moe_router_topk = mlp.config.moe_router_topk 
    moe_router_pre_softmax = mlp.config.moe_router_pre_softmax 
    moe_ffn_hidden_size = mlp.config.moe_ffn_hidden_size 
    moe_shared_expert_intermediate_size = mlp.config.moe_shared_expert_intermediate_size
    func_name = mlp.config.activation_func.__name__
    if func_name == 'gelu':
        activation_func_name = ActivationFuncName.gelu
    elif func_name == 'silu':
        activation_func_name = ActivationFuncName.silu 
    elif func_name == 'squared_relu':
        activation_func_name = ActivationFuncName.squared_relu
    else: 
        raise ValueError(
            f"Invalid activation function"
            f"Choices: {list(ActivationFuncName.__members__.keys())}"
            f"Got: {func_name}"
        )
    ep_rank = mlp.ep_group.rank()
    # print(f"ep_rank: {ep_rank}")

    # Experts
    experts = _find_submodule(mlp, 'experts')
    assert experts is not None, f"Can not find experts in {mlp}"
    if isinstance(experts, SequentialMLP):
        experts_type = ExpertsType.SequentialMLP
    elif isinstance(experts, TEGroupedMLP):
        experts_type = ExpertsType.TEGroupedMLP
    else:
        raise TypeError(f"the experts type {type(experts)} is not supported by upcycling.")
    
    # MLP in dense 
    dense_mlp = _find_submodule(dense_model, 'mlp')
    assert dense_mlp is not None, f"Can not find mlp module in {dense_model}"
    dense_ffn_hidden_size = dense_mlp.config.ffn_hidden_size

    assert dense_ffn_hidden_size % moe_ffn_hidden_size == 0, (
        "FFN hidden size of dense model must be divisible by "
        "FFN hidden size of moe model"
    )
    granularity = dense_ffn_hidden_size // moe_ffn_hidden_size 
    
    assert num_experts % granularity == 0, (
        "Number of experts must be divisible by "
        "granularity"
    )
    expansion_rate = num_experts // granularity
    # assert moe_shared_expert_intermediate_size % moe_ffn_hidden_size == 0, (
    #     "Shared experts size must be divisible by moe ffn hidden size"
    # )
    num_shared_experts = moe_shared_expert_intermediate_size // moe_ffn_hidden_size if moe_shared_expert_intermediate_size is not None else 0
    return (
        num_local_experts,
        moe_router_topk,
        granularity,
        expansion_rate, 
        experts_type,
        gated_linear_unit, 
        activation_func_name, 
        moe_router_pre_softmax, 
        ep_rank, 
        moe_shared_expert_intermediate_size, 
        num_shared_experts,
    )

def _convert_to_moe_state_dict(moe_model, dense_model):
    (
        num_local_experts,
        moe_router_topk,
        granularity,
        expansion_rate, 
        experts_type,
        gated_linear_unit, 
        activation_func_name, 
        moe_router_pre_softmax, 
        ep_rank, 
        moe_shared_expert_intermediate_size, 
        num_shared_experts,
    ) = _get_config(moe_model, dense_model)

    # def _process_router_param(value): # value là tensor? 
    #     value = value.data.data.clone()
    #     # Split tensor thành `granularity` phần theo dimension 0
    #     value = torch.tensor_split(value, granularity, dim=0)[0]
    #     value = [t.repeat(granularity, 1) for t in value]
    #     value = torch.cat(value, dim=0)
    #     return value
    
    def _get_moe_activation_scale(): 
        if moe_router_pre_softmax:
            # https://arxiv.org/abs/2410.07524 Ethan He 
            moe_activation_scale = (expansion_rate * granularity * granularity) / moe_router_topk
        else: 
            moe_activation_scale = granularity
        return moe_activation_scale
    
    def _get_weight_scale(): 
        moe_activation_scale = _get_moe_activation_scale()
        if gated_linear_unit == True: 
            scale = moe_activation_scale ** (1/3)
        elif activation_func_name == ActivationFuncName.squared_relu: 
            scale = moe_activation_scale ** (1/3) 
        else: 
            scale = moe_activation_scale ** (1/2) 
        return scale 

    def _process_fc1_weight_param(param, svd = True):
        param = param.clone()
        # print(f"param before upcycled: {param}")
        if svd:
            print("convert linear_fc1 using SVD")
            if activation_func_name == ActivationFuncName.silu and gated_linear_unit:
                def svd(A, k):
                    A_fp16 = A.float()
                    U, S, Vh = torch.linalg.svd(A_fp16, full_matrices=False)
                    Z = U[:, :k].T @ A_fp16   #
                    return Z.to(A.dtype)
                param_1, param_2 = torch.chunk(param, 2, dim=0)
                print(f"param 1's shape {param_1.shape}")
                k = int(param_1.shape[0]/granularity)
                param_1_svd = svd(param_1, k)
                param_2_svd = svd(param_2, k)
                params = [torch.cat([param_1_svd, param_2_svd], dim = 0) for i in range(granularity)]
                params = params * expansion_rate
                # print(f"param linear_fc1 after upcycled by SVD: {params}")
                return params
            else:
                print("Only support convert linear_fc1 using SVD with SwiGLU activation function, back to He and Khattar's method")
            
        scale = _get_weight_scale()
        param = param * scale 
        if activation_func_name == ActivationFuncName.silu and gated_linear_unit:
            param_1, param_2 = torch.chunk(param, 2, dim = 0)
            params_1 = torch.tensor_split(param_1, granularity, dim = 0)
            params_2 = torch.tensor_split(param_2, granularity, dim = 0)
            params = [torch.cat([params_1[i], params_2[i]], dim = 0) for i in range(granularity)]
        else:
            params = torch.tensor_split(param, granularity, dim = 0)
        params = params * expansion_rate 
        return params 
    def _process_fc1_bias_param(param): 
        params = _process_fc1_weight_param(param, svd=False)
        params = [tensor.squeeze(0) for tensor in params]
        return params 
    
    def _process_fc2_weight_param(param, svd = True):
        param = param.clone()
        # print(f"param before upcycled: {param}")
        if svd:
            print("Convert linear_fc2 using PCA")
            def randomized_svd(X_inp: torch.Tensor, k: int, n_oversamples: int = 10, n_iter: int = 2, center: bool = True):
                """
                Randomized SVD approximation (Halko et al.). Suitable when k << min(N,D).
                Returns same outputs as pca_svd but computed approximately.
                """
                X = X_inp.float()
                assert X.dim() == 2
                device = X.device
                dtype = X.dtype
                N, D = X.shape
            
                if center:
                    mu = X.mean(dim=0, keepdim=True)
                    Xc = X - mu
                else:
                    mu = torch.zeros(1, D, device=device, dtype=dtype)
                    Xc = X
            
                l = k + n_oversamples
                # 1) random Gaussian test matrix
                Omega = torch.randn(D, l, device=device, dtype=dtype)
                # 2) sample range: Y = Xc @ Omega  => [N, l]
                Y = Xc @ Omega
            
                # power iterations (optional, improves accuracy for decaying singular values)
                for _ in range(n_iter):
                    Y = Xc @ (Xc.T @ Y)
            
                # 3) orthonormalize Y -> Q (N x l)
                Q, _ = torch.linalg.qr(Y, mode='reduced')
            
                # 4) B = Q^T Xc  => (l x D)
                B = Q.T @ Xc
            
                # 5) svd on small matrix B
                Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)  # Ub:[l,l], Vh:[l,D]
                # approximate U = Q @ Ub
                U_approx = Q @ Ub[:, :k]
                S_k = S[:k]
                Vh_k = Vh[:k, :]   # [k, D]
            
                components = Vh_k.T.contiguous()    # [D, k]
                X_proj = Xc @ components            # [N, k]
                return X_proj.to(X_inp.dtype)
            if activation_func_name == ActivationFuncName.silu and gated_linear_unit:
                k = int(param.shape[1] / granularity)
                param = randomized_svd(param, k)
                params = [param for i in range(granularity)]
                params = params * expansion_rate
                # print(f"param linear_fc2 after upcycled by SVD: {params}")
                return params
            else: 
                print("Only support convert linear_fc2 using PCA with SwiGLU, back to He and Khattar's method")
        scale = _get_weight_scale()
        param = param * scale
        params = torch.tensor_split(param, granularity, dim=1)
        params = params * expansion_rate
        return params

    def _process_fc2_bias_param(param):
        param = param.clone()
        params = param.repeat(granularity * expansion_rate, 1)
        return params
    
    dense_state_dict = copy.deepcopy(dense_model.state_dict())
    moe_state_dict = copy.deepcopy(moe_model.state_dict())
    for key in dense_state_dict.keys() & moe_state_dict.keys():
        moe_state_dict[key] = dense_state_dict[key]

    def _convert_key_value(
        dist_dict = moe_state_dict, 
        src_dict  = dense_state_dict, 
        key_replace_old = None, 
        key_replace_new = None, 
        value_process_func = lambda x:x 
    ):
        keys = _get_keys_endswith(src_dict, key_replace_old)
        for key in keys: 
            value = src_dict[key]
            new_value = value_process_func(value)
            new_key = key.replace(key_replace_old, key_replace_new)
            dist_dict[new_key] = new_value.clone() if hasattr(new_value, 'clone') else new_value
        return 
    
    _convert_key_value(
        key_replace_old='mlp.linear_fc1.layer_norm.weight', 
        key_replace_new='pre_mlp_layernorm.weight',
    )
    _convert_key_value(
        key_replace_old='mlp.linear_fc1.layer_norm_bias',
        key_replace_new='pre_mlp_layernorm.bias',
    )
    _convert_key_value(
        src_dict=moe_state_dict,
        key_replace_old='mlp.router.weight',
        key_replace_new='mlp.router.weight',
    )

    def _expand_key_value(
        dist_dict = moe_state_dict, 
        src_dict = dense_state_dict, 
        key_replace_old = None, 
        key_replace_new = None, 
        value_process_func = lambda x:x , 
        num_local_experts = num_local_experts,
    ):
        keys = _get_keys_endswith(src_dict, key_replace_old)
        for key in keys: 
            param = src_dict[key]
            # if 'mlp.linear_fc1.weight' or 'mlp.linear_fc2.weight' in key:
                # print(f"param of {key} to be converted {param}\n")
            params = value_process_func(param)
            # if 'mlp.linear_fc1.weight' or 'mlp.linear_fc2.weight' in key:
                # print(f"param of {key} after converted: {params}")
            # print(f"num_local_experts: {num_local_experts}")
            for idx in range(num_local_experts):
                new_key = key.replace(key_replace_old, key_replace_new).format(idx)
                dist_dict[new_key] = params[ep_rank * num_local_experts + idx]
        return 
    
    if experts_type == ExpertsType.SequentialMLP: 
        _expand_key_value(
            key_replace_old='mlp.linear_fc1.weight',
            key_replace_new='mlp.experts.local_experts.{}.linear_fc1.weight', 
            value_process_func= _process_fc1_weight_param,
        )
        _expand_key_value(
            key_replace_old='mlp.linear_fc1.bias', 
            key_replace_new='mlp.experts.local_experts.{}.linear_fc1.bias',
            value_process_func=_process_fc1_bias_param,
        )
        _expand_key_value(
            key_replace_old='mlp.linear_fc2.weight',
            key_replace_new='mlp.experts.local_experts.{}.linear_fc2.weight',
            value_process_func=_process_fc2_weight_param,
        )
        _expand_key_value(
            key_replace_old='mlp.linear_fc2.bias',
            key_replace_new='mlp.experts.local_experts.{}.linear_fc2.bias',
            value_process_func=_process_fc2_bias_param,
        )
    elif experts_type == ExpertsType.TEGroupedMLP:
        _expand_key_value(
            key_replace_old='mlp.linear_fc1.weight',
            key_replace_new='mlp.experts.linear_fc1.weight{}',
            value_process_func=_process_fc1_weight_param,
        )
        _expand_key_value(
            key_replace_old='mlp.linear_fc2.weight',
            key_replace_new='mlp.experts.linear_fc2.weight{}',
            value_process_func=_process_fc2_weight_param,
        )
    else:
        raise ValueError(f"unknown moe weight format {experts_type}")

    return moe_state_dict
def upcycle_state_dict(moe_model, dense_model):
    state_dict = {}
    if len(moe_model) == 1:
        assert len(dense_model) == 1, "Number of dense model must equal to number of moe models"
        state_dict['model'] = _convert_to_moe_state_dict(moe_model[0], dense_model[0])
    else: 
        assert len(moe_model) == len(dense_model)
        for i in range(len(moe_model)):
            state_dict['model%d' % i] = _convert_to_moe_state_dict(
                dense_model[i].state_dict(), moe_model[i]
            )
    return state_dict 



