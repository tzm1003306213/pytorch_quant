""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim

# from .adafactor import Adafactor
# from .adahessian import Adahessian
# from .adamp import AdamP
# from .lookahead import Lookahead
# from .nadam import Nadam
# from .novograd import NovoGrad
# from .nvnovograd import NvNovoGrad
# from .radam import RAdam
from .rmsprop_tf import RMSpropTF
# from .sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), rcf_name='alpha'):
    decay = []
    no_decay = []
    rcf = []
    for name, param in model.named_parameters():
        if rcf_name in name:
            rcf.append(param)
            continue
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return ([{'params': no_decay, 'weight_decay': 0.},
             {'params': decay, 'weight_decay': weight_decay}],
            [{'params': rcf, 'weight_decay': 0.}])


def create_optimizer(args, model, filter_bias_and_bn=True):
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay
        parameters, _ = add_weight_decay(model, weight_decay, skip)
    else:
        parameters = model.parameters()

    return create_optimizer_param(args, parameters)


def create_optimizer_rcf(args, model, rcf_name='alpha'):
    weight_decay = args.weight_decay
    if not weight_decay:
        weight_decay = 0.
    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay
    parameters, parameters_rcf = add_weight_decay(model, weight_decay, skip, rcf_name=rcf_name)

    return create_optimizer_param(args, parameters), optim.Adam(parameters_rcf, lr=args.lr_rcf)


def create_optimizer_param(args, parameters):
    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_lower = args.opt.lower()
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    # elif opt_lower == 'nadam':
    #     optimizer = Nadam(parameters, **opt_args)
    # elif opt_lower == 'radam':
    #     optimizer = RAdam(parameters, **opt_args)
    # elif opt_lower == 'adamp':        
    #     optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    # elif opt_lower == 'sgdp':        
    #     optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    # elif opt_lower == 'adafactor':
    #     if not args.lr:
    #         opt_args['lr'] = None
    #     optimizer = Adafactor(parameters, **opt_args)
    # elif opt_lower == 'adahessian':
    #     optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    # elif opt_lower == 'novograd':
    #     optimizer = NovoGrad(parameters, **opt_args)
    # elif opt_lower == 'nvnovograd':
    #     optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    # if len(opt_split) > 1:
    #     if opt_split[0] == 'lookahead':
    #         optimizer = Lookahead(optimizer)

    return optimizer
