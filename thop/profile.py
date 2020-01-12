import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
import warnings
from pandas import DataFrame

from distutils.version import LooseVersion

from .count_hooks import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
    logger.warning(
        "You are using an old version PyTorch {version}, which THOP is not going to support in the future.".format(
            version=torch.__version__))

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.LayerNorm: count_bn,
    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.Embedding: emb_ops,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample
}


def profile(model, inputs, want_op_file=False, custom_ops=None, verbose=True):
    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logger.warning("Either .total_ops or .total_params is already defined in %s. "
                           "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops = 0
    total_params = 0

    mynn={"Layer Name":[],"Input Features":[], "Output Features":[], "Dict Size of Emb":[], "Emb Vector Size":[], "Norm Size":[], "FLOPs":[]}
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        strr=str(m)
        layer_name=''
        for ch in strr:
            if ch=='(':
                break
            layer_name=layer_name+ch
        mynn["Layer Name"].append(layer_name)

        if hasattr(m, "in_features"):
            mynn["Input Features"].append(str(m.in_features))
        else:
            mynn["Input Features"].append("-")

        if hasattr(m, "out_features"):
            mynn["Output Features"].append(str(m.out_features))
        else:
            mynn["Output Features"].append("-")

        if hasattr(m, "num_embeddings"):
            mynn["Dict Size of Emb"].append(str(m.num_embeddings))
        else:
            mynn["Dict Size of Emb"].append("-")

        if hasattr(m, "embedding_dim"):
            mynn["Emb Vector Size"].append(str(m.embedding_dim))
        else:
            mynn["Emb Vector Size"].append("-")

        if hasattr(m, "normalized_shape"):
            mynn["Norm Size"].append(str(m.normalized_shape[0]))
        else:
            mynn["Norm Size"].append("-")

        mynn["FLOPs"].append(str(m.total_ops.item()))
        total_ops += m.total_ops
        total_params += m.total_params
        
    df = DataFrame(mynn, columns= ["Layer Name","Input Features","Output Features","Dict Size of Emb","Emb Vector Size","Norm Size","FLOPs"])
    
    if want_op_file==True:
        export_csv = df.to_csv (r'output_file.csv', index = None, header=True)
    else:
        print(df)
    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")
#     if want_op_file == True:
#         return total_ops, total_params, op_file
#     else:
    return total_ops, total_params
