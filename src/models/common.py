import MinkowskiEngine as ME
import torch
import torch.nn as nn


def conv(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    dilation=1,
    bias=False,
    region_type=0,
    dimension=3,
):
    if not isinstance(region_type, ME.RegionType):
        if region_type == 0:
            region_type = ME.RegionType.HYPER_CUBE
        elif region_type == 1:
            region_type = ME.RegionType.HYPER_CROSS
        else:
            raise ValueError("Unsupported region type")

    kernel_generator = ME.KernelGenerator(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        region_type=region_type,
        dimension=dimension,
    )

    return ME.MinkowskiConvolution(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=dimension,
    )


def conv_tr(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    bias=False,
    region_type=ME.RegionType.HYPER_CUBE,
    dimension=-1,
):
    assert dimension > 0, "Dimension must be a positive integer"
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        is_transpose=True,
        region_type=region_type,
        dimension=dimension,
    )

    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=dimension,
    )


def get_norm(norm_type, num_feats, bn_momentum=0.05, dimension=-1):
    if norm_type == "BN":
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == "IN":
        return ME.MinkowskiInstanceNorm(num_feats)
    elif norm_type == "INBN":
        return nn.Sequential(
            ME.MinkowskiInstanceNorm(num_feats),
            ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum),
        )
    else:
        raise ValueError(f"Type {norm_type}, not defined")


def get_nonlinearity(non_type):
    if non_type == "ReLU":
        return ME.MinkowskiReLU()
    elif non_type == "ELU":
        return ME.MinkowskiELU()
    else:
        raise ValueError(f"Type {non_type}, not defined")
