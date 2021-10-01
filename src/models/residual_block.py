import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch
import torch.nn as nn
from src.models.common import conv, conv_tr, get_nonlinearity, get_norm


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = "BN"

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        bn_momentum=0.1,
        region_type=0,
        D=3,
    ):
        super(BasicBlockBase, self).__init__()

        self.conv1 = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            region_type=region_type,
            dimension=D,
        )
        self.norm1 = get_norm(
            self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=D
        )
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            region_type=region_type,
            dimension=D,
        )
        self.norm2 = get_norm(
            self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=D
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class BasicBlockBN(BasicBlockBase):
    NORM_TYPE = "BN"


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = "IN"


class BasicBlockINBN(BasicBlockBase):
    NORM_TYPE = "INBN"


def get_block(
    norm_type,
    inplanes,
    planes,
    stride=1,
    dilation=1,
    downsample=None,
    bn_momentum=0.1,
    region_type=0,
    dimension=3,
):
    if norm_type == "BN":
        Block = BasicBlockBN
    elif norm_type == "IN":
        Block = BasicBlockIN
    elif norm_type == "INBN":
        Block = BasicBlockINBN
    elif norm_type == "SE":
        Block = SEBlock
    else:
        raise ValueError(f"Type {norm_type}, not defined")

    return Block(
        inplanes,
        planes,
        stride,
        dilation,
        downsample,
        bn_momentum,
        region_type,
        dimension,
    )


def conv_norm_non(
    inc,
    outc,
    kernel_size,
    stride,
    dimension,
    bn_momentum=0.05,
    region_type=ME.RegionType.HYPER_CUBE,
    norm_type="BN",
    nonlinearity="ELU",
):
    return nn.Sequential(
        conv(
            in_channels=inc,
            out_channels=outc,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
            bias=False,
            region_type=region_type,
            dimension=dimension,
        ),
        get_norm(norm_type, outc, bn_momentum=bn_momentum, dimension=dimension),
        get_nonlinearity(nonlinearity),
    )


class SEBlock(nn.Module):
    expansion = 1
    NORM_TYPE = "BN"

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        bn_momentum=0.1,
        region_type=0,
        D=3,
    ):
        super(SEBlock, self).__init__()

        self.conv1 = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            region_type=region_type,
            dimension=D,
        )
        self.norm1 = get_norm(
            self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=D
        )
        self.squeeze = ME.MinkowskiGlobalSumPooling()
        self.fc1 = ME.MinkowskiConvolution(
            in_channels=planes,
            out_channels=int(planes / 4),
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.fc2 = ME.MinkowskiConvolution(
            in_channels=int(planes / 4),
            out_channels=planes,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            region_type=region_type,
            dimension=D,
        )
        self.norm2 = get_norm(
            self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=D
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        se = self.squeeze(out)
        se = self.fc1(se)
        se = MEF.relu(se)
        se = self.fc2(se)
        se = MEF.sigmoid(se)
        feats = []
        batch_size = len(se.decomposed_features)
        for i in range(batch_size):
            F = out.features_at(i)
            scale = se.features_at(i)
            feats.append(F * scale)
        feats = torch.cat(feats, 0)
        out = ME.SparseTensor(
            feats,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
        )

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)
        return out
