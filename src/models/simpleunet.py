import gin
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch
from src.models.common import conv, conv_tr, get_norm


@gin.configurable()
class SimpleNetBase(ME.MinkowskiNetwork):
    NORM_TYPE = None
    CHANNELS = [None, 32, 64, 128]
    TR_CHANNELS = [None, 32, 32, 64]

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        bn_momentum=0.1,
        conv1_kernel_size=3,
        normalize_feature=False,
        D=3,
    ):
        super(SimpleNetBase, self).__init__(D)
        NORM_TYPE = self.NORM_TYPE
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        self.normalize_feature = normalize_feature
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.norm1 = get_norm(
            NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D
        )

        self.conv2 = conv(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.norm2 = get_norm(
            NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D
        )

        self.conv3 = conv(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.norm3 = get_norm(
            NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D
        )

        self.conv3_tr = conv_tr(
            in_channels=CHANNELS[3],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.norm3_tr = get_norm(
            NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D
        )

        self.conv2_tr = conv_tr(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.norm2_tr = get_norm(
            NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D
        )

        self.conv1_tr = conv(
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.norm1_tr = get_norm(
            NORM_TYPE, TR_CHANNELS[1], bn_momentum=bn_momentum, dimension=D
        )

        self.final = conv(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D,
        )

    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out = MEF.relu(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out = MEF.relu(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out = MEF.relu(out_s4)

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, out_s2)

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out_s1_tr = MEF.relu(out)

        out = ME.cat(out_s1_tr, out_s1)
        out = self.conv1_tr(out)
        out = self.norm1_tr(out)
        out = MEF.relu(out)

        out = self.final(out)

        if self.normalize_feature:
            return ME.SparseTensor(
                out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
                coords_key=out.coords_key,
                coords_manager=out.coords_man,
            )
        else:
            return out


class SimpleNetIN(SimpleNetBase):
    NORM_TYPE = "IN"


class SimpleNetBN(SimpleNetBase):
    NORM_TYPE = "BN"


class SimpleNetBNE(SimpleNetBN):
    CHANNELS = [None, 16, 32, 32]
    TR_CHANNELS = [None, 16, 16, 32]


class SimpleNetINE(SimpleNetBNE):
    NORM_TYPE = "IN"
