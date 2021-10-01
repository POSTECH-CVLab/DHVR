import MinkowskiEngine as ME
import torch

eps = 1e-8


def is_rotation(R):
    assert R.shape == (
        3,
        3,
    ), f"rotation matrix should be in shape (3, 3) but got {R.shape} input."
    rrt = R @ R.t()
    I = torch.eye(3)
    err = torch.norm(I - rrt)
    return err < eps


def skew_symmetric(vectors):
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)

    r00 = torch.zeros_like(vectors[:, 0])
    r01 = -vectors[:, 2]
    r02 = vectors[:, 1]
    r10 = vectors[:, 2]
    r11 = torch.zeros_like(r00)
    r12 = -vectors[:, 0]
    r20 = -vectors[:, 1]
    r21 = vectors[:, 0]
    r22 = torch.zeros_like(r00)

    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(
        -1, 3, 3
    )
    return R


def axis_angle_to_rotation(axis_angles):
    if axis_angles.dim() == 1:
        axis_angles = axis_angles.unsqueeze(0)

    angles = torch.norm(axis_angles, p=2, dim=-1, keepdim=True)
    axis = axis_angles / angles

    K = skew_symmetric(axis)
    K_square = torch.bmm(K, K)
    I = torch.eye(3).to(axis_angles.device).repeat(K.shape[0], 1, 1)

    R = (
        I
        + torch.sin(angles).unsqueeze(-1) * K
        + (1 - torch.cos(angles).unsqueeze(-1)) * K_square
    )

    return R.squeeze(0)


def rotation_to_axis_angle(R):
    if R.dim() == 2:
        R = R.unsqueeze(0)

    theta = torch.acos(((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2 + eps)
    sin_theta = torch.sin(theta)

    singular = torch.zeros(3, dtype=torch.float32).to(theta.device)

    multi = 1 / (2 * sin_theta + eps)
    rx = multi * (R[:, 2, 1] - R[:, 1, 2]) * theta
    ry = multi * (R[:, 0, 2] - R[:, 2, 0]) * theta
    rz = multi * (R[:, 1, 0] - R[:, 0, 1]) * theta

    axis_angles = torch.stack((rx, ry, rz), dim=-1)
    singular_indices = torch.logical_or(sin_theta == 0, sin_theta.isnan())
    axis_angles[singular_indices] = singular

    return axis_angles.squeeze(0)


def gaussianNd(kernel_size=5, dimension=3):
    dim = [kernel_size] * dimension
    siz = torch.LongTensor(dim)
    sig_sq = (siz.float() / 2 / 2.354).pow(2)
    siz2 = (siz - 1) // 2

    axis = torch.meshgrid(
        [torch.arange(-siz2[i], siz2[i] + 1) for i in range(siz.shape[0])]
    )
    gaussian = torch.exp(
        -torch.stack(
            [axis[i].float().pow(2) / 2 / sig_sq[i] for i in range(sig_sq.shape[0])],
            dim=0,
        ).sum(dim=0)
    )
    gaussian = gaussian / gaussian.sum()
    return gaussian


def sparse_gaussian(data, kernel_size=5, dimension=3):
    # prepare input sparse tensor
    if isinstance(data, ME.SparseTensor):
        sinput = data
    else:
        raise TypeError()

    # build gaussian kernel weight
    hsfilter = gaussianNd(kernel_size, dimension).to(data.device)

    # prepare conv layer
    conv = ME.MinkowskiConvolution(
        in_channels=1, out_channels=1, kernel_size=kernel_size, dimension=dimension
    )
    with torch.no_grad():
        conv.kernel.data = hsfilter.reshape(-1, 1, 1)

    # forward
    out = conv(sinput)
    return out
