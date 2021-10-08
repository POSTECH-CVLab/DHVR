import random

import numpy as np
import torch
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T


class Compose:
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, coords, feats):
    for transform in self.transforms:
      coords, feats = transform(coords, feats)
    return coords, feats


class Jitter:
  def __init__(self, mu=0, sigma=0.01):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats += self.sigma * torch.randn(feats.shape[0], feats.shape[1])
      if self.mu != 0:
        feats += self.mu
    return coords, feats


class ChromaticShift:
  def __init__(self, mu=0, sigma=0.1):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats[:, :3] += torch.randn(self.mu, self.sigma, (1, 3))
    return coords, feats
