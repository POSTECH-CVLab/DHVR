import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from faiss.contrib.torch_utils import (
    swig_ptr_from_FloatTensor,
    swig_ptr_from_IndicesTensor,
)

res = faiss.StandardGpuResources()


def _hash(arr, M=None):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M ** d
        else:
            hash_vec += arr[d] * M ** d
    return hash_vec


def intersection(source, target, exclusive=False):
    hash_seed = max(source.shape[0], target.shape[0])
    key_source = _hash(source.numpy(), hash_seed)
    key_target = _hash(target.numpy(), hash_seed)
    mask = np.isin(key_target, key_source, assume_unique=False)

    if exclusive:
        mask = np.logical_not(mask)
    return mask


def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_IndicesTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr, k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I


def find_knn_gpu(query, ref, knn=1, nn_max_n=None, return_distance=False):
    assert (
        ref.shape[-1] == query.shape[-1]
    ), f"Reference dataset and Query dataset have different channel dimension, {ref.shape[-1]} != {query.shape[-1]}"
    ch = ref.shape[-1]
    index_cpu = faiss.IndexFlatL2(ch)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    torch.cuda.synchronize()
    ref_ptr = swig_ptr_from_FloatTensor(ref)
    index.add_c(ref.shape[0], ref_ptr)
    dist, indx = search_index_pytorch(index, query, k=knn)
    torch.cuda.empty_cache()
    if return_distance:
        return dist, indx
    else:
        return indx


def find_radius_gpu(F0, F1, len_batch, radius=None, return_row_splits=False):
    """
    !! This function requires custom open3d !!
    """
    try:
        import open3d.ml.torch as ml

        assert (
            F0.shape[1] == F1.shape[1] == 3
        ), "find_radius_gpu function only support 3D data."

        len_batch = torch.Tensor(len_batch).long()
        cum_len_batch = len_batch.cumsum(0)
        queries_row_splits = torch.zeros(cum_len_batch.shape[0] + 1).long()
        queries_row_splits[1:] = cum_len_batch[:, 0]
        points_row_splits = torch.zeros(cum_len_batch.shape[0] + 1).long()
        points_row_splits[1:] = cum_len_batch[:, 1]

        table = ml.ops.build_spatial_hash_table(
            F1,
            radius,
            points_row_splits=points_row_splits,
            hash_table_size_factor=1 / 32,
        )

        neighbor_index, neighbor_row_splits, _ = ml.ops.fixed_radius_search(
            F1,
            F0,
            radius,
            points_row_splits=points_row_splits,
            queries_row_splits=queries_row_splits,
            ignore_query_point=True,
            **table._asdict(),
        )
        if return_row_splits:
            return neighbor_index.cpu(), neighbor_row_splits.cpu()
        else:
            return neighbor_index.cpu()

    except ImportError:
        print(
            "unable to import open3d.ml.torch. Please compile open3d with torch ml module enabled."
        )


def feature_matching(F0, F1, knn=1, mutual=True):
    nn_index = find_knn_gpu(F0, F1, return_distance=False)
    matches = torch.cat(
        [torch.arange(nn_index.shape[0]).unsqueeze(1), nn_index.cpu()], knn
    )
    if mutual:
        nn_index_reverse = find_knn_gpu(F1, F0)
        matches_reverse = torch.cat(
            [
                nn_index_reverse.cpu(),
                torch.arange(nn_index_reverse.shape[0]).unsqueeze(1),
            ],
            1,
        )
        hash_seed = max(matches.shape[0], matches_reverse.shape[0])
        key = _hash(matches.numpy(), hash_seed)
        key_reverse = _hash(matches_reverse.numpy(), hash_seed)
        intersection = np.isin(key, key_reverse, assume_unique=False)
        matches = matches[intersection]

    torch.cuda.empty_cache()
    return matches
