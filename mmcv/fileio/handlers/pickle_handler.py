# Copyright (c) OpenMMLab. All rights reserved.
import io
import pickle
from contextlib import contextmanager

from .base import BaseFileHandler


@contextmanager
def _map_torch_cuda_storages_to_cpu_if_needed():
    """Load pickled CUDA tensors on CPU-only machines.

    Pickles produced by ``mmcv.dump`` may contain torch tensors. If any tensor
    storage was serialized from CUDA, ``pickle.load`` delegates storage restore
    to ``torch.storage._load_from_bytes``. On machines where CUDA is not
    available this raises before callers can move tensors to CPU.

    Keep the normal PyTorch behavior when CUDA is available. On CPU-only
    machines, temporarily map torch storages to CPU for this single load call.
    """
    try:
        import torch
    except ImportError:
        yield
        return

    if torch.cuda.is_available() or not hasattr(torch.storage, "_load_from_bytes"):
        yield
        return

    original_load_from_bytes = torch.storage._load_from_bytes

    def load_from_bytes_cpu(storage_bytes):
        buffer = io.BytesIO(storage_bytes)
        try:
            return torch.load(buffer, map_location=torch.device("cpu"), weights_only=False)
        except TypeError:
            buffer.seek(0)
            return torch.load(buffer, map_location=torch.device("cpu"))

    torch.storage._load_from_bytes = load_from_bytes_cpu
    try:
        yield
    finally:
        torch.storage._load_from_bytes = original_load_from_bytes


class PickleHandler(BaseFileHandler):

    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        with _map_torch_cuda_storages_to_cpu_if_needed():
            return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super(PickleHandler, self).load_from_path(
            filepath, mode='rb', **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('protocol', 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('protocol', 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super(PickleHandler, self).dump_to_path(
            obj, filepath, mode='wb', **kwargs)
