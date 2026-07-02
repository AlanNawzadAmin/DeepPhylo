import torch


def get_cuda_mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def get_mps_mem():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    else:
        raise ValueError("MPS backend not available")


def make_mem_hook(memfile, name, hook_type, steps, datatype):
    def hook_cuda(self, *args):
        alloc, reserved = get_cuda_mem()
        memfile.write(
            f"{name},{hook_type},{steps[-1]},{datatype[-1]},{alloc},{reserved}\n"
        )

    def hook_mps(self, *args):
        alloc, reserved = get_mps_mem()
        memfile.write(
            f"{name},{hook_type},{steps[-1]},{datatype[-1]},{alloc},{reserved}\n"
        )

    if torch.backends.mps.is_available():
        return hook_mps
    else:
        return hook_cuda
