from .SSAN_R import SSAN_R


def get_model(max_iter):
    return SSAN_R(max_iter=max_iter)
