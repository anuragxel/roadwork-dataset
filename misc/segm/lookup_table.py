import torch
import numpy as np


# LEXICON
# Lexicon:
# ar_?_: array
# t?_?_: tensor
# th_?_: tensor explicitly on host (cpu)
# td_?_: tensor on a specific device (can also be the host)
# td_i_: tensor of int32 type ("i") on a specific device
# td_u_: tensor of uint8 type ("u") on a specific device


def get_lookup_table(
    ar_u_key: np.ndarray,
    ar_u_val: np.ndarray,
    device,
    v_val_default = 0,
):
    """
    Creates a lookup table built from arrays ar_u_key and ar_u_value of shapes
    [n_key, n_dim_key] and [n_key, n_dim_val] respectively

    Each [key, value] pair at index i in the lookup table lut is assigned as:
        ar_u_lut[k_encoded] = ar_u_val[i]
    where k_encoded is the encoding of the n-dimensional key ar_u_key[i]:
        l_key_encoded = [
            ar_u_key[i, 0] * (256 ** 0)
            + ar_u_key[i, 1] * (256 ** 1)
            + ...
            + ar_u_key[i, n - 1] * (256 ** (n - 1))
        ]

    Example:
        For [R, G, B] keys values are multiplied by [256^0, 256^1, 256^2]
        respectively
        The endoded key is therefore a 3-byte value where:
            - the 8 rightmost bits correspond to Red
            - the 8 middle bits correspond to Green
            - the 8 leftmost bits correspond to Blue

    Args:
        ar_u_key (numpy.ndarray): array of keys
            Must have shape [n_key, n_dim_key]
        ar_u_val (numpy.ndarray): array of values
            Must have shape [n_key, n_dim_val]
        v_val_default: padding value for undefined keys

    Returns:
        (numpy.ndarray): numpy lookup table with shape [256 ** n_dim_key, n_dim_val]
        (torch.Tensor): pytorch lookup table
    """
    # validate compatibility between keys and values
    n_key = ar_u_key.shape[0]
    n_val = ar_u_val.shape[0]
    if n_key != n_val:
        raise ValueError(
            f"Number of keys and values should be the same:"
            f"\n- ar_u_key.shape[0] = {ar_u_key.shape[0]} (n_key)"
            f"\n- ar_u_val.shape[0] = {ar_u_val.shape[0]} (n_val)"
        )
    if ar_u_key.dtype != np.uint8 or ar_u_val.dtype != np.uint8:
        raise ValueError(
            f"Both ar_u_key and ar_u_val should be of dtype np.uint8:"
            f"\n- ar_u_key.dtype = {ar_u_key.dtype}"
            f"\n- ar_u_val.dtype = {ar_u_val.dtype}"
        )
    # for each index in ar_u_key corresponds a value in ar_u_val
    # for indexes not in ar_u_key, the corresponding value is set to v_default
    n_chn_key = ar_u_key.shape[1]
    n_chn_val = ar_u_val.shape[1]
    # encode colors into lookup keys
    # requires a multiplier for each channel: [256^0, 256^1, ..., 256^n_chn]
    ar_i_code = np.zeros(shape = [1, n_chn_key], dtype = np.int32)
    ar_i_code[[0]] = 256 ** np.arange(n_chn_key)
    # convert keys to int32 for computation of indices above 255
    ar_i_key = ar_u_key.astype(np.int32)
    ar_i_key_encoded = (ar_i_key * ar_i_code).sum(axis = 1)
    # initialize default lookup table (int32 dtype is a later requirement)
    ar_i_lut = np.full(
        shape = [256 ** n_chn_key, n_chn_val],
        fill_value = v_val_default,
        dtype = np.int32
    )
    ar_i_lut[ar_i_key_encoded] = ar_u_val
    th_i_lut = torch.from_numpy(ar_i_lut).to(device)
    return ar_i_lut, th_i_lut


def lookup_chw(
    td_u_input: torch.tensor,
    td_i_lut: torch.tensor,
):
    """
    Applies a lookup table to a CHW tensor

    Args:
        td_u_input (torch.Tensor): input to apply the lookup table to
            Must have dtype uint8 and be on same device as th_i_lut
            Must have format CHW
        th_i_lut (torch.Tensor): lookup table
            Must have dtype int32

    Returns:
        (torch.Tensor): transformed tensor with looked-up values
            Has same dtype and is in same device as td_u_input
            Has format CHW
    """
    # validate compatibility between input and lookup table
    if (
        td_u_input.dtype != torch.uint8
        or td_i_lut.dtype != torch.int32
        or td_u_input.ndim != 3
        or td_u_input.device != td_i_lut.device
    ):
        raise ValueError(
            f"Incompatible dimensions, dtypes or devices for LuT or input:"
            f"\n- td_i_lut.dtype = {td_i_lut.dtype}, expected torch.int32"
            f"\n- td_u_input.dtype = {td_u_input.dtype}, expected torch.uint8"
            f"\n- td_u_input.ndim = {td_u_input.ndim}, expected 3"
            f"\n- td_u_input.device = {td_u_input.device} should be the same as td_i_lut.device = {td_i_lut.device}"
        )
    # encode colors into lookup keys
    # requires a multiplier for each channel: [256^0, 256^1, ..., 256^n_chn]
    n_chn_input = td_u_input.size()[0]
    td_i_code = torch.zeros(
        size = [n_chn_input, 1, 1],
        dtype = torch.int32,
        device = td_i_lut.device,
    )
    td_i_code[:, 0, 0] = 256 ** torch.arange(n_chn_input)
    # apply encoding for each channel separately
    # [  0,   0,   0] ->   0 * 256^0 +   0 * 256^1 +   0 * 256^2
    # [  0,   0,   1] ->   1 * 256^0 +   0 * 256^1 +   0 * 256^2
    # ...
    # [255, 255, 255] -> 255 * 256^0 + 255 * 256^1 + 255 * 256^2
    td_i_encoded = (td_u_input * td_i_code).sum(dim = 0)
    n_chn_output = td_i_lut.size()[1]
    td_u_output = torch.zeros(
        size = [n_chn_output, td_u_input.size()[1], td_u_input.size()[2]],
        dtype = torch.uint8,
    )
    for i_chn in torch.arange(n_chn_output):
        td_u_output[i_chn] = torch.take(
            input = td_i_lut[:, i_chn],
            index = td_i_encoded,
        ).reshape(td_u_output[i_chn].size())
    return td_u_output


def lookup_nchw(
    td_u_input: torch.tensor,
    td_i_lut: torch.tensor,
):
    """
    Applies a lookup table to a NCHW matrix (N: number of patches)

    Args:
        td_u_input (torch.Tensor): input to apply the lookup table to
            Must have dtype uint8 and be on same device as th_i_lut
            Must have format NCHW
        th_i_lut (torch.Tensor): lookup table
            Must have dtype int32

    Returns:
        (torch.Tensor): transformed matrix with looked-up values
            Has same dtype and is in same device as td_u_input
            Has format NCHW
    """
    # validate compatibility between input and lookup table
    if (
        td_u_input.dtype != torch.uint8
        or td_i_lut.dtype != torch.int32
        or td_u_input.ndim != 4
        or td_u_input.device != td_i_lut.device
    ):
        raise ValueError(
            f"Incompatible dimensions, dtypes or devices for LuT or input:"
            f"\n- td_i_lut.dtype = {td_i_lut.dtype}, expected torch.int32"
            f"\n- td_u_input.dtype = {td_u_input.dtype}, expected torch.uint8"
            f"\n- td_u_input.ndim = {td_u_input.ndim}, expected 4"
            f"\n- td_u_input.device = {td_u_input.device} should be the same as td_i_lut.device = {td_i_lut.device}"
        )
    # encode colors into lookup keys
    # requires a multiplier for each channel: [256^0, 256^1, ..., 256^n_chn]
    n_chn_input = td_u_input.size()[1]
    td_i_code = torch.zeros(
        size = [1, n_chn_input, 1, 1],
        dtype = torch.int32,
        device = td_i_lut.device,
    )
    td_i_code[0, :, 0, 0] = 256 ** torch.arange(n_chn_input)
    # apply encoding for each channel separately
    # [  0,   0,   0] ->   0 * 256^0 +   0 * 256^1 +   0 * 256^2
    # [  0,   0,   1] ->   1 * 256^0 +   0 * 256^1 +   0 * 256^2
    # ...
    # [255, 255, 255] -> 255 * 256^0 + 255 * 256^1 + 255 * 256^2
    td_i_encoded = (td_u_input * td_i_code).sum(dim = 1)
    n_chn_output = td_i_lut.size()[1]
    td_u_output = torch.zeros(
        size = [
            td_u_input.size()[0],
            n_chn_output,
            td_u_input.size()[2],
            td_u_input.size()[3]
        ],
        dtype = torch.uint8,
        device = td_i_lut.device,
    )
    for i_chn in torch.arange(n_chn_output):
        td_u_output[:, i_chn] = torch.take(
            input = td_i_lut[:, i_chn],
            index = td_i_encoded,
        ).reshape(td_u_output[:, i_chn].size())
    return td_u_output
