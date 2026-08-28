"""
This module defines the neural network metamodel.
"""

from __future__ import annotations

import keyword
import re
from typing import Self

from besser.BUML.metamodel.structural import BehaviorImplementation, NamedElement


class TensorOp(NamedElement):
    """
    This class represents a tensor operation. It encapsulates
    attributes such as the name and the type of the tensor operation.

    Args:
        name (str): The name of the tensor operation.
        tns_type (str): The type of the tensor operation.
        concatenate_dim (int): The dimension along which the tensors
            will be concatenated. Only relevant for concatenate
            operation.
        layers_of_tensors (list[str | float]): The list that
            defines the inputs of the tensor op. Elements can be layer
            names or scalar values.
        reshape_dim (list[int]): New shape for reshape operation.
        transpose_dim (list[int]): Transpose dimension specification.
        permute_dim (list[int]): Desired ordering of dimensions for
            permute.
        reduce_dim (int): Dimension for reduction operations like max 
            or mean.
        reduce_keepdims (bool): Whether to keep dimensions after
            reduction.
        shape_dim (int): Dimension index to extract from shape.
        input_reused (bool): Whether input is reused by another
            operation.
        actual_vars (list[str]): Component type labels for each input
            in concatenate and binary operations. Each element is
            either "output" or "hidden", indicating which component to
            reference from the corresponding layer. Used when layers
            have multiple output components (e.g., RNN layers with
            separate output and hidden states) to generate correct
            variable references during code generation.
        subscript_indices (list[dict]): A list of dictionaries
            containing indices for subscript operations.
        repeat_dim (list[int | str]): Repetition counts for
            repeat operation. Each element specifies how many times to
            repeat along that dimension. Elements can be integers for
            fixed counts (e.g., 2, 3) or strings representing
            variable/tensorop names that evaluate to integers at
            runtime (e.g., 'batch_size', 'n').
        interpolate_size (Tuple[int, ...]): Target dimensions for
            interpolation as tuple (H, W) or (D, H, W).
            Example: (224, 224) for 2D resize.
        interpolate_scale (float): Scale factor for interpolation.
        interpolate_mode (str): Interpolation mode. Valid values:
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear',
            'area', 'nearest-exact', 'lanczos3', 'lanczos5',
            'gaussian', 'mitchellcubic'. Default: 'bilinear'.
        pad_amount (list[list[int]]): Padding amounts as nested list.
            Each inner list contains [before, after] padding for one
            dimension. Example: [[1, 1], [2, 2]] pads first dim by 1
            on each side, second dim by 2.
        pad_mode (str): Padding mode like 'constant', 'reflect', or 
            'replicate'.
        pad_value (float): Value for constant padding (default 0).
        dropout_rate (float): Dropout probability, range [0.0, 1.0].
            Default: 0.5.
        dropout_training_aware (bool): Whether dropout behavior changes
            between training and inference modes. Default: False.
        split_dim (int): Dimension along which to split (supports
            negative indexing). Default: 0.
        split_sizes (int | list[int]): Number of equal chunks
            (int) or size per chunk (list). Required.
        permute_in (bool): Whether to permute input dimensions for
            spatial ops.
        permute_out (bool): Whether to permute output dimensions 
            for spatial ops.
        input_var (str): Input variable name for this tensor operation.
        output_var (str): Output variable name for this tensor
            operation.
        output_vars (list[str]): Output variable names for multi-output 
            operations (split/chunk).

    Attributes:
        name (str): The name of the tensor operation.
        tns_type (str): The type of the tensor operation.
        concatenate_dim (int): Concatenation dimension.
        layers_of_tensors (list[str | float]): Input layers or
            scalars.
        reshape_dim (list[int]): Reshape target shape.
        transpose_dim (list[int]): Transpose dimensions.
        permute_dim (list[int]): Permute dimension ordering.
        reduce_dim (int): Reduction dimension.
        reduce_keepdims (bool): Keep dimensions after reduction.
        shape_dim (int): Shape dimension to extract.
        input_reused (bool): Input reuse flag.
        actual_vars (list[str]): Component type labels ("output" 
            or "hidden") for each input. Used to select the correct
            component when layers have multiple outputs.
        subscript_indices (list[dict]): Subscript indices.
        repeat_dim (list[int | str]): Repetition counts for
            repeat operation. Each element specifies how many times
            to repeat along that dimension. Elements can be integers
            for fixed counts (e.g., 2, 3) or strings representing
            variable/tensorop names that evaluate to integers at
            runtime (e.g., 'batch_size', 'n').
        interpolate_size (Tuple[int, ...]): Target dimensions as
            tuple.
        interpolate_scale (float): Interpolation scale.
        interpolate_mode (str): Interpolation mode. Default 'bilinear'.
        pad_amount (list[list[int]]): Nested list of padding amounts
            per dimension.
        pad_mode (str): Padding mode ('constant', 'reflect',
            'replicate').
        pad_value (float): Value for constant padding.
        dropout_rate (float): Dropout rate [0.0, 1.0]. Default: 0.5.
        dropout_training_aware (bool): Training-aware dropout.
            Default: False.
        split_dim (int): Split dimension (supports negative indexing).
            Default: 0.
        split_sizes (int | list[int]): Number of equal chunks
            (int) or size per chunk (list).
        permute_in (bool): Input permute flag.
        permute_out (bool): Output permute flag.
        input_var (str): Input variable name for this tensor operation.
        output_var (str): Output variable name for this tensor
            operation.
        output_vars (list[str]): Output variable names for multi-output 
            operations (split/chunk).
    """
    def __init__(self, name: str, tns_type: str,
                 concatenate_dim: int | None = None,
                 layers_of_tensors: list[str | float] | None = None,
                 reshape_dim: list[int] | None = None,
                 transpose_dim: list[int] | None = None,
                 permute_dim: list[int] | None = None,
                 reduce_dim: int | None = None,
                 reduce_keepdims: bool = False, shape_dim: int | None = None,
                 input_reused: bool = False,
                 actual_vars: list[str] | None = None,
                 subscript_indices: list[dict] | None = None,
                 repeat_dim: list[int | str] | None = None,
                 interpolate_size: tuple | None = None,
                 interpolate_scale: float | None = None,
                 interpolate_mode: str = 'bilinear',
                 pad_amount : list[list[int]] | None = None,
                 pad_mode: str = 'constant', pad_value: float = 0.0,
                 dropout_rate: float | None = None,
                 dropout_training_aware: bool = True, split_dim: int = 0,
                 split_sizes: int | list[int] | None = None,
                 permute_in: bool = False, permute_out: bool = False,
                 input_var: str | None = None, output_var: str | None = None,
                 output_vars: list[str] | None = None):
        super().__init__(name)
        self.concatenate_dim: int = concatenate_dim
        self.layers_of_tensors: list[str | float] = layers_of_tensors
        self.reshape_dim: list[int] = reshape_dim
        self.transpose_dim: list[int] = transpose_dim
        self.permute_dim: list[int] = permute_dim
        self.reduce_dim: int = reduce_dim
        self.reduce_keepdims: bool = reduce_keepdims
        self.shape_dim: int = shape_dim
        self.input_reused: bool = input_reused
        self.actual_vars: list[str] = actual_vars
        self.subscript_indices: list[dict] = subscript_indices
        self.repeat_dim: list[int | str] = repeat_dim
        self.interpolate_size = interpolate_size
        self.interpolate_scale: float = interpolate_scale
        self.interpolate_mode: str = interpolate_mode
        self.pad_amount = pad_amount
        self.pad_mode: str = pad_mode
        self.pad_value: float = pad_value
        self.dropout_rate: float = dropout_rate
        self.dropout_training_aware: bool = dropout_training_aware
        self.split_dim: int = split_dim
        self.split_sizes: int | list[int] = split_sizes
        self.permute_in: bool = permute_in
        self.permute_out: bool = permute_out
        self.input_var: str = input_var
        self.output_var: str = output_var
        self.output_vars: list[str] = output_vars
        self.tns_type: str = tns_type

    @property
    def tns_type(self) -> str:
        """str: Get the type of the tensorOp."""
        return self.__tns_type

    @tns_type.setter
    def tns_type(self, tns_type: str):
        """
        str: Set the type of the tensorOp.

        Raises:
            ValueError: If the type is not one of the allowed options:
            'reshape', 'concatenate', 'multiply', 'matmultiply',
            'permute', 'transpose', 'mean', 'max', 'squeeze',
            'unsqueeze', 'binop_add', 'binop_subtract',
            'binop_multiply', 'binop_divide', 'binop_floor_divide',
            'subscript', 'shape_dim', 'normalize', 'repeat',
            'interpolate', 'pad', 'dropout', 'zeros_like', 'split', 
            'identity'
        """
        valid_types = [
            'reshape', 'concatenate', 'multiply', 'matmultiply', 'permute', 
            'transpose', 'mean', 'max', 'squeeze', 'unsqueeze', 'binop_add', 
            'binop_subtract', 'binop_multiply', 'binop_divide', 
            'binop_floor_divide', 'subscript', 'shape_dim', 'normalize', 
            'repeat', 'interpolate', 'pad', 'dropout', 'zeros_like', 
            'split', 'identity'
        ]
        if tns_type not in valid_types:
            raise ValueError("Invalid value of tensorOp type")
        self.__tns_type = tns_type
        self._validate()

    @property
    def concatenate_dim(self) -> int:
        """
        int: Get the dimension along which the tensors will be
        concatenated with the cat operation.
        """
        return self.__concatenate_dim

    @concatenate_dim.setter
    def concatenate_dim(self, concatenate_dim: int):
        """
        int: Set the dimension along which the tensors will be
        concatenated with the cat operation.
        """
        if (
            concatenate_dim is not None
            and (not isinstance(concatenate_dim, int))
        ):
            raise TypeError(
                "concatenate_dim must be int, got "
                f"{type(concatenate_dim).__name__}"
            )
        self.__concatenate_dim = concatenate_dim

    @property
    def layers_of_tensors(self) -> list[str | float]:
        """
        list[str | float]: Get the list that defines the inputs
        of the tensor op. Elements of the list can be either names
        of layers from which the tensors originate (str) or scalar
        values (float) for binary operations with constants.
        """
        return self.__layers_of_tensors

    @layers_of_tensors.setter
    def layers_of_tensors(self, layers_of_tensors: list[str | float]):
        """
        list[str | float]: Set the list of layers names from
        which the tensors, on which tensor ops are performed,
        originate. Can include scalar values (float) for binary
        operations with constants.
        """
        if layers_of_tensors is not None:
            if not isinstance(layers_of_tensors, list):
                raise TypeError(
                    "layers_of_tensors must be list, got "
                    f"{type(layers_of_tensors).__name__}"
                )
            for i, elem in enumerate(layers_of_tensors):
                if not isinstance(elem, (str, float)):
                    raise TypeError(
                        f"layers_of_tensors[{i}] must be str or float, got "
                        f"{type(elem).__name__}"
                    )
        self.__layers_of_tensors = layers_of_tensors

    @property
    def reshape_dim(self) -> list[int]:
        """
        list[int]: Get the list specifying the new shape of the tensor
        after reshaping with the view operation.
        """
        return self.__reshape_dim

    @reshape_dim.setter
    def reshape_dim(self, reshape_dim: list[int]):
        """
        list[int]: Set the list specifying the new shape of the tensor
        after reshaping with the view operation.
        """
        if reshape_dim is not None:
            if not isinstance(reshape_dim, list):
                raise TypeError(
                    "reshape_dim must be list, got "
                    f"{type(reshape_dim).__name__}"
                )
            if not all(isinstance(d, (int, str)) for d in reshape_dim):
                raise TypeError("reshape_dim elements must be int or str")
        self.__reshape_dim = reshape_dim

    @property
    def transpose_dim(self) -> list[int]:
        """
        list[int]: Get the list specifying the transpose dimensions.
        """
        return self.__transpose_dim

    @transpose_dim.setter
    def transpose_dim(self, transpose_dim: list[int]):
        """
        list[int]: Set the list specifying the transpose dimensions.
        """
        if transpose_dim is not None:
            if not isinstance(transpose_dim, list):
                raise TypeError(
                    "transpose_dim must be list, got "
                    f"{type(transpose_dim).__name__}"
                )
            if len(transpose_dim) != 2:
                raise ValueError(
                    "transpose_dim must have exactly 2 elements, got "
                    f"{len(transpose_dim)}"
                )
            if not all(isinstance(d, int) for d in transpose_dim):
                raise TypeError("transpose_dim elements must be int")
        self.__transpose_dim = transpose_dim

    @property
    def permute_dim(self) -> list[int]:
        """
        list[int]: Get the list containing the desired ordering of
        dimensions for permute operation.
        """
        return self.__permute_dim

    @permute_dim.setter
    def permute_dim(self, permute_dim: list[int]):
        """
        list[int]: Set the list containing the desired ordering of
        dimensions for permute operation.
        """
        if permute_dim is not None:
            if not isinstance(permute_dim, list):
                raise TypeError(
                    "permute_dim must be list, got "
                    f"{type(permute_dim).__name__}"
                )
            if not all(isinstance(d, int) for d in permute_dim):
                raise TypeError("permute_dim elements must be int")
        self.__permute_dim = permute_dim

    @property
    def shape_dim(self) -> int:
        """int: Get the dimension index for shape extraction."""
        return self.__shape_dim

    @shape_dim.setter
    def shape_dim(self, shape_dim: int):
        """int: Set the dimension index for shape extraction."""
        if shape_dim is not None and (not isinstance(shape_dim, int)):
            raise TypeError(
                f"shape_dim must be int, got {type(shape_dim).__name__}"
            )
        self.__shape_dim = shape_dim

    @property
    def input_reused(self) -> bool:
        """
        bool: Get whether the input to this layer is reused as input to
        another layer.
        """
        return self.__input_reused

    @input_reused.setter
    def input_reused(self, input_reused: bool):
        """
        bool: Set whether the input to this layer is reused as input to
        another layer.
        """
        if input_reused is not None and (not isinstance(input_reused, bool)):
            raise TypeError(
                "input_reused must be bool, got "
                f"{type(input_reused).__name__}"
            )
        self.__input_reused = input_reused

    @property
    def reduce_dim(self) -> int:
        """int: Get the dimension for reduction operations."""
        return self.__reduce_dim

    @reduce_dim.setter
    def reduce_dim(self, reduce_dim: int):
        """int: Set the dimension for reduction operations."""
        if (reduce_dim is not None and (not isinstance(reduce_dim, int))):
            raise TypeError(
                f"reduce_dim must be int, got {type(reduce_dim).__name__}"
            )
        self.__reduce_dim = reduce_dim

    @property
    def reduce_keepdims(self) -> bool:
        """bool: Get whether to keep dimensions after reduction."""
        return self.__reduce_keepdims

    @reduce_keepdims.setter
    def reduce_keepdims(self, reduce_keepdims: bool):
        """bool: Set whether to keep dimensions after reduction."""
        if (
            reduce_keepdims is not None
            and not isinstance(reduce_keepdims, bool)
        ):
            raise TypeError(
                "reduce_keepdims must be bool, got "
                f"{type(reduce_keepdims).__name__}"
            )
        self.__reduce_keepdims = reduce_keepdims

    @property
    def actual_vars(self) -> list[str]:
        """
        list[str]: Get component type labels for each input.
        Returns a list of strings ("output" or "hidden") indicating
        which component each input in layers_of_tensors refers to.
        Used by the code generator to emit correct variable references
        when layers have multiple output components (e.g., RNN layers
        with separate output and hidden states).
        """
        return self.__actual_vars

    @actual_vars.setter
    def actual_vars(self, actual_vars: list[str]):
        """
        list[str]: Set component type labels for each input.
        Each element should be "output" or "hidden", corresponding
        to inputs in layers_of_tensors. Used to determine which
        component variable to reference during code generation.
        """
        if actual_vars is not None:
            if not isinstance(actual_vars, list):
                raise TypeError(
                    "actual_vars must be list, got "
                    f"{type(actual_vars).__name__}"
                )
            if not all(
                isinstance(v, str) and v in ("output", "hidden")
                for v in actual_vars
            ):
                raise ValueError(
                    f"actual_vars must contain only 'output' or 'hidden', "
                    f"got {actual_vars}"
                )
        self.__actual_vars = actual_vars

    @property
    def subscript_indices(self) -> list[dict]:
        """list[dict]: Get the indices for subscript operations."""
        return self.__subscript_indices

    @subscript_indices.setter
    def subscript_indices(self, subscript_indices: list[dict]):
        """list[dict]: Set the indices for subscript operations."""
        # subscript_indices: list of dicts
        # (variable length based on tensor dimensions)
        if subscript_indices is not None:
            if not isinstance(subscript_indices, list):
                raise TypeError(
                    "subscript_indices must be list, got "
                    f"{type(subscript_indices).__name__}"
                )
            if len(subscript_indices) == 0:
                raise ValueError("subscript_indices cannot be empty")

            for i, elem in enumerate(subscript_indices):
                if not isinstance(elem, dict):
                    raise TypeError(
                        "subscript_indices[{i}] must be dict, got "
                        f"{type(elem).__name__}"
                    )
                if "type" not in elem:
                    raise ValueError(
                        f"subscript_indices[{i}] missing required 'type' key"
                    )
                if elem["type"] == "index":
                    if "value" not in elem:
                        raise ValueError(
                            f"subscript_indices[{i}] type 'index' requires "
                            f"'value' key"
                        )
                    if not isinstance(elem["value"], int):
                        raise TypeError(
                            f"subscript_indices[{i}]['value'] must be int, "
                            f"got {type(elem['value']).__name__}"
                        )
                elif elem["type"] == "slice":
                    for key in ["start", "stop", "step"]:
                        if (
                            key in elem
                            and elem[key] is not None
                            and not isinstance(elem[key], int)
                        ):
                            raise TypeError(
                                f"subscript_indices[{i}]['{key}'] must be int"
                                f" or None, got {type(elem[key]).__name__}"
                            )
                else:
                    raise ValueError(
                        f"subscript_indices[{i}]['type'] must be 'index' "
                        f"or 'slice', got {elem['type']}"
                    )
        self.__subscript_indices = subscript_indices

    @property
    def repeat_dim(self) -> list[int | str]:
        """
        list[int | str]: Get repetition counts for repeat
        operation. Each element specifies how many times to repeat
        along that dimension. Elements can be integers for fixed
        counts (e.g., 2, 3) or strings representing variable/tensorop
        names that evaluate to integers at runtime
        (e.g., 'batch_size', 'n').
        """
        return self.__repeat_dim

    @repeat_dim.setter
    def repeat_dim(self, repeat_dim: list[int | str]):
        """
        list[int | str]: Set repetition counts for repeat
        operation. Each element specifies how many times to repeat
        along that dimension. Elements can be integers for fixed
        counts or strings representing variable/tensorop names that
        evaluate to integers at runtime.
        """
        if repeat_dim is not None:
            if not isinstance(repeat_dim, list):
                raise TypeError(
                    "repeat_dim must be list, got "
                    f"{type(repeat_dim).__name__}"
                )
            for i, elem in enumerate(repeat_dim):
                if not isinstance(elem, (int, str)):
                    raise TypeError(
                        f"repeat_dim[{i}] must be int or str, got "
                        f"{type(elem).__name__}"
                    )
        self.__repeat_dim = repeat_dim

    @property
    def interpolate_size(self):
        """Get target size for interpolation."""
        return self.__interpolate_size

    @interpolate_size.setter
    def interpolate_size(self, interpolate_size: tuple):
        """Set target size for interpolation. Must be a tuple of
        integers."""
        if interpolate_size is not None:
            if not isinstance(interpolate_size, tuple):
                raise TypeError(
                    "interpolate_size must be a tuple, got "
                    f"{type(interpolate_size).__name__}"
                )
            if not all(isinstance(v, int) for v in interpolate_size):
                raise TypeError(
                    "All elements in interpolate_size must be integers"
                )
        self.__interpolate_size = interpolate_size

    @property
    def interpolate_scale(self) -> float:
        """float: Get scale factor for interpolation."""
        return self.__interpolate_scale

    @interpolate_scale.setter
    def interpolate_scale(self, interpolate_scale: float):
        """float: Set scale factor for interpolation. Must be > 0."""
        if interpolate_scale is not None:
            if not isinstance(interpolate_scale, (int, float)):
                raise TypeError(
                    "interpolate_scale must be numeric (int or float), got "
                    f"{type(interpolate_scale).__name__}"
                )
            if interpolate_scale <= 0:
                raise ValueError(
                    f"interpolate_scale must be > 0, got {interpolate_scale}"
                )
        self.__interpolate_scale = interpolate_scale
    @property
    def interpolate_mode(self) -> str:
        """str: Get interpolation mode."""
        return self.__interpolate_mode

    @interpolate_mode.setter
    def interpolate_mode(self, interpolate_mode: str):
        """str: Set interpolation mode. Must be one of the valid
        modes."""
        if interpolate_mode is not None:
            if not isinstance(interpolate_mode, str):
                raise TypeError(
                    "interpolate_mode must be a string, got "
                    f"{type(interpolate_mode).__name__}"
                )
            valid_modes = {
                'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear',
                'area', 'nearest-exact', 'lanczos3', 'lanczos5', 'gaussian',
                'mitchellcubic'
            }
            if interpolate_mode not in valid_modes:
                raise ValueError(
                    f"interpolate_mode must be one of {sorted(valid_modes)}, "
                    f"got '{interpolate_mode}'"
                )
        self.__interpolate_mode = interpolate_mode

    @property
    def pad_amount(self):
        """list[list[int]]: Get padding amounts as nested list 
           [[before, after], ...] per dimension."""
        return self.__pad_amount

    @pad_amount.setter
    def pad_amount(self, pad_amount):
        """Set padding amounts. Expected format: [[before, after], ...]
        for each dimension. All values must be >= 0."""
        if pad_amount is not None:
            if not isinstance(pad_amount, list):
                raise ValueError("pad_amount must be a list")
            for dim_pad in pad_amount:
                if not isinstance(dim_pad, list) or len(dim_pad) != 2:
                    raise ValueError(
                        "Each dimension in pad_amount must be a 2-element "
                        "list [before, after]"
                    )
                if any(v < 0 for v in dim_pad):
                    raise ValueError("Padding values must be >= 0")
        self.__pad_amount = pad_amount

    @property
    def pad_mode(self) -> str:
        """str: Get padding mode ('constant', 'reflect',
        'replicate')."""
        return self.__pad_mode

    @pad_mode.setter
    def pad_mode(self, pad_mode: str):
        """str: Set padding mode ('constant', 'reflect',
        'replicate')."""
        valid_pad = ['constant', 'reflect', 'replicate']
        if pad_mode is not None and pad_mode not in valid_pad:
            raise ValueError(
                "pad_mode must be 'constant', 'reflect', or 'replicate', got "
                f"'{pad_mode}'"
            )
        self.__pad_mode = pad_mode

    @property
    def pad_value(self) -> float:
        """float: Get value for constant padding (used only when 
           pad_mode='constant')."""
        return self.__pad_value

    @pad_value.setter
    def pad_value(self, pad_value: float):
        """float: Set value for constant padding (used only when
        pad_mode='constant'). Can be any numeric value (int or
        float)."""
        if pad_value is not None:
            if not isinstance(pad_value, (int, float)):
                raise TypeError(
                    f"pad_value must be numeric (int or float), got "
                    f"{type(pad_value).__name__}"
                )
            if pad_value != 0 and self.__pad_mode not in (None, 'constant'):
                raise ValueError(
                    "pad_value can only be set when pad_mode='constant', "
                    f"current pad_mode='{self.__pad_mode}'"
                )
        self.__pad_value = pad_value

    @property
    def dropout_rate(self) -> float:
        """float: Get dropout probability."""
        return self.__dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate: float):
        """Set dropout probability. Must be in range [0.0, 1.0]."""
        if dropout_rate is not None:
            if not isinstance(dropout_rate, (int, float)):
                raise TypeError(
                    "dropout_rate must be numeric (int or float), got "
                    f"{type(dropout_rate).__name__}"
                )
            if not (0.0 <= dropout_rate <= 1.0):
                raise ValueError(
                    "dropout_rate must be in range [0.0, 1.0], got "
                    f"{dropout_rate}"
                )
        self.__dropout_rate = dropout_rate

    @property
    def dropout_training_aware(self) -> bool:
        """bool: Get whether dropout is training aware."""
        return self.__dropout_training_aware

    @dropout_training_aware.setter
    def dropout_training_aware(self, dropout_training_aware: bool):
        """Set whether dropout is training aware. Must be bool."""
        if (
            dropout_training_aware is not None
            and not isinstance(dropout_training_aware, bool)
        ):
            raise TypeError(
                "dropout_training_aware must be bool, got "
                f"{type(dropout_training_aware).__name__}"
            )
        self.__dropout_training_aware = dropout_training_aware

    @property
    def split_dim(self) -> int:
        """int: Get dimension along which to split."""
        return self.__split_dim

    @split_dim.setter
    def split_dim(self, split_dim: int):
        """Set dimension along which to split. Supports negative
        indexing."""
        if split_dim is not None and not isinstance(split_dim, int):
            raise TypeError(
                f"split_dim must be int, got {type(split_dim).__name__}"
            )
        self.__split_dim = split_dim

    @property
    def split_sizes(self) -> int | list[int]:
        """int | list[int]: Get number of splits or
        list of split sizes."""
        return self.__split_sizes

    @split_sizes.setter
    def split_sizes(self, split_sizes: int | list[int]):
        """Set number of splits (int) or size per chunk
        (list of ints)."""
        if split_sizes is not None:
            if isinstance(split_sizes, int):
                if split_sizes <= 0:
                    raise ValueError(
                        f"split_sizes must be > 0, got {split_sizes}"
                    )
            elif isinstance(split_sizes, list):
                if not all(isinstance(x, int) and x > 0 for x in split_sizes):
                    raise ValueError(
                        "split_sizes list must contain only positive ints"
                    )
            else:
                raise TypeError(
                    "split_sizes must be int or list[int], got "
                    f"{type(split_sizes).__name__}"
                )
        self.__split_sizes = split_sizes

    @property
    def permute_in(self) -> bool:
        """bool: Get whether to permute input dimensions."""
        return self.__permute_in

    @permute_in.setter
    def permute_in(self, permute_in: bool):
        """bool: Set whether to permute input dimensions."""
        if (permute_in is not None and (not isinstance(permute_in, bool))):
            raise TypeError(
                f"permute_in must be bool, got {type(permute_in).__name__}"
            )
        self.__permute_in = permute_in

    @property
    def permute_out(self) -> bool:
        """bool: Get whether to permute output dimensions."""
        return self.__permute_out

    @permute_out.setter
    def permute_out(self, permute_out: bool):
        """bool: Set whether to permute output dimensions."""
        if (permute_out is not None and (not isinstance(permute_out, bool))):
            raise TypeError(
                f"permute_out must be bool, got {type(permute_out).__name__}"
            )
        self.__permute_out = permute_out

    @property
    def input_var(self) -> str:
        """str: Get the input variable name for this tensor
        operation."""
        return self.__input_var

    @input_var.setter
    def input_var(self, input_var: str):
        """str: Set the input variable name for this tensor
        operation."""
        if (
            input_var is not None
            and not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*$', input_var)
        ):
            raise ValueError(
                "input_var must be a valid identifier, or a comma-separated "
                "list of identifiers, each starting with a letter "
                "or underscore"
            )
        self.__input_var = input_var

    @property
    def output_var(self) -> str:
        """str: Get the output variable name for this tensor
        operation."""
        return self.__output_var

    @output_var.setter
    def output_var(self, output_var: str):
        """str: Set the output variable name for this tensor
        operation."""
        if (
            output_var is not None
            and not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', output_var)
        ):
            raise ValueError(
                "output_var must be a valid identifier starting with a letter"
                "or underscore"
            )
        self.__output_var = output_var

    @property
    def output_vars(self) -> list[str]:
        """list[str]: Get the output variable names for multi-output 
        operations (split/chunk)."""
        return self.__output_vars

    @output_vars.setter
    def output_vars(self, output_vars: list[str]):
        """list[str]: Set the output variable names for multi-output 
        operations (split/chunk)."""
        if output_vars is not None:
            if not isinstance(output_vars, list):
                raise TypeError(
                    "output_vars must be a list, got "
                    f"{type(output_vars).__name__}"
                )
            for var in output_vars:
                if not isinstance(var, str):
                    raise TypeError(
                        f"each element of output_vars must be str, "
                        f"got {type(var).__name__}"
                    )
                if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', var):
                    raise ValueError(
                        f"'{var}' is not valid, each output_var must be a "
                        "valid identifier starting with an alphabetic "
                        "character"
                    )
            if len(output_vars) != self.__split_sizes:
                raise ValueError(
                    f"Length of output_vars ({len(output_vars)}) must match "
                    f"the number of splits specified in split_sizes "
                    f"({self.__split_sizes})"
                )
        self.__output_vars = output_vars

    @property
    def output_var_joined(self) -> str:
        """str: Returns comma-joined string for multi-output 
        ops (split/chunk)."""
        if self.output_vars:
            return ", ".join(self.output_vars)
        return self.output_var or ""

    def _validate(self):
        """Validate the tensor operation parameters based on
        the type of operation."""
        binops = ['binop_add', 'binop_subtract', 'binop_multiply',
                  'binop_divide', 'binop_floor_divide']
        if self.tns_type == 'reshape' and self.reshape_dim is None:
            raise ValueError(
                "reshape_dim parameter cannot be None when type is 'reshape'"
            )
        elif self.tns_type == 'concatenate' and self.concatenate_dim is None:
            raise ValueError(
                "concatenate_dim parameter cannot be None when type is "
                "'concatenate'"
            )
        elif self.tns_type == 'transpose' and self.transpose_dim is None:
            raise ValueError(
                "transpose_dim parameter cannot be None when type is "
                "'transpose'"
            )
        elif self.tns_type == 'permute' and self.permute_dim is None:
            raise ValueError(
                "permute_dim parameter cannot be None when type is 'permute'"
            )
        elif (
            self.tns_type in  [
                'shape_dim', 'mean', 'max',
                'squeeze', 'unsqueeze', 'normalize'
            ] 
        and self.reduce_dim is None
        ):
            raise ValueError(
                "reduce_dim parameter cannot be None when "
                f"type is {self.tns_type}"
            )
        elif self.tns_type == 'max' and self.reduce_keepdims is None:
            raise ValueError(
                "reduce_keepdims parameter cannot be None when type is 'max'"
            )
        elif self.tns_type == 'subscript' and self.subscript_indices is None:
            raise ValueError(
                "subscript_indices parameter cannot be None when "
                "type is 'subscript'"
            )
        elif self.tns_type == 'repeat' and self.repeat_dim is None:
            raise ValueError(
                "repeat_dim parameter cannot be None when type is 'repeat'"
            )
        elif self.tns_type == 'interpolate':
            if (
                self.interpolate_size is None
                and self.interpolate_scale is None
            ):
                raise ValueError(
                    "Either interpolate_size or interpolate_scale must "
                    "be set when type is 'interpolate'"
                )
            if (
                self.interpolate_size is not None
                and self.interpolate_scale is not None
            ):
                raise ValueError(
                    "Cannot set both interpolate_size and interpolate_scale. "
                     "Use one or the other")
        elif self.tns_type == 'pad':
            if self.pad_amount is None:
                raise ValueError(
                    "pad_amount parameter cannot be None when type is 'pad'"
                )
            elif self.pad_mode is None:
                self.pad_mode = 'constant'
            elif self.pad_mode == 'constant' and self.pad_value is None:
                self.pad_value = 0.0
        elif self.tns_type == 'dropout' and self.dropout_rate is None:
            raise ValueError(
                "dropout_rate parameter cannot be None when type is 'dropout'"
            )
        elif self.tns_type == 'split':
            if self.split_dim is None:
                self.split_dim = 0
            elif self.split_sizes is None:
                raise ValueError(
                    "split_sizes parameter cannot be None "
                    "when type is 'split'"
                )
        if self.tns_type in binops + ['multiply', 'matmultiply']:
            if self.layers_of_tensors is None:
                raise ValueError(
                    "layers_of_tensors parameter should "
                    f"be provided for {self.tns_type} operations"
                )
        elif (
            self.tns_type in ['max', 'mean', 'normalize', 'repeat', 'reshape',
                              'shape_dim', 'split', 'squeeze', 'transpose',
                              'unsqueeze', 'zeros_like']
            and self.layers_of_tensors is None
            and self.input_var is None
        ):
            raise ValueError(
                "Either layers_of_tensors or input_var parameter should "
                f"be provided for {self.tns_type} operations"
            )

    def __repr__(self):
        return (
            f'TensorOp({self.name}, {self.tns_type}, {self.concatenate_dim}, '
            f'{self.layers_of_tensors}, {self.reshape_dim}, '
            f'{self.transpose_dim}, {self.permute_dim}, {self.reduce_dim}, '
            f'{self.reduce_keepdims}, {self.shape_dim}, {self.input_reused}, '
            f'{self.actual_vars}, {self.subscript_indices}, '
            f'{self.repeat_dim}, {self.interpolate_size}, '
            f'{self.interpolate_scale}, {self.interpolate_mode}, '
            f'{self.pad_amount}, {self.pad_mode}, {self.pad_value}, '
            f'{self.dropout_rate}, {self.dropout_training_aware}, '
            f'{self.split_dim}, {self.split_sizes}, {self.permute_in}, '
            f'{self.permute_out}, {self.input_var}, {self.output_var}, '
            f'{self.output_vars})'
        )

class Layer(NamedElement):
    """
    This class represents a layer of the neural network.
    It encapsulates attributes such as the name of the layer and 
    the activation function.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse). When True, the
            generator should NOT define this layer in __init__, only
            call it in forward/call.
        permute_in (bool): Whether to permute input dimensions before 
            processing. Used for format conversions
            (e.g., NHWC to NCHW).
        permute_out (bool): Whether to permute output dimensions after 
            processing. Used for format conversions
            (e.g., NCHW back to NHWC).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse). When True, 
            the generator should NOT define this layer in 
            __init__, only call it in forward/call.
        permute_in (bool): Whether to permute input dimensions before 
            processing.
        permute_out (bool): Whether to permute output dimensions after 
            processing.
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.
    """
    def __init__(self, name: str, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 is_layer_call: bool = False, permute_in: bool = False,
                 permute_out: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        super().__init__(name)
        self.actv_func: str = actv_func
        self.name_module_input: str = name_module_input
        self.input_reused: bool = input_reused
        self.is_layer_call: bool = is_layer_call
        self.permute_in: bool = permute_in
        self.permute_out: bool = permute_out
        self.input_var: str = input_var
        self.output_var: str = output_var

    @property
    def actv_func(self) -> str:
        """str: Get the actv_func."""
        return self.__actv_func

    @actv_func.setter
    def actv_func(self, actv_func: str):
        """str: Set the actv_func."""
        #if actv_func is not None and actv_func not in [
        #    'relu', 'leaky_relu', 'sigmoid', 'softmax', 'tanh', 'gelu'
        #]:
        #    raise ValueError("Invalid value of actv_func")
        self.__actv_func = actv_func

    @property
    def name_module_input(self) -> str:
        """str: Get the name of the layer from which the inputs
        originate."""
        return self.__name_module_input

    @name_module_input.setter
    def name_module_input(self, name_module_input: str):
        """str: Set the name of the layer from which the inputs
        originate."""
        self.__name_module_input = name_module_input

    @property
    def input_reused(self) -> bool:
        """bool: Get whether the input to this layer is reused as
        input to another layer."""
        return self.__input_reused

    @input_reused.setter
    def input_reused(self, input_reused: bool):
        """bool: Set whether the input to this layer is reused as
            input to another layer."""
        self.__input_reused = input_reused

    @property
    def is_layer_call(self) -> bool:
        """bool: Get whether this represents a call to
            an already-defined layer."""
        return self.__is_layer_call

    @is_layer_call.setter
    def is_layer_call(self, is_layer_call: bool):
        """bool: Set whether this represents a call to
            an already-defined layer."""
        self.__is_layer_call = is_layer_call

    @property
    def permute_in(self) -> bool:
        """bool: Get whether to permute input dimensions before
        processing."""
        return self.__permute_in

    @permute_in.setter
    def permute_in(self, permute_in: bool):
        """bool: Set whether to permute input dimensions before
        processing."""
        self.__permute_in = permute_in

    @property
    def permute_out(self) -> bool:
        """bool: Get whether to permute output dimensions after
        processing."""
        return self.__permute_out

    @permute_out.setter
    def permute_out(self, permute_out: bool):
        """bool: Set whether to permute output dimensions after
        processing."""
        self.__permute_out = permute_out

    @property
    def input_var(self) -> str:
        """str: Get the input variable name for this layer."""
        return self.__input_var

    @input_var.setter
    def input_var(self, input_var: str):
        """str: Set the input variable name for this layer."""
        self.__input_var = input_var

    @property
    def output_var(self) -> str:
        """str: Get the output variable name for this layer."""
        return self.__output_var

    @output_var.setter
    def output_var(self, output_var: str):
        """str: Set the output variable name for this layer."""
        self.__output_var = output_var

    def __repr__(self):
        return (
            f'Layer({self.name}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.permute_in}, {self.permute_out}, '
            f'{self.input_var}, {self.output_var})'
        )

class CNN(Layer):
    """
    Represents a layer that is generally used in convolutional neural
    networks.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        kernel_dim (list[int]): A list containing the dimensions of
            the convolving or pooling kernel (i.e., [depth, height,
            width]).
        stride_dim (list[int]): A list containing the dimensions of
            the stride of the convolution or pooling (i.e., [depth,
            height, width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse). When True, the 
            generator should NOT define this layer in __init__, only
            call it in forward/call.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        kernel_dim (list[int]): A list containing the dimensions of
            the convolving or pooling kernel (i.e., [depth, height,
            width]).
        stride_dim (list[int]): A list containing the dimensions of
            the stride of the convolution or pooling (i.e., [depth,
            height, width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): Inherited from Layer. The name of
            the layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input
            to this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is
            a call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """

    def __init__(self, name: str, kernel_dim: list[int],
                 stride_dim: list[int], padding_amount: int = 0,
                 padding_type: str = "valid", actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 permute_in: bool = False, permute_out: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        super().__init__(name, actv_func, name_module_input, input_reused,
                         is_layer_call, permute_in, permute_out, input_var,
                         output_var)
        self.kernel_dim: list[int] = kernel_dim
        self.stride_dim: list[int] = stride_dim
        self.padding_amount: int = padding_amount
        self.padding_type: str = padding_type

    @property
    def kernel_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: list[int]):
        """list[int]: Set the list of dimensions of the kernel."""
        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: list[int]):
        """list[int]: Set the list of dimensions of the stride."""
        if stride_dim is None:
            self.__stride_dim = self.kernel_dim
        else:
            self.__stride_dim = stride_dim

    @property
    def padding_amount(self) -> int:
        """int: Get the amount of padding added to the input."""
        return self.__padding_amount

    @padding_amount.setter
    def padding_amount(self, padding_amount: int):
        """int: Set the amount of padding added to the input."""
        self.__padding_amount = padding_amount

    @property
    def padding_type(self) -> str:
        """str: Get the type of padding applied to the input."""
        return self.__padding_type

    @padding_type.setter
    def padding_type(self, padding_type: str):
        """
        str: Set the type of padding applied to the input.

        Raises:
            ValueError: If the padding type provided is none of
            these: 'same' or 'valid'.
        """
        if padding_type not in ['same', 'valid']:
            raise ValueError ("Invalid padding type")
        self.__padding_type = padding_type


    @property
    def permute_in(self) -> bool:
        """bool: Get whether to permute the dim of the input."""
        return self.__permute_in

    @permute_in.setter
    def permute_in(self, permute_in: bool):
        """bool: Set whether to permute the dim of the input."""
        self.__permute_in = permute_in

    @property
    def permute_out(self) -> bool:
        """bool: Get whether to permute the dim of the output."""
        return self.__permute_out

    @permute_out.setter
    def permute_out(self, permute_out: bool):
        """bool: Set whether to permute the dim of the output."""
        self.__permute_out = permute_out


    def __repr__(self):
        return (
            f'CNN({self.name}, {self.actv_func}, {self.kernel_dim}, '
            f'{self.stride_dim}, {self.padding_amount}, {self.padding_type}, '
            f'{self.permute_in}, {self.permute_out}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class ConvolutionalLayer(CNN):
    """
    Represents a convolutional layer.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        kernel_dim (list[int]): A list containing the dimensions of
            the convolving or pooling kernel (i.e., [depth, height,
            width]).
        stride_dim (list[int]): A list containing the dimensions of
            the stride of the convolution or pooling (i.e., [depth,
            height, width]).
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by
            the convolution.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        dilation (list[int]): Spacing between kernel elements.
            Default is [1].
        groups (int): Number of blocked connections from input 
            to output channels. Default is 1.
        bias (bool): If True, adds a learnable bias to the output.
            Default is True.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to an 
            already-defined layer (layer reuse).

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        kernel_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the convolving or pooling kernel (i.e.,
            [depth, height, width]).
        stride_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the stride of the convolution or pooling
            (i.e., [depth, height, width]).
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by
            the convolution.
        padding_amount (int): Inherited from CNN. The amount of padding
            added to the input.
        padding_type (str): Inherited from CNN. The type of padding
            applied to the input.
        dilation (list[int]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input 
            to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        permute_in (bool): Inherited from CNN. Whether the dimensions
            of the input need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        permute_out (bool): Inherited from CNN. Whether the dimensions
            of the output need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input
            to this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is
            a call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """

    def __init__(self, name: str, kernel_dim: list[int], out_channels: int,
                 stride_dim: list[int], in_channels: int | None = None,
                 padding_amount: int = 0, padding_type: str = "valid",
                 dilation: list[int] | None = None, groups: int = 1,
                 bias: bool = True, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 permute_in: bool = False, permute_out: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        super().__init__(name, kernel_dim, stride_dim, padding_amount,
                         padding_type, actv_func, name_module_input,
                         input_reused, permute_in, permute_out, is_layer_call,
                         input_var, output_var)
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.dilation: list[int] = dilation if dilation is not None else [1]
        self.groups: int = groups
        self.bias: bool = bias

    @property
    def in_channels(self) -> int:
        """int: Get the number of channels in the input image."""
        return self.__in_channels

    @in_channels.setter
    def in_channels(self, in_channels: int):
        """int: Set the number of channels in the input image."""
        self.__in_channels = in_channels

    @property
    def out_channels(self) -> int:
        """int: Get the number of channels produced
        by the convolution."""
        return self.__out_channels

    @out_channels.setter
    def out_channels(self, out_channels: int):
        """int: Set the number of channels produced
        by the convolution."""
        self.__out_channels = out_channels

    @property
    def dilation(self) -> list[int]:
        """list[int]: Get the spacing between kernel elements."""
        return self.__dilation

    @dilation.setter
    def dilation(self, dilation: list[int]):
        """list[int]: Set the spacing between kernel elements."""
        self.__dilation = dilation

    @property
    def groups(self) -> int:
        """int: Get the number of blocked connections from input 
        to output channels."""
        return self.__groups

    @groups.setter
    def groups(self, groups: int):
        """int: Set the number of blocked connections from input 
        to output channels."""
        self.__groups = groups

    @property
    def bias(self) -> bool:
        """bool: Get whether the layer uses bias."""
        return self.__bias

    @bias.setter
    def bias(self, bias: bool):
        """bool: Set whether the layer uses bias."""
        self.__bias = bias


    def __repr__(self):
        return (
            f'ConvolutionaLayer({self.name}, {self.kernel_dim}, '
            f'{self.out_channels}, {self.stride_dim}, {self.in_channels}, '
            f'{self.padding_amount}, {self.padding_type}, {self.dilation}, '
            f'{self.groups}, {self.bias}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.permute_in}, {self.permute_out}, {self.is_layer_call}, '
            f'{self.input_var}, {self.output_var})'
        )

class Conv1D(ConvolutionalLayer):
    """
    Represents a type of convolutional layer that applies a 1D
    convolution.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        kernel_dim (list[int]): A list containing the dimensions of
            the convolving or pooling kernel (i.e., [depth, height,
            width]).
        stride_dim (list[int]): A list containing the dimensions of
            the stride of the convolution or pooling (i.e., [depth,
            height, width]).
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by
            the convolution.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        dilation (list[int]): Spacing between kernel elements.
            Default is [1].
        groups (int): Number of blocked connections from input to
            output channels. Default is 1.
        bias (bool): If True, adds a learnable bias to the output. 
            Default is True.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        kernel_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the convolving or pooling kernel
            (i.e., [depth, height, width]).
        stride_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the stride of the convolution or pooling
            (i.e., [depth, height, width]).
        in_channels (int): Inherited from ConvolutionalLayer. It
            represents the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It
            represents the number of channels produced by the
            convolution.
        padding_amount (int): Inherited from CNN. It represents the
            amount of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type
            of padding applied to the input.
        dilation (list[int]): Inherited from ConvolutionalLayer. 
            Spacing between kernel elements.
        groups (int): Inherited from ConvolutionalLayer. 
            Number of blocked connections from input to output channels.
        bias (bool): Inherited from ConvolutionalLayer. If True, 
            adds a learnable bias to the output.
        permute_in (bool): Inherited from CNN. Whether the dimensions
            of the input need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        permute_out (bool): Inherited from CNN. Whether the dimensions
            of the output need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """
    def __init__(self, name: str, kernel_dim: list[int], out_channels: int,
                 stride_dim: list[int] | None = None, in_channels: int | None = None,
                 padding_amount: int = 0, padding_type: str = "valid",
                 dilation: list[int] | None = None, groups: int = 1,
                 bias: bool = True, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 permute_in: bool = False, permute_out: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        if stride_dim is None:
            stride_dim = [1]
        super().__init__(name, kernel_dim, out_channels, stride_dim,
                         in_channels, padding_amount, padding_type, dilation,
                         groups, bias, actv_func, name_module_input, 
                         input_reused, permute_in, permute_out, is_layer_call,
                         input_var, output_var)

    @property
    def kernel_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: list[int]):
        """list[int]: Set the list of dimensions of the kernel.
        An error is raised if the list contains more than 1 element
        (dimension)."""
        if len(kernel_dim) != 1:
            raise ValueError(
                "kernel_dim list must have exactly 1 element (dimension)."
            )

        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: list[int]):
        """list[int]: Set the list of dimensions of the stride.
        An error is raised if the list contains more than 1 element
        (dimension)."""
        if len(stride_dim) != 1:
            raise ValueError(
                "stride_dim list must have exactly 1 element (dimension)."
            )
        self.__stride_dim = stride_dim


    def __repr__(self):
        return (
            f'Conv1D({self.name}, {self.actv_func}, {self.kernel_dim}, '
            f'{self.out_channels}, {self.stride_dim}, {self.in_channels}, '
            f'{self.padding_amount}, {self.padding_type}, {self.dilation}, '
            f'{self.groups}, {self.bias}, {self.name_module_input}, '
            f'{self.input_reused}, {self.permute_in}, {self.permute_out}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class Conv2D(ConvolutionalLayer):
    """
    Represents a type of convolutional layer that applies a 2D
    convolution.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        kernel_dim (list[int]): A list containing the dimensions of
            the convolving or pooling kernel (i.e., [depth, height,
            width]).
        stride_dim (list[int]): A list containing the dimensions of
            the stride of the convolution or pooling
            (i.e., [depth, height, width]).
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by
            the convolution.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        dilation (list[int]): Spacing between kernel elements.
            Default is [1].
        groups (int): Number of blocked connections from input 
            to output channels. Default is 1.
        bias (bool): If True, adds a learnable bias to the output. 
            Default is True.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).

    Attributes:
        name (str): Inherited from Layer. It represents the name of the
            layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        kernel_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the convolving or pooling kernel
            (i.e., [depth, height, width]).
        stride_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the stride of the convolution or pooling
            (i.e., [depth, height, width]).
        in_channels (int): Inherited from ConvolutionalLayer. It
            represents the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It
            represents the number of channels produced by the
            convolution.
        padding_amount (int): Inherited from CNN. It represents the
            amount of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the
            type of padding applied to the input.
        dilation (list[int]): Inherited from ConvolutionalLayer. 
            Spacing between kernel elements.
        groups (int): Inherited from ConvolutionalLayer. Number of 
            blocked connections from input to output channels.
        bias (bool): Inherited from ConvolutionalLayer. If True, 
            adds a learnable bias to the output.
        permute_in (bool): Inherited from CNN. Whether the dimensions
            of the input need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        permute_out (bool): Inherited from CNN. Whether the dimensions
            of the output need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """
    def __init__(self, name: str, kernel_dim: list[int], out_channels: int,
                 stride_dim: list[int] | None = None, in_channels: int | None = None,
                 padding_amount: int = 0, padding_type: str = "valid",
                 dilation: list[int] | None = None, groups: int = 1,
                 bias: bool = True, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 permute_in: bool = False, permute_out: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        if stride_dim is None:
            stride_dim = [1, 1]
        super().__init__(name, kernel_dim, out_channels, stride_dim,
                         in_channels, padding_amount, padding_type, dilation,
                         groups, bias, actv_func, name_module_input,
                         input_reused, permute_in, permute_out, is_layer_call,
                         input_var, output_var)

    @property
    def kernel_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: list[int]):
        """list[int]: Set the list of dimensions of the kernel.
        An error is raised if the list contains more than 2 elements
        (dimensions)."""
        if len(kernel_dim) != 2:
            raise ValueError(
                "kernel_dim list must have exactly 2 elements (dimensions)."
            )

        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: list[int]):
        """list[int]: Set the list of dimensions of the stride.
        An error is raised if the list contains more than 2 elements
        (dimensions)."""
        if len(stride_dim) != 2:
            raise ValueError(
                "stride_dim list must have exactly 2 elements (dimensions)."
            )
        self.__stride_dim = stride_dim

    def __repr__(self):
        return (
            f'Conv2D({self.name}, {self.kernel_dim}, {self.out_channels}, '
            f'{self.stride_dim}, {self.in_channels}, {self.padding_amount}, '
            f'{self.padding_type}, {self.dilation}, {self.groups}, '
            f'{self.bias}, {self.actv_func}, {self.name_module_input}, '
            f'{self.input_reused}, {self.permute_in}, {self.permute_out}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class Conv3D(ConvolutionalLayer):
    """
    Represents a type of convolutional layer that applies a 3D
    convolution.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        kernel_dim (list[int]): A list containing the dimensions of
            the convolving or pooling kernel (i.e., [depth, height,
            width]).
        stride_dim (list[int]): A list containing the dimensions of
            the stride of the convolution or pooling
            (i.e., [depth, height, width]).
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by
            the convolution.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        dilation (list[int]): Spacing between kernel elements.
            Default is [1].
        groups (int): Number of blocked connections from input to 
            output channels. Default is 1.
        bias (bool): If True, adds a learnable bias to the output.
            Default is True.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        kernel_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the convolving or pooling kernel
            (i.e., [depth, height, width]).
        stride_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the stride of the convolution or pooling
            (i.e., [depth, height, width]).
        in_channels (int): Inherited from ConvolutionalLayer. It
            represents the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It
            represents the number of channels produced by the
            convolution.
        padding_amount (int): Inherited from CNN. It represents the
            amount of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type
            of padding applied to the input.
        dilation (list[int]): Inherited from ConvolutionalLayer. 
            Spacing between kernel elements.
        groups (int): Inherited from ConvolutionalLayer. Number of
            blocked connections from input to output channels.
        bias (bool): Inherited from ConvolutionalLayer. If True, 
            adds a learnable bias to the output.
        permute_in (bool): Inherited from CNN. Whether the dimensions
            of the input need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        permute_out (bool): Inherited from CNN. Whether the dimensions
            of the output need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name for
            this layer.
        output_var (str): Inherited from Layer. Output variable name
            for this layer.
    """
    def __init__(self, name: str, kernel_dim: list[int], out_channels: int,
                 stride_dim: list[int] | None = None, in_channels: int | None = None,
                 padding_amount: int = 0, padding_type: str = "valid",
                 dilation: list[int] | None = None, groups: int = 1,
                 bias: bool = True, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 permute_in: bool = False, permute_out: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        if stride_dim is None:
            stride_dim = [1, 1, 1]
        super().__init__(name, kernel_dim, out_channels, stride_dim,
                         in_channels, padding_amount, padding_type, dilation,
                         groups, bias, actv_func, name_module_input,
                         input_reused, permute_in, permute_out, is_layer_call,
                         input_var, output_var)

    @property
    def kernel_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: list[int]):
        """list[int]: Set the list of dimensions of the kernel.
        An error is raised if the list does not contains exactly 3
        elements (dimensions)."""
        if len(kernel_dim) != 3:
            raise ValueError(
                "kernel_dim list must have exactly 3 element (dimensions)."
            )
        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: list[int]):
        """list[int]: Set the list of dimensions of the stride.
        An error is raised if the list does not contains exactly
        3 elements (dimensions)."""
        if len(stride_dim) != 3:
            raise ValueError(
                "stride_dim list must have exactly 3 elements (dimensions)."
            )
        self.__stride_dim = stride_dim

    def __repr__(self):
        return (
            f'Conv3D({self.name}, {self.kernel_dim}, {self.out_channels}, '
            f'{self.stride_dim}, {self.in_channels}, {self.padding_amount}, '
            f'{self.padding_type}, {self.dilation}, {self.groups}, '
            f'{self.bias}, {self.actv_func}, {self.name_module_input}, '
            f'{self.input_reused}, {self.permute_in}, {self.permute_out}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )


class PoolingLayer(CNN):
    """
    Represents a type of layer that performs a pooling operation.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the
            pooling operation.
        kernel_dim (list[int]): A list containing the dimensions of
            the convolving or pooling kernel (i.e., [depth, height,
            width]).
        stride_dim (list[int]): A list containing the dimensions of
            the stride of the convolution or pooling
            (i.e., [depth, height, width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        pooling_type (str): The type of pooling. Either average or max.
        output_dim (list[int]): The output dimensions of the adaptive
            pooling operation. Only relevant for adaptive pooling.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is
            reused as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the
            layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the
            pooling operation.
        kernel_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the convolving or pooling kernel
            (i.e., [depth, height, width]).
        stride_dim (list[int]): Inherited from CNN. A list containing
            the dimensions of the stride of the convolution or pooling
            (i.e., [depth, height, width]).
        padding_amount (int): Inherited from CNN. It represents the
            amount of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type
            of padding applied to the input.
        pooling_type (str): The type of pooling. Either average or max.
        output_dim (list[int]): The output dimensions of the adaptive
            pooling operation. Only relevant for adaptive pooling.
        permute_in (bool): Inherited from CNN. Whether the dimensions
            of the input need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        permute_out (bool): Inherited from CNN. Whether the dimensions
            of the output need to be permuted. Relevant for PyTorch.
            It is used to make PyTorch model equivalent to TensorFlow
            model.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name for 
            this layer.
        output_var (str): Inherited from Layer. Output variable name
            for this layer.
    """
    def __init__(self, name: str, pooling_type: str, dimension: str,
                 kernel_dim: list[int] | None = None, stride_dim: list[int] | None = None,
                 padding_amount: int = 0, padding_type: str = "valid",
                 output_dim: list[int] | None = None, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 permute_in: bool = False, permute_out: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        self.pooling_type: str = pooling_type
        self.dimension: str = dimension
        self.output_dim: list[int] = output_dim
        if output_dim is None:
            output_dim = []
        super().__init__(name, kernel_dim, stride_dim, padding_amount,
                         padding_type, actv_func, name_module_input,
                         input_reused, permute_in, permute_out, is_layer_call,
                         input_var, output_var)

    @property
    def kernel_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: list[int]):
        """list[int]: Set the list of dimensions of the kernel.
        An error is raised if the length of the list does not match
        the dimensionality of the pooling operation."""
        if not (self.pooling_type.startswith("adaptive") or
                self.pooling_type.startswith("global")):
            if self.dimension == "1D" and len(kernel_dim) != 1:
                raise ValueError(
                    "kernel_dim list must have exactly 1 element (dimension)."
                )
            elif self.dimension == "2D" and len(kernel_dim) != 2:
                raise ValueError(
                    "kernel_dim list must have exactly 2 elements "
                    "(dimensions)."
                )
            elif self.dimension == "3D" and len(kernel_dim) != 3:
                raise ValueError(
                    "kernel_dim list must have exactly 3 elements "
                    "(dimensions)."
                )
        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> list[int]:
        """list[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: list[int]):
        """list[int]: Set the list of dimensions of the stride.
        An error is raised if the length of the list does not match
        the dimensionality of the pooling operation."""
        if (
            not (self.pooling_type.startswith("adaptive") or
                 self.pooling_type.startswith("global"))
            and stride_dim is not None
        ):
            if self.dimension == "1D" and len(stride_dim) != 1:
                raise ValueError(
                    "kernel_dim list must have exactly 1 element (dimension)."
                )
            elif self.dimension == "2D" and len(stride_dim) != 2:
                raise ValueError(
                    "kernel_dim list must have exactly 2 elements "
                    "(dimensions)."
                )
            elif self.dimension == "3D" and len(stride_dim) != 3:
                raise ValueError(
                    "kernel_dim list must have exactly 3 elements "
                    "(dimensions)."
                    )
            self.__stride_dim = stride_dim

        elif stride_dim is None:
            self.__stride_dim = self.kernel_dim
        else:
            self.__stride_dim = stride_dim

    @property
    def pooling_type(self) -> str:
        """str: Get the type of pooling applied."""
        return self.__pooling_type

    @pooling_type.setter
    def pooling_type(self, pooling_type: str):
        """
        str: Set the type of pooling.

        Raises:
            ValueError: If the pooling type provided is none of these:
            'average', 'adaptive_average', 'max' or 'adaptive_max'.
        """

        if pooling_type not in [
            'average', 'adaptive_average', 'max', 'adaptive_max',
            'global_average', 'global_max'
        ]:
            raise ValueError ("Invalid pooling type")
        self.__pooling_type = pooling_type

    @property
    def dimension(self) -> str:
        """str: Get the dimensionality of the pooling."""
        return self.__dimension

    @dimension.setter
    def dimension(self, dimension: str):
        """
        str: Set the dimensionality of the pooling.

        Raises:
            ValueError: If the pooling dimensionality is none of
            these: '1D', '2D', or '3D'.
        """

        if dimension not in ['1D', '2D', '3D']:
            raise ValueError ("Invalid pooling dimensionality")
        self.__dimension = dimension

    @property
    def output_dim(self) -> list[int]:
        """list[int]: Get the output dimensions
        of the adaptive pooling."""
        return self.__output_dim

    @output_dim.setter
    def output_dim(self, output_dim: list[int]):
        """list[int]: Set the output dimensions
        of the adaptive pooling."""
        self.__output_dim = output_dim

    def __repr__(self):
        return (
            f'PoolingLayer({self.name}, {self.actv_func}, '
            f'{self.pooling_type}, {self.dimension}, {self.kernel_dim}, '
            f'{self.stride_dim}, {self.padding_amount}, {self.padding_type}, '
            f'{self.output_dim}, {self.name_module_input}, '
            f'{self.input_reused}, {self.permute_in}, {self.permute_out}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class LayerModifier(Layer):
    """
    Represents a type of layer that applies transformations or
        adjustments to other layers, enhancing their behavior or
        performance.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_module_input (str): The name of the layer from which
            the inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the
            layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name for 
            this layer.
        output_var (str): Inherited from Layer. Output variable name
            for this layer.
    """

    def __repr__(self):
        return (
            f'LayerModifier({self.name}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class NormalizationLayer(LayerModifier):
    """
    Represents a type of layer that applies normalization techniques.

    Args:
        name (str): The name of the layer.
        eps (float): Epsilon for numerical stability.
        affine (bool): Whether to learn affine parameters gamma/beta.
        permute_in (bool): Whether to permute input dimensions before 
            processing.
        permute_out (bool): Whether to permute output dimensions after 
            processing.
        actv_func (str): The type of the activation function.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        eps (float): Epsilon value for numerical stability.
        affine (bool): Whether to learn affine parameters (gamma/beta).
        permute_in (bool): Inherited from Layer. Whether to permute
            input dimensions.
        permute_out (bool): Inherited from Layer. Whether to permute
            output dimensions.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """
    def __init__(self, name: str, eps: float = 1e-5, affine: bool = True,
                 permute_in: bool = False, permute_out: bool = False,
                 actv_func: str | None = None, name_module_input: str | None = None,
                 input_reused: bool = False, is_layer_call: bool = False,
                 input_var: str | None = None, output_var: str | None = None):
        super().__init__(name, actv_func, name_module_input, input_reused,
                         is_layer_call, permute_in, permute_out, input_var,
                         output_var)
        self.eps: float = eps
        self.affine: bool = affine

    @property
    def eps(self) -> float:
        """float: Get the epsilon value for numerical stability."""
        return self.__eps

    @eps.setter
    def eps(self, eps: float):
        """float: Set the epsilon value for numerical stability."""
        self.__eps = eps

    @property
    def affine(self) -> bool:
        """bool: Get whether to learn affine parameters
        (gamma/beta)."""
        return self.__affine

    @affine.setter
    def affine(self, affine: bool):
        """bool: Set whether to learn affine parameters
        (gamma/beta)."""
        self.__affine = affine

    def __repr__(self):
        return (
            f'NormalizationLayer({self.name}, {self.eps}, {self.affine}, '
            f'{self.permute_in}, {self.permute_out}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class BatchNormLayer(NormalizationLayer):
    """
    Represents a type of layer that normalizes inputs within
    mini-batches to maintain consistent mean and variance, enhancing
    training speed and stability.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        num_features (int): The number of channels or features in each
            input sample.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the
            input data to be normalized using batch normalization.
        eps (float): Epsilon for numerical stability.
        momentum (float): Momentum for running mean/variance.
        affine (bool): Whether to learn affine parameters gamma/beta.
        track_running_stats (bool): Whether to track running 
            mean/variance.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        num_features (int): The number of channels or features in each
            input sample.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the
            input data to be normalized using batch normalization.
        eps (float): Inherited from NormalizationLayer. Epsilon value 
            for numerical stability.
        momentum (float): Momentum for running mean/variance.
        affine (bool): Inherited from NormalizationLayer. Whether 
            to learn affine parameters (gamma/beta).
        track_running_stats (bool): Whether to track running
            mean/variance.
        permute_in (bool): Whether the dimensions of the input need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        permute_out (bool): Whether the dimensions of the output need
            to be permuted. Relevant for PyTorch. It is used to make
            PyTorch model equivalent to TensorFlow model.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name for 
            this layer.
        output_var (str): Inherited from Layer. Output variable name
            for this layer.
    """
    def __init__(self, name: str, num_features: int, dimension: str,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True,
                 permute_in: bool = False, permute_out: bool = False,
                 actv_func: str | None = None, name_module_input: str | None = None,
                 input_reused: bool = False, is_layer_call: bool = False,
                 input_var: str | None = None,
                 output_var: str | None = None):
        super().__init__(name, eps, affine, permute_in, permute_out,
                         actv_func, name_module_input, input_reused,
                         is_layer_call, input_var, output_var)
        self.num_features: int = num_features
        self.dimension: str = dimension
        self.momentum: float = momentum
        self.track_running_stats: bool = track_running_stats

    @property
    def num_features(self) -> int:
        """int: Get the number of channels or features."""
        return self.__num_features

    @num_features.setter
    def num_features(self, num_features: int):
        """int: Set the number of channels or features."""
        self.__num_features = num_features

    @property
    def dimension(self) -> str:
        """str: Get the dimensionality of the input data to be
        normalized."""
        return self.__dimension

    @dimension.setter
    def dimension(self, dimension: str):
        """
        str: Set the dimensionality of the input data to be normalized.

        Raises:
            ValueError: If the dimensionality of the input data is none
            of these: '1D', '2D', or '3D'.
        """

        if dimension not in ['1D', '2D', '3D']:
            raise ValueError ("Invalid data dimensionality")
        self.__dimension = dimension

    @property
    def momentum(self) -> float:
        """float: Get the momentum for running mean/variance."""
        return self.__momentum

    @momentum.setter
    def momentum(self, momentum: float):
        """float: Set the momentum for running mean/variance."""
        self.__momentum = momentum

    @property
    def track_running_stats(self) -> bool:
        """bool: Get whether to track running mean/variance."""
        return self.__track_running_stats

    @track_running_stats.setter
    def track_running_stats(self, track_running_stats: bool):
        """bool: Set whether to track running mean/variance."""
        self.__track_running_stats = track_running_stats

    def __repr__(self):
        return (
            f'BatchNormLayer({self.name}, {self.num_features}, '
            f'{self.dimension}, {self.eps}, {self.momentum}, {self.affine}, '
            f'{self.track_running_stats}, {self.actv_func}, '
            f'{self.permute_in}, {self.permute_out}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class LayerNormLayer(NormalizationLayer):
    """
    Represents a type of layer that normalizes the inputs across
    the features of a single data sample, rather than across
    the batch, to stabilize and accelerate training by reducing internal
    covariate shift.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        normalized_shape (list[int]): A list of integers specifying 
            the shape of the trailing dimensions over which layer 
            normalization is applied. These correspond to the last N 
            dimensions of the input tensor.
        eps (float): Epsilon for numerical stability.
        affine (bool): Whether to learn affine parameters gamma/beta.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        normalized_shape (list[int]): A list of integers specifying 
            the shape of the trailing dimensions over which layer 
            normalization is applied. These correspond to the last N 
            dimensions of the input tensor.
        eps (float): Inherited from NormalizationLayer. Epsilon value 
            for numerical stability.
        affine (bool): Inherited from NormalizationLayer. Whether 
            to learn affine parameters (gamma/beta).
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """
    def __init__(self, name: str, normalized_shape: list[int],
                 eps: float = 1e-5, affine: bool = True,
                 actv_func: str | None = None, name_module_input: str | None = None,
                 input_reused: bool = False, is_layer_call: bool = False,
                 input_var: str | None = None, output_var: str | None = None):
        super().__init__(name, eps, affine, False, False, actv_func,
                         name_module_input, input_reused, is_layer_call,
                         input_var, output_var)
        self.normalized_shape: list[int] = normalized_shape

    @property
    def normalized_shape(self) -> list[int]:
        """list[int]: Get the list containing the dimensions or axis
        indices over which layer normalization is applied."""
        return self.__normalized_shape

    @normalized_shape.setter
    def normalized_shape(self, normalized_shape: list[int]):
        """list[int]: Set the list containing the dimensions or axis
        indices over which layer normalization is applied."""
        self.__normalized_shape = normalized_shape

    def __repr__(self):
        return (
            f'LayerNormLayer({self.name}, {self.normalized_shape}, '
            f'{self.eps}, {self.affine}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class DropoutLayer(LayerModifier):
    """
    Represents a type of layer that randomly sets a fraction of input
    units to zero during training to prevent overfitting and improve
    generalization.

    Args:
        name (str): The name of the layer.
        rate (float): It represents a float between 0 and 1. Fraction
            of the input units to drop.
        dimension (str | None): The dimensionality for spatial dropout
            ('1', '2', '3'). None for regular element-wise dropout.
        permute_in (bool): Whether to permute input dimensions before 
            processing.
        permute_out (bool): Whether to permute output dimensions after 
            processing.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        rate (float): It represents a float between 0 and 1. Fraction
            of the input units to drop.
        dimension (str | None): The dimensionality for spatial dropout.
        permute_in (bool): Inherited from Layer. Whether to permute
            input dimensions.
        permute_out (bool): Inherited from Layer. Whether to permute
            output dimensions.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """
    def __init__(self, name: str, rate: float, dimension: str | None = None,
                 permute_in: bool = False, permute_out: bool = False,
                 name_module_input: str | None = None, input_reused: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        super().__init__(name, None, name_module_input, input_reused,
                         is_layer_call, permute_in, permute_out, input_var,
                         output_var)
        self.rate: float = rate
        self.dimension: str | None = dimension

    @property
    def rate(self) -> float:
        """float: Get the fraction of the input units to drop."""
        return self.__rate

    @rate.setter
    def rate(self, rate: float):
        """float: Set the fraction of the input units to drop."""
        self.__rate = rate

    @property
    def dimension(self) -> str | None:
        """str | None: Get the dimensionality for spatial dropout."""
        return self.__dimension

    @dimension.setter
    def dimension(self, dimension: str | None):
        """str | None: Set the dimensionality for spatial dropout."""
        self.__dimension = dimension

    def __repr__(self):
        return (
            f'DropoutLayer({self.name}, {self.rate}, {self.dimension}, '
            f'{self.permute_in}, {self.permute_out}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class RNN(Layer):
    """
    Represents a type of layer used in recurrent neural networks (RNN)
    for processing sequential data by using memory from previous steps
    to inform current outputs.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input
            features.
        hidden_size (int): It represents the number of units in the
            hidden state, which captures the network's internal
            representation of the input sequence.
        bidirectional (bool): Whether the layer is bidirectional or not.
        dropout (float): If non-zero, it introduces a Dropout layer on
            the outputs of the current sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are
            provided as (batch, seq, feature) instead of (seq, batch,
            feature). Only relevant to PyTorch.
        bias (bool): If True, the layer uses bias weights.
            Default is True.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        return_type (str): Whether to return the hidden states, the
            last output in the output sequence or the full sequence.
        hx_source (str): The name of the source layer for initial
            hidden state (used in encoder-decoder architectures).
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        hidden_state_var (str): Variable name for hidden state output.
        cell_state_var (str): Variable name for cell state output
            (LSTM only).
        hidden_unused (bool): Whether hidden state output is unused by 
            subsequent layers.
        cell_unused (bool): Whether cell state output is unused by 
            subsequent layers (LSTM only).
        hidden_subscript_source (str): Source variable for hidden
            subscript assignment (e.g., 'h' in 'x = h').
        hidden_subscript_target (str): Target variable for hidden
            subscript assignment (e.g., 'x' in 'x = h').
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        input_size (int): It represents the dimensionality of the input
            features.
        hidden_size (int): It represents the number of units in the
            hidden state, which captures the network's internal
            representation of the input sequence.
        bidirectional (bool): Whether the layer is bidirectional
            or not.
        dropout (float): If non-zero, it introduces a Dropout layer on
            the outputs of the current sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are
            provided as (batch, seq, feature) instead of (seq, batch,
            feature). Only relevant to PyTorch.
        bias (bool): If True, the layer uses bias weights.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        return_type (str): Whether to return the hidden states, the
            last output in the output sequence or the full sequence.
        hx_source (str): The name of the source layer for initial
            hidden state.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
        hidden_state_var (str): Variable name for hidden state output.
        cell_state_var (str): Variable name for cell state output
            (LSTM only).
        hidden_unused (bool): Whether hidden state output is unused 
            by subsequent layers.
        cell_unused (bool): Whether cell state output is unused 
            by subsequent layers (LSTM only).
        hidden_subscript_source (str): Source variable for hidden
            subscript assignment (e.g., 'h' in 'x = h').
        hidden_subscript_target (str): Target variable for hidden
            subscript assignment (e.g., 'x' in 'x = h').
    """
    def __init__(self, name: str, hidden_size: int, return_type: str = "full",
                 input_size: int | None = None, bidirectional: bool = False,
                 dropout: float = 0.0, batch_first: bool = True,
                 bias: bool = True, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 hx_source: str | None = None, is_layer_call: bool = False,
                 hidden_state_var: str | None = None, cell_state_var: str | None = None,
                 hidden_unused: bool = False, cell_unused: bool = False,
                 hidden_subscript_source: str | None = None,
                 hidden_subscript_target: str | None = None, input_var: str | None = None,
                 output_var: str | None = None):
        super().__init__(name, actv_func, name_module_input, input_reused,
                         is_layer_call, False, False, input_var, output_var)
        self.bidirectional: bool = bidirectional
        self.dropout: float = dropout
        self.batch_first: bool = batch_first
        self.bias: bool = bias
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.return_type: str = return_type
        self.hx_source: str = hx_source
        self.hidden_state_var: str = hidden_state_var
        self.cell_state_var: str = cell_state_var
        self.hidden_unused: bool = hidden_unused
        self.cell_unused: bool = cell_unused
        self.hidden_subscript_source: str = hidden_subscript_source
        self.hidden_subscript_target: str = hidden_subscript_target

    @property
    def input_size(self) -> int:
        """int: Get the dimensionality of the input features
        of the layer."""
        return self.__input_size

    @input_size.setter
    def input_size(self, input_size: int):
        """int: Set the dimensionality of the input features
        of the layer."""
        self.__input_size = input_size

    @property
    def hidden_size(self) -> int:
        """int: Get the number of units in the hidden state."""
        return self.__hidden_size

    @hidden_size.setter
    def hidden_size(self, hidden_size: int):
        """int: Set the number of units in the hidden state."""
        self.__hidden_size = hidden_size

    @property
    def bidirectional(self) -> bool:
        """bool: Get whether the layer is bidirectional or not."""
        return self.__bidirectional

    @bidirectional.setter
    def bidirectional(self, bidirectional: bool):
        """bool: Set whether the layer is bidirectional or not."""
        self.__bidirectional = bidirectional

    @property
    def dropout(self) -> float:
        """float: Get the dropout ratio of the layer."""
        return self.__dropout

    @dropout.setter
    def dropout(self, dropout: float):
        """float: Set the dropout ratio of the layer."""
        self.__dropout = dropout

    @property
    def batch_first(self) -> bool:
        """bool: Get whether the input and output tensors are
        provided as (batch, seq, feature)."""
        return self.__batch_first

    @batch_first.setter
    def batch_first(self, batch_first: bool):
        """bool: Set whether the input and output tensors are
        provided as (batch, seq, feature)."""
        self.__batch_first = batch_first

    @property
    def bias(self) -> bool:
        """bool: Get whether the layer uses bias weights."""
        return self.__bias

    @bias.setter
    def bias(self, bias: bool):
        """bool: Set whether the layer uses bias weights."""
        self.__bias = bias

    @property
    def return_type(self) -> str:
        """str: Whether to return the hidden states, the last output
        in the output sequence, the full sequence, or both output
        and hidden."""
        return self.__return_type

    @return_type.setter
    def return_type(self, return_type: str):
        """
        str: Whether to return the hidden states, the last output in
            the output sequence, the full sequence, or both output
            and hidden.
        Raises:
            ValueError: If the return_type is none of these:
            'hidden', 'last', 'full', or 'both'.
        """

        if return_type not in ['hidden', 'last', 'full', 'both']:
            raise ValueError ("Invalid return type")
        self.__return_type = return_type

    @property
    def hx_source(self) -> str:
        """str: Get the name of the source layer for initial
        hidden state."""
        return self.__hx_source

    @hx_source.setter
    def hx_source(self, hx_source: str):
        """str: Set the name of the source layer for initial
        hidden state."""
        self.__hx_source = hx_source

    @property
    def hidden_state_var(self) -> str:
        """str: Get the variable name for hidden state output."""
        return self.__hidden_state_var

    @hidden_state_var.setter
    def hidden_state_var(self, hidden_state_var: str):
        """str: Set the variable name for hidden state output."""
        self.__hidden_state_var = hidden_state_var

    @property
    def cell_state_var(self) -> str:
        """str: Get the variable name for cell state 
        output (LSTM only)."""
        return self.__cell_state_var

    @cell_state_var.setter
    def cell_state_var(self, cell_state_var: str):
        """str: Set the variable name for cell state 
        output (LSTM only)."""
        self.__cell_state_var = cell_state_var

    @property
    def hidden_unused(self) -> bool:
        """bool: Get whether hidden state output is unused 
        by subsequent layers."""
        return self.__hidden_unused

    @hidden_unused.setter
    def hidden_unused(self, hidden_unused: bool):
        """bool: Set whether hidden state output is unused 
        by subsequent layers."""
        self.__hidden_unused = hidden_unused

    @property
    def cell_unused(self) -> bool:
        """bool: Get whether cell state output is unused
        by subsequent layers (LSTM only)."""
        return self.__cell_unused

    @cell_unused.setter
    def cell_unused(self, cell_unused: bool):
        """bool: Set whether cell state output is unused
        by subsequent layers (LSTM only)."""
        self.__cell_unused = cell_unused

    @property
    def hidden_subscript_source(self) -> str:
        """str: Get the source variable for hidden subscript 
        assignment."""
        return self.__hidden_subscript_source

    @hidden_subscript_source.setter
    def hidden_subscript_source(self, hidden_subscript_source: str):
        """str: Set the source variable for hidden subscript
        assignment."""
        self.__hidden_subscript_source = hidden_subscript_source

    @property
    def hidden_subscript_target(self) -> str:
        """str: Get the target variable for hidden subscript
        assignment."""
        return self.__hidden_subscript_target

    @hidden_subscript_target.setter
    def hidden_subscript_target(self, hidden_subscript_target: str):
        """str: Set the target variable for hidden subscript
        assignment."""
        self.__hidden_subscript_target = hidden_subscript_target

    def __repr__(self):
        return (
            f'RNN({self.name}, {self.hidden_size}, {self.return_type}, '
            f'{self.input_size}, {self.bidirectional}, {self.dropout}, '
            f'{self.batch_first}, {self.bias}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.hx_source}, {self.is_layer_call}, {self.input_var}, '
            f'{self.output_var}, {self.hidden_state_var}, '
            f'{self.cell_state_var}, {self.hidden_unused}, '
            f'{self.cell_unused}, {self.hidden_subscript_source}, '
            f'{self.hidden_subscript_target})'
        )

class SimpleRNNLayer(RNN):
    """
    Represents a fully-connected RNN layer where the output is to
    be fed back as the new input.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input
            features.
        hidden_size (int): It represents the number of units in the
            hidden state, which captures the network's internal
            representation of the input sequence.
        bidirectional (bool): Whether the layer is bidirectional
            or not.
        dropout (float): If non-zero, it introduces a Dropout layer on
            the outputs of the RNN sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are
            provided as (batch, seq, feature) instead of (seq, batch,
            feature). Only relevant to PyTorch.
        bias (bool): If True, the layer uses bias weights.
            Default is True.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        return_type (str): Whether to return the hidden states, the
            last output in the output sequence or the full sequence.
        hx_source (str): The name of the source layer for initial 
            hidden state.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.
        hidden_state_var (str): Variable name for hidden state output.
        hidden_unused (bool): Whether hidden state output is unused.
        hidden_subscript_source (str): Source variable for hidden
            subscript assignment.
        hidden_subscript_target (str): Target variable for hidden
            subscript assignment.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        input_size (int): Inherited from RNN. It represents the
            dimensionality of the input features.
        hidden_size (int): Inherited from RNN. It represents the number
            of units in the hidden state, which captures the network's
            internal representation of the input sequence.
        bidirectional (bool): Inherited from RNN. Whether the layer is
            bidirectional or not.
        dropout (float): Inherited from RNN. If non-zero, it introduces
            a Dropout layer on the outputs of the RNN sub layers except
            the last one.
        batch_first (bool): Inherited from RNN. If True, the input and
            output tensors are provided as (batch, seq, feature)
            instead of (seq, batch, feature). Only relevant to PyTorch.
        bias (bool): Inherited from RNN. If True, the layer uses bias
            weights.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        return_type (str): Inherited from RNN. Whether to return the
            hidden states, the last output in the output sequence or
            the full sequence.
        hx_source (str): Inherited from RNN. The name of the source
            layer for initial hidden state.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
        hidden_state_var (str): Inherited from RNN. Variable name 
            for hidden state output.
        hidden_unused (bool): Inherited from RNN. Whether hidden state
            output is unused.
        hidden_subscript_source (str): Inherited from RNN. Source
            variable for hidden subscript assignment.
        hidden_subscript_target (str): Inherited from RNN. Target
            variable for hidden subscript assignment.
    """

    def __repr__(self):
        return (
            f'SimpleRNNLayer({self.name}, {self.hidden_size}, '
            f'{self.return_type}, {self.input_size}, {self.bidirectional}, '
            f'{self.dropout}, {self.batch_first}, {self.bias}, '
            f'{self.actv_func}, {self.name_module_input}, '
            f'{self.input_reused}, {self.hx_source}, {self.is_layer_call}, '
            f'{self.input_var}, {self.output_var}, '
            f'{self.hidden_state_var}, {None}, {self.hidden_unused}, '
            f'{None}, {self.hidden_subscript_source}, '
            f'{self.hidden_subscript_target})'
        )

class LSTMLayer(RNN):
    """
    Represents a Long Short-Term Memory layer.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the
            input features.
        hidden_size (int): It represents the number of units in the
            hidden state, which captures the network's internal
            representation of the input sequence.
        bidirectional (bool): Whether the layer is bidirectional or not.
        dropout (float): If non-zero, it introduces a Dropout layer on
            the outputs of the LSTM sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are
            provided as (batch, seq, feature) instead of (seq, batch,
            feature). Only relevant to PyTorch.
        bias (bool): If True, the layer uses bias weights.
            Default is True.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        return_type (str): Whether to return the hidden states, the
            last output in the output sequence or the full sequence.
        hx_source (str): The name of the source layer for initial
            hidden state (used in encoder-decoder architectures).
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.
        hidden_state_var (str): Variable name for hidden state output.
        cell_state_var (str): Variable name for cell state output
            (LSTM only).
        hidden_unused (bool): Whether hidden state output is unused.
        cell_unused (bool): Whether cell state output is unused
            (LSTM only).
        hidden_subscript_source (str): Source variable for hidden
            subscript assignment.
        hidden_subscript_target (str): Target variable for hidden
            subscript assignment.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        input_size (int): Inherited from RNN. It represents the
            dimensionality of the input features.
        hidden_size (int): Inherited from RNN. It represents the number
            of units in the hidden state, which captures the network's
            internal representation of the input sequence.
        bidirectional (bool): Inherited from RNN. Whether the layer is
            bidirectional or not.
        dropout (float): Inherited from RNN. If non-zero, it introduces
            a Dropout layer on the outputs of the LSTM sub layers
            except the last one.
        batch_first (bool): Inherited from RNN. If True, the input and
            output tensors are provided as (batch, seq, feature)
            instead of (seq, batch, feature). Only relevant to PyTorch.
        bias (bool): Inherited from RNN. If True, the layer uses bias
            weights.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        return_type (str): Inherited from RNN. Whether to return the
            hidden states, the last output in the output sequence or
            the full sequence.
        hx_source (str): Inherited from RNN. The name of the source
            layer for initial hidden state.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
        hidden_state_var (str): Inherited from RNN. Variable name 
            for hidden state output.
        cell_state_var (str): Inherited from RNN. Variable name 
            for cell state output.
        hidden_unused (bool): Inherited from RNN. Whether hidden state 
            output is unused.
        cell_unused (bool): Inherited from RNN. Whether cell state
            output is unused.
        hidden_subscript_source (str): Inherited from RNN. Source
            variable for hidden subscript assignment.
        hidden_subscript_target (str): Inherited from RNN. Target
            variable for hidden subscript assignment.
    """

    def __repr__(self):
        return (
            f'LSTMLayer({self.name}, {self.hidden_size}, {self.return_type}, '
            f'{self.input_size}, {self.bidirectional}, {self.dropout}, '
            f'{self.batch_first}, {self.bias}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.hx_source}, {self.is_layer_call}, {self.input_var}, '
            f'{self.output_var}, {self.hidden_state_var}, '
            f'{self.cell_state_var}, {self.hidden_unused}, '
            f'{self.cell_unused}, {self.hidden_subscript_source}, '
            f'{self.hidden_subscript_target})'
        )

class GRULayer(RNN):
    """
    Represents a Gated Recurrent Unit layer.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input
            features.
        hidden_size (int): It represents the number of units in the
            hidden state, which captures the network's internal
            representation of the input sequence.
        bidirectional (bool): Whether the layer is bidirectional
            or not.
        dropout (float): If non-zero, it introduces a Dropout layer on
            the outputs of the GRU sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are
            provided as (batch, seq, feature) instead of (seq, batch,
            feature). Only relevant to PyTorch.
        bias (bool): If True, the layer uses bias weights.
            Default is True.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        return_type (str): Whether to return the hidden states, the
            last output in the output sequence or the full sequence.
        hx_source (str): The name of the source layer for initial
            hidden state (used in encoder-decoder architectures).
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.
        hidden_state_var (str): Variable name for hidden state output.
        hidden_unused (bool): Whether hidden state output is unused.
        hidden_subscript_source (str): Source variable for hidden
            subscript assignment.
        hidden_subscript_target (str): Target variable for hidden
            subscript assignment.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        input_size (int): Inherited from RNN. It represents the
            dimensionality of the input features.
        hidden_size (int): Inherited from RNN. It represents the number
            of units in the hidden state, which captures the network's
            internal representation of the input sequence.
        bidirectional (bool): Inherited from RNN. Whether the layer is
            bidirectional or not.
        dropout (float): Inherited from RNN. If non-zero, it introduces
            a Dropout layer on the outputs of the GRU sub layers except
            the last one.
        batch_first (bool): Inherited from RNN. If True, the input and
            output tensors are provided as (batch, seq, feature)
            instead of (seq, batch, feature). Only relevant to PyTorch.
        bias (bool): Inherited from RNN. If True, the layer uses bias
            weights.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        return_type (str): Inherited from RNN. Whether to return
            the hidden states, the last output in the output sequence
            or the full sequence.
        hx_source (str): Inherited from RNN. The name of the source
            layer for initial hidden state.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
        hidden_state_var (str): Inherited from RNN. Variable name for 
            hidden state output.
        hidden_unused (bool): Inherited from RNN. Whether hidden state 
            output is unused.
        hidden_subscript_source (str): Inherited from RNN. Source
            variable for hidden subscript assignment.
        hidden_subscript_target (str): Inherited from RNN. Target
            variable for hidden subscript assignment.
    """

    def __repr__(self):
        return (
            f'GRULayer({self.name}, {self.hidden_size}, {self.return_type}, '
            f'{self.input_size}, {self.bidirectional}, {self.dropout}, '
            f'{self.batch_first}, {self.bias}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.hx_source}, {self.is_layer_call}, {self.input_var}, '
            f'{self.output_var}, {self.hidden_state_var}, {None}, '
            f'{self.hidden_unused}, {None}, {self.hidden_subscript_source}, '
            f'{self.hidden_subscript_target})'
        )

class GeneralLayer(Layer):
    """
    Represents a layer designed to handle general operations and
    transformations.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_module_input (str): The name of the layer from which
            the inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        name_module_input (str): Inherited from Layer. The name of the
            layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """

    def __repr__(self):
        return (
            f'GeneralLayer({self.name}, {self.actv_func}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class LinearLayer(GeneralLayer):
    """
    Represents a densely-connected NN layer that applies a linear
    transformation to the input data.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        in_features (int): It represents the size of each input sample.
        out_features (int): It represents the size of each output
            sample.
        bias (bool): If True, adds a learnable bias to the output.
        name_module_input (str): The name of the layer from which the
            inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        in_features (int): It represents the size of each input sample.
        out_features (int): It represents the size of each output
            sample.
        bias (bool): If True, adds a learnable bias to the output.
        name_module_input (str): Inherited from Layer. The name of
            the layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input
            to this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """
    def __init__(self, name: str, out_features: int, in_features: int | None = None,
                 bias: bool = True, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        super().__init__(name, actv_func, name_module_input, input_reused,
                         is_layer_call, False, False, input_var, output_var)
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.bias: bool = bias

    @property
    def in_features(self) -> int:
        """int: Get the size of the input sample."""
        return self.__in_features

    @in_features.setter
    def in_features(self, in_features: int):
        """int: Set the size of the input sample."""
        self.__in_features = in_features

    @property
    def out_features(self) -> int:
        """int: Get the size of the output sample."""
        return self.__out_features

    @out_features.setter
    def out_features(self, out_features: int):
        """int: Set the size of the output sample."""
        self.__out_features = out_features

    @property
    def bias(self) -> bool:
        """bool: Get whether the layer uses bias."""
        return self.__bias

    @bias.setter
    def bias(self, bias: bool):
        """bool: Set whether the layer uses bias."""
        self.__bias = bias

    def __repr__(self):
        return (
            f'LinearLayer({self.name}, {self.actv_func}, '
            f'{self.out_features}, {self.in_features}, {self.bias}, '
            f'{self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class FlattenLayer(GeneralLayer):
    """
    Represents a layer that flattens a contiguous range of dims
    into a tensor.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        start_dim (int): The first dimension to flatten.
        end_dim (int): The last dim to flatten.
        name_module_input (str): The name of the layer from which
            the inputs originate.
        input_reused (bool): Whether the input to this layer is
            reused as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        start_dim (int): The first dimension to flatten.
        end_dim (int): The last dim to flatten.
        name_module_input (str): Inherited from Layer. The name of
            the layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input
            to this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """
    def __init__(self, name: str, start_dim: int = 1, end_dim: int = -1,
                 actv_func: str | None = None, name_module_input: str | None = None,
                 input_reused: bool = False, is_layer_call: bool = False,
                 input_var: str | None = None, output_var: str | None = None):
        super().__init__(name, actv_func, name_module_input, input_reused,
                         is_layer_call, False, False, input_var, output_var)
        self.start_dim: int = start_dim
        self.end_dim: int = end_dim

    @property
    def start_dim(self) -> int:
        """int: Get the first dimension to flatten."""
        return self.__start_dim

    @start_dim.setter
    def start_dim(self, start_dim: int):
        """int: Set the first dimension to flatten."""
        self.__start_dim = start_dim

    @property
    def end_dim(self) -> int:
        """int: Get the last dimension to flatten."""
        return self.__end_dim

    @end_dim.setter
    def end_dim(self, end_dim: int):
        """int: Set the last dimension to flatten."""
        self.__end_dim = end_dim

    def __repr__(self):
        return (
            f'FlattenLayer({self.name}, {self.actv_func}, {self.start_dim}, '
            f'{self.end_dim}, {self.name_module_input}, {self.input_reused}, '
            f'{self.is_layer_call}, {self.input_var}, {self.output_var})'
        )

class EmbeddingLayer(GeneralLayer):
    """
    Represents a layer that learns dense vector representations of
    the input data.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        num_embeddings (int): The size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (int): If specified, the entries at padding_idx 
            are masked and do not contribute to the gradient.
            Default is None.
        permute_in (bool): Whether to permute input dimensions before 
            processing.
        permute_out (bool): Whether to permute output dimensions after 
            processing.
        name_module_input (str): The name of the layer from which
            the inputs originate.
        input_reused (bool): Whether the input to this layer is reused
            as input to another layer.
        is_layer_call (bool): True if this represents a call to 
            an already-defined layer (layer reuse).
        input_var (str): Input variable name for this layer.
        output_var (str): Output variable name for this layer.

    Attributes:
        name (str): Inherited from Layer. It represents the name of
            the layer.
        actv_func (str): Inherited from Layer. It represents the type
            of the activation function.
        num_embeddings (int): The size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (int): If specified, the entries at padding_idx
            are masked.
        permute_in (bool): Whether to permute input dimensions before 
            processing.
        permute_out (bool): Whether to permute output dimensions after 
            processing.
        name_module_input (str): Inherited from Layer. The name of
            the layer from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to
            this layer is reused as input to another layer.
        is_layer_call (bool): Inherited from Layer. True if this is a
            call to an already-defined layer.
        input_var (str): Inherited from Layer. Input variable name 
            for this layer.
        output_var (str): Inherited from Layer. Output variable name 
            for this layer.
    """
    def __init__(self, name: str, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None, permute_in: bool = False,
                 permute_out: bool = False, actv_func: str | None = None,
                 name_module_input: str | None = None, input_reused: bool = False,
                 is_layer_call: bool = False, input_var: str | None = None,
                 output_var: str | None = None):
        super().__init__(name, actv_func, name_module_input, input_reused,
                         is_layer_call, False, False, input_var, output_var)
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.padding_idx: int = padding_idx
        self.permute_in: bool = permute_in
        self.permute_out: bool = permute_out

    @property
    def num_embeddings(self) -> int:
        """int: Get the size of the dictionary of embeddings."""
        return self.__num_embeddings

    @num_embeddings.setter
    def num_embeddings(self, num_embeddings: int):
        """int: Set the size of the dictionary of embeddings."""
        self.__num_embeddings = num_embeddings

    @property
    def embedding_dim(self) -> int:
        """int: Get the size of each embedding vector."""
        return self.__embedding_dim

    @embedding_dim.setter
    def embedding_dim(self, embedding_dim: int):
        """int: Set the size of each embedding vector."""
        self.__embedding_dim = embedding_dim

    @property
    def padding_idx(self) -> int:
        """int: Get the padding index."""
        return self.__padding_idx

    @padding_idx.setter
    def padding_idx(self, padding_idx: int):
        """int: Set the padding index."""
        self.__padding_idx = padding_idx

    def __repr__(self):
        return (
            f'EmbeddingLayer({self.name}, {self.actv_func}, '
            f'{self.num_embeddings}, {self.embedding_dim}, '
            f'{self.padding_idx}, {self.name_module_input}, '
            f'{self.input_reused}, {self.is_layer_call}, '
            f'{self.input_var}, {self.output_var})'
        )

class Feature(NamedElement):
    """
    A feature is a measurable property or characteristic of an object
    used to represent and describe it within a dataset.

    Args:
        name (str): The name of the feature.

    Attributes:
        name (str): The name of the feature.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self):
        return f'Feature({self.name}'

class Label:
    """
    A label is a value assigned to an observation, representing
    the target variable for prediction.

    Args:
        col_name (str): The name of the column containing the labels.
        label_name (str): The name of a label in the dataset. If
            the prediction task is regression, it can be omitted.

    Attributes:
        col_name (str): The name of the column containing the labels.
        label_name (str): The name of a label in the dataset. If
            the prediction task is regression, it can be omitted.
    """
    def __init__(self, col_name: str, label_name: str | None = None):
        self.col_name: str = col_name
        self.label_name: str = label_name

    @property
    def col_name(self) -> str:
        """str: Get the name of the column containing the labels."""
        return self.__col_name

    @col_name.setter
    def col_name(self, col_name: str):
        """str: Set the name of the column containing the labels."""
        self.__col_name = col_name

    @property
    def label_name(self) -> str:
        """str: Get the name of the label."""
        return self.__label_name

    @label_name.setter
    def label_name(self, label_name: str):
        """str: Set the name of the label."""
        self.__label_name = label_name

    def __repr__(self):
        return f'Label({self.col_name}, {self.label_name})'

class Image(Feature):
    """
    Image represents features designed for handling data with spatial
    characteristics, typically including attributes such as height
    and width.

    Args:
        shape (list[int]): The shape of the image in the form
            [height, width, channels]. Default to [256, 256].
        normalize (bool): If true, the images will be normalized
            to zero mean and unit standard deviation.

    Attributes:
        shape (list[int]): The shape of the image in the form
            [height, width, channels]. Default to [256, 256].
        normalize (bool): If true, the images will be normalized
            to zero mean and unit standard deviation.

    """
    def __init__(self, shape: list[int] | None = None,
                 normalize: bool = False):
        if shape is None:
            shape = [256, 256]
        self.shape: list[int] = shape
        self.normalize: bool = normalize

    @property
    def shape(self) -> list[int]:
        """list[int]: Get the shape of the image."""
        return self.__shape

    @shape.setter
    def shape(self, shape: list[int]):
        """list[int]: Set the shape of the image."""
        self.__shape = shape

    @property
    def normalize(self) -> bool:
        """bool: If true, the images will be normalized to zero mean
        and unit standard deviation."""
        return self.__normalize

    @normalize.setter
    def normalize(self, normalize: bool):
        """bool: If true, the images will be normalized to zero mean
        and unit standard deviation."""
        self.__normalize = normalize

    def __repr__(self):
        return f'Image({self.shape}, {self.normalize})'

class Structured(Feature):
    """
    Represents features organized in a systematic manner, typically
    with well-defined columns and rows, often found in tabular 
    datasets.

    Args:
        name (str): The name of the feature.

    Attributes:
        name (str): Inherited from Feature. It represents the name of
            the feature.
    """

    def __repr__(self):
        return f'Structured({self.name})'

class Dataset(NamedElement):
    """
    Represents the collection of data instances used for training or
    evaluation, where each instance comprises features and
    corresponding labels.

    Args:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing
            the dataset.
        task_type (str): The type of prediction task associated with
            the dataset.
        input_format (str): The format of the input dataset.
        image (Image): An image instance that contains the shape
            desired for the images if input_format parameter is set
            to 'images'.
        labels (set[Label]): The set of labels in the dataset.

    Attributes:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing
            the dataset.
        task_type (str): The type of prediction task associated with
            the dataset.
        input_format (str): The format of the input dataset.
        image (Image): An image instance that contains the shape
            desired for the images if input_format parameter is set
            to 'images'.
        labels (set[Label]): The set of labels in the dataset.
    """
    def __init__(self, name: str, path_data: str, task_type: str | None = None,
                 input_format: str | None = None, image: Image | None = None,
                 labels: set[Label] | None = None):
        if labels is None:
            labels = set()
        super().__init__(name)
        self.path_data: str = path_data
        # Initialize backing fields unconditionally so the getters never raise
        # AttributeError on a Dataset constructed without these optional
        # values. The setters validate against allowlists and so cannot
        # accept ``None``; assign to the mangled attribute directly
        # to bypass setter validation.
        self.__task_type: str | None = None
        self.__input_format: str | None = None
        if task_type is not None:
            self.task_type = task_type
        if input_format is not None:
            self.input_format = input_format
        self.image: Image = image
        self.labels: set[Label] = labels

    @property
    def path_data(self) -> str:
        """str: Get the directory location containing the dataset."""
        return self.__path_data

    @path_data.setter
    def path_data(self, path_data: str):
        """str: Set the directory location containing the dataset."""
        self.__path_data = path_data

    @property
    def task_type(self) -> str:
        """str: Get the type of prediction task associated with
        the dataset."""
        return self.__task_type

    @task_type.setter
    def task_type(self, task_type: str):
        """
        str: Set the type of prediction task associated with
        the dataset.

        Raises:
            ValueError: If task_type is not one of the allowed
            options: 'binary', 'multi_class', and 'regression'
        """

        if task_type not in ['binary', 'multi_class', 'regression']:
            raise ValueError(f"Invalid value of task_type: '{task_type}'")

        self.__task_type = task_type

    @property
    def input_format(self) -> str:
        """str: Get the format of the input dataset."""
        return self.__input_format

    @input_format.setter
    def input_format(self, input_format: str):
        """
        str: Set the format of the input dataset.

        Raises:
            ValueError: If input_format is not one of the allowed
            options: 'csv' and 'images'
        """
        if input_format not in ['csv', 'images']:
            raise ValueError(
                f"Invalid value of input_format: '{input_format}'"
            )
        self.__input_format = input_format

    @property
    def image(self) -> Image:
        """Image: Get the dimensions of the images."""
        return self.__image

    @image.setter
    def image(self, image: Image):
        """Image: Set the dimensions of the images."""
        self.__image = image

    def add_image(self, image: Image):
        """Image: Add the dimensions of the image."""
        self.__image = image
        return self

    @property
    def labels(self) -> set[Label]:
        """set[Label]: Get the set of labels."""
        return self.__labels

    @labels.setter
    def labels(self, labels: set[Label]):
        """set[Label]: Set the set of labels."""
        self.__labels = labels

    def add_label(self, label: Label):
        """Label: add a label to the set of labels."""
        self.__labels.add(label)
        return self

    def __repr__(self):
        return (
            f'Dataset({self.name}, {self.path_data}, {self.task_type}, '
            f'{self.input_format}, {self.image}, {self.labels})'
        )

class Configuration:
    """
    Represents a collection of parameters essential for training and
    evaluating neural networks.

    Args:
        batch_size (int): The number of data samples processed in each
            iteration during training or inference in a neural network.
        epochs (int): It refers to the number of complete passes
            through the entire dataset during the training, with each
            epoch consisting of one iteration through all data samples.
        learning_rate (float): The step size used to update the model
            parameters during optimization.
        optimizer (str): The method or algorithm used to adjust the
            model parameters iteratively during training to minimize
            the loss function and improve model performance.
        loss_function (str): The method used to calculate the
            difference between predicted and actual values, guiding
            the model towards better predictions.
        metrics list[str]: Quantitative measures used to evaluate
            the performance of NN models.
        weight_decay (float): It represents the strength of L2
            regularisation applied to the model's parameters during
            optimization.
        momentum (float): It represents a hyperparameter in
            optimization that helps speed up training by using past
            gradients to smooth out updates.

    Attributes:
        batch_size (int): The number of data samples processed in each
            iteration during training or inference in a neural network.
        epochs (int): It refers to the number of complete passes
            through the entire dataset during the training, with each
            epoch consisting of one iteration through all data samples.
        learning_rate (float): The step size used to update the model
            parameters during optimization.
        optimizer (str): The method or algorithm used to adjust the
            model parameters iteratively during training to minimize
            the loss function and improve model performance.
        loss_function (str): The method used to calculate the
            difference between predicted and actual values, guiding
            the model towards better predictions.
        metrics list[str]: Quantitative measures used to evaluate
            the performance of NN models.
        weight_decay (float): It represents the strength of L2
            regularisation applied to the model's parameters during
            optimization.
        momentum (float): It represents a hyperparameter in
            optimization that helps speed up training by using past
            gradients to smooth out updates.
    """
    def __init__(self, batch_size: int, epochs: int, learning_rate: float,
                 optimizer: str, loss_function: str, metrics: list[str],
                 weight_decay: float = 0, momentum: float = 0):
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.optimizer: str = optimizer
        self.loss_function: str = loss_function
        self.metrics: list[str] = metrics
        self.weight_decay: float = weight_decay
        self.momentum: float = momentum

    @property
    def batch_size(self) -> int:
        """int: Get the number of data samples processed in each
        iteration during training or inference in a neural network."""
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """int: Set the number of data samples processed in each
        iteration during training or inference in a neural network."""
        self.__batch_size = batch_size

    @property
    def epochs(self) -> int:
        """int: Get the number of complete passes through the entire
        dataset during the training."""
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs: int):
        """
        int: Set the number of complete passes through the entire
        dataset during the training."""
        self.__epochs = epochs

    @property
    def learning_rate(self) -> float:
        """float: Get the step size used to update the model parameters
        during optimization."""
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        """float: Set step size used to update the model parameters
        during optimization."""
        self.__learning_rate = learning_rate

    @property
    def optimizer(self) -> str:
        """str: Get the algorithm used to adjust the model parameters
        iteratively during training to minimize the loss function."""
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: str):
        """
        str: Set the algorithm used to adjust the model parameters
            iteratively during training to minimize the loss function.

        Raises:
            ValueError: If optimizer is not one of the allowed
            options: 'sgd', 'adam', 'adamW' and 'adagrad'
        """

        if optimizer not in ['sgd', 'adam', 'adamW', 'adagrad']:
            raise ValueError("Invalid value of optimizer")
        self.__optimizer = optimizer

    @property
    def loss_function(self) -> str:
        """str: Get the method used to calculate the difference between
        predicted and actual values, guiding the model towards better
        predictions."""
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, loss_function: str):
        """
        str: Set the method used to calculate the difference between
            predicted and actual values, guiding the model towards
            better predictions.

        Raises:
            ValueError: If loss_function is not one of the allowed
            options: 'crossentropy', 'binary_crossentropy' and 'mse'
        """

        if loss_function not in [
            'crossentropy', 'binary_crossentropy', 'mse'
        ]:
            raise ValueError(
                f"Invalid value of loss_function: '{loss_function}'"
            )
        self.__loss_function = loss_function

    @property
    def metrics(self) -> list[str]:
        """list[str]: Get the measures for evaluating the performance
        of the model."""
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics: list[str]):
        """
        list[str]: Set the measures for evaluating the performance
            of the model.

        Raises:
            ValueError: If metrics is not one of the allowed options:
            accuracy', 'precision', 'recall', 'f1-score' and 'mae'
        """
        valid_metrics = ['accuracy', 'precision', 'recall', 'f1-score', 'mae']
        if (
            isinstance(metrics, list)
            and all(isinstance(metric, str) for metric in metrics)
        ):
            if all(metric in valid_metrics for metric in metrics):
                self.__metrics = metrics
            else:
                invalid_metrics = [
                    m for m in metrics if m not in valid_metrics
                ]
                raise ValueError(
                    f"Invalid metric(s) provided: {invalid_metrics}"
                )
        else:
            raise ValueError("'metrics' must be a list of strings.")

    @property
    def weight_decay(self) -> float:
        """float: Get the strength of L2 regularisation applied during
        optimization."""
        return self.__weight_decay

    @weight_decay.setter
    def weight_decay(self, weight_decay: float):
        """float: Set the strength of L2 regularisation applied during
        optimization."""
        self.__weight_decay = weight_decay

    @property
    def momentum(self) -> float:
        """float: Get the value of the momentum hyperparameter."""
        return self.__momentum

    @momentum.setter
    def momentum(self, momentum: float):
        """float: Set the value of the momentum hyperparameter."""
        self.__momentum = momentum


    def __repr__(self):
        return (
            f'Configuration({self.batch_size}, {self.epochs}, '
            f'{self.learning_rate}, {self.optimizer}, {self.loss_function}, '
            f'{self.metrics}, {self.weight_decay}, {self.momentum})'
        )

class NN(BehaviorImplementation):
    """
    It is a subclass of the BehaviorImplementation class and comprises
    the fundamental properties and behaviors of a neural network model.

    Args:
        name (str): The name of the neural network model.
        configuration (Configuration): The parameters related to the
            NN training and evaluation.
        train_data (Dataset): The dataset used to train the NN model.
        test_data (Dataset): The dataset used to evaluate the NN model.
        input_var (str): Input variable name for the network.
        return_vars (str): Comma-separated string of variable names
            returned by the forward/call method (e.g., "rep, recon").

    Attributes:
        name (str): The name of the neural network model.
        configuration (Configuration): The parameters related to the
            NN training and evaluation.
        train_data (Dataset): The dataset used to train the NN model.
        test_data (Dataset): The dataset used to evaluate the NN model.
        input_var (str): Input variable name for the network.
        return_vars (str): Comma-separated string of variable names
            returned by the forward/call method (e.g., "rep, recon").
    """
    def __init__(self, name: str, configuration: Configuration | None = None,
                 train_data: Dataset | None = None, test_data: Dataset | None = None,
                 input_var: str | None = None, return_vars: str | None = None):
        super().__init__(name)
        self.configuration: Configuration = configuration
        self.__sub_nns: list[NN] = []
        self.__layers: list[Layer] = []
        self.__tensor_ops: list[TensorOp] = []
        self.__modules: list[NN | Layer | TensorOp] = []
        self.train_data: Dataset = train_data
        self.test_data: Dataset = test_data
        self.input_var: str = input_var
        self.return_vars: str = return_vars

    @property
    def sub_nns(self) -> list[NN]:
        """list[NN]: Get the sub NN models list of the main model."""
        return self.__sub_nns

    @sub_nns.setter
    def sub_nns(self, sub_nns: list[NN]):
        """list[NN]: Set the sub NN models list of the main model."""
        raise AttributeError("sub_nns attribute is read-only")

    def add_sub_nn(self, sub_nn: NN):
        """Self: Add a subnn to the NN model."""
        if isinstance(sub_nn, NN):
            self.__sub_nns.append(sub_nn)
            self.__modules.append(sub_nn)
        else:
            raise TypeError("'sub_nn' must be of type NN.")
        return self

    @property
    def layers(self) -> list[Layer]:
        """list[Layer]: Get the list of layers."""
        return self.__layers

    @layers.setter
    def layers(self, layers: list[Layer]):
        """list[Layer]: Set the list of layers."""
        raise AttributeError("layers attribute is read-only")

    def add_layer(self, layer: Layer) -> Self:
        """Self: Add a layer to the NN model."""
        if isinstance(layer, Layer):
            self.__layers.append(layer)
            self.__modules.append(layer)
        else:
            raise TypeError("'layer' must be of type Layer.")
        return self

    @property
    def tensor_ops(self) -> list[TensorOp]:
        """list[TensorOp]: Get the list of tensor Ops."""
        return self.__tensor_ops

    @tensor_ops.setter
    def tensor_ops(self, tensor_ops: list[TensorOp]):
        """list[TensorOp]: Set the list of tensor Ops ."""
        raise AttributeError("tensor_ops attribute is read-only")

    def add_tensor_op(self, tensor_op: TensorOp) -> Self:
        """Self: Add a tensor Op to the NN model."""
        if isinstance(tensor_op, TensorOp):
            self.__tensor_ops.append(tensor_op)
            self.__modules.append(tensor_op)
        else:
            raise TypeError("'tensor_op' must be of type TensorOp.")
        return self

    @property
    def modules(self) -> list[NN | Layer | TensorOp]:
        """list[NN | Layer | TensorOp]: Get the modules list
        of the main model."""
        return self.__modules

    @modules.setter
    def modules(self, modules: list[NN | Layer | TensorOp]):
        """list[NN | Layer | TensorOp]: Set the modules list
        of the main model."""
        self.__modules = modules

    @property
    def configuration(self) -> Configuration:
        """Configuration: Get the parameters related to the NN training
        and evaluation."""
        return self.__configuration

    @configuration.setter
    def configuration(self, configuration: Configuration):
        """Configuration: Set the parameters related to the NN training
        and evaluation."""
        self.__configuration = configuration

    @property
    def train_data(self) -> Dataset:
        """Dataset: Get the dataset used to train the NN model."""
        return self.__train_data

    @train_data.setter
    def train_data(self, train_data: Dataset):
        """Dataset: Set the dataset used to train the NN model."""
        self.__train_data = train_data

    @property
    def test_data(self) -> Dataset:
        """Dataset: Get the dataset used to evaluate the NN model."""
        return self.__test_data

    @test_data.setter
    def test_data(self, test_data: Dataset):
        """Dataset: Set the dataset used to evaluate the NN model."""
        self.__test_data = test_data

    @property
    def input_var(self) -> str:
        """str: Get the input variable name for this NN."""
        return self.__input_var

    @input_var.setter
    def input_var(self, input_var: str):
        """str: Set the input variable name for this NN."""
        if (
            input_var is not None
            and not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', input_var)
        ):
            raise ValueError(
                "input_var must be a valid identifier starting "
                "with a letter"
            )
        self.__input_var = input_var

    @property
    def return_vars(self) -> str:
        """str: Get the comma-separated string of variable names 
           returned by forward/call method."""
        return self.__return_vars

    @return_vars.setter
    def return_vars(self, return_vars: str):
        """str: Set the comma-separated string of variable names 
           returned by forward/call method."""
        if return_vars is not None:
            parts = [part.strip() for part in return_vars.split(",")]

            if not parts or any(part == "" for part in parts):
                raise ValueError(
                    "return_vars must be a non-empty comma-separated "
                    "list of variable names, with no empty entries"
                )

            for part in parts:
                if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', part):
                    raise ValueError(
                        f"Each variable in return_vars must be a valid "
                        f"identifier starting with a letter. "
                        f"Invalid entry: '{part}'"
                    )
        self.__return_vars = return_vars

    def add_configuration(self, configuration: Configuration) -> Self:
        """Self: Add the configuration to the NN model."""
        self.__configuration = configuration
        return self

    def add_train_data(self, train_data: Dataset) -> Self:
        """Self: Add the training dataset to the NN model."""
        self.__train_data = train_data
        return self

    def add_test_data(self, test_data: Dataset) -> Self:
        """Self: Add the test dataset to the NN model."""
        self.__test_data = test_data
        return self

    def __repr__(self):
        return (
            f'NN({self.name}, {self.configuration}, {self.modules}, '
            f'{self.train_data}, {self.test_data}, {self.return_vars}, '
            f'{self.input_var})'
            )

    def validate(self, raise_exception: bool = True,
                 _visited: set | None = None) -> dict:
        """
        Validate the neural network model.

        Checks performed:
            * Module names are unique within this NN scope.
            * Layer ``name_module_input`` references resolve to
              a module defined in the same NN.
            * TensorOp ``layers_of_tensors`` string entries resolve to
              a module defined in the same NN.
            * Sub-NNs are acyclic (no NN directly or transitively
              contains itself).
            * Each sub-NN is itself valid (recursive validation).
            * Warnings are emitted for empty NNs and for missing
              configuration on a top-level NN that has training data.

        Args:
            raise_exception: If True, raises ``ValueError`` when errors
                are found. Warnings never raise.
            _visited: Internal — tracks NN instances already validated
                to stop infinite recursion on cyclic sub-NN graphs.

        Returns:
            dict: ``{"success": bool, "errors": list[str], 
                     "warnings": list[str]}``
        """
        errors: list[str] = []
        warnings: list[str] = []

        if _visited is None:
            _visited = set()
        if id(self) in _visited:
            return {"success": True, "errors": errors, "warnings": warnings}
        _visited.add(id(self))

        self._validate_module_uniqueness(errors)
        self._validate_module_input_references(errors)
        self._validate_tensor_op_references(errors)
        self._validate_first_module_entry_point(errors)
        self._validate_numerical_bounds(errors)
        self._validate_module_names(errors, warnings)
        self._validate_input_output_var_chain(errors)
        cycle_detected = self._validate_sub_nn_acyclic(errors)
        if not cycle_detected:
            self._validate_sub_nns_recursive(errors, warnings, _visited)
        self._collect_nn_warnings(warnings)
        self._validate_dataset_consistency(warnings)

        result = {"success": len(errors) == 0, "errors": errors,
                  "warnings": warnings}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _module_names(self) -> set:
        """Names of every module declared in this NN (layers, 
        tensor_ops, sub_nns)."""
        return {m.name for m in self.modules}

    def _validate_module_uniqueness(self, errors: list):
        seen: dict = {}
        for module in self.modules:
            name = module.name
            if name in seen:
                errors.append(
                    f"NN '{self.name}' has duplicate module name '{name}' "
                    f"(declared twice)."
                )
            seen[name] = module

    def _validate_module_input_references(self, errors: list):
        names = {"INPUT"} | self._module_names()
        for layer in self.layers:
            ref = layer.name_module_input
            if ref and ref not in names:
                errors.append(
                    f"NN '{self.name}': layer '{layer.name}' references "
                    f"input module '{ref}' which is not defined in this NN."
                )

    def _validate_tensor_op_references(self, errors: list):
        names = {"INPUT"} | self._module_names()
        for tensor_op in self.tensor_ops:
            entries = tensor_op.layers_of_tensors or []
            for entry in entries:
                if isinstance(entry, str) and entry not in names:
                    errors.append(
                        f"NN '{self.name}': tensorOp '{tensor_op.name}'"
                        f" references input module '{entry}' which is"
                        f" not defined in this NN."
                    )

    def _validate_first_module_entry_point(self, errors: list):
        if not self.modules:
            return
        first = self.modules[0]
        if not isinstance(first, Layer):
            return
        if first.name_module_input:
            errors.append(
                f"NN '{self.name}': first module '{first.name}' must not "
                f"declare a 'name_module_input' (it is the entry point)."
            )

    def _validate_input_output_var_chain(self, errors: list):
        """Validate input_var and output_var consistency
        across the NN."""
        if not self.modules:
            return

        # Check first module's input_var against NN's input_var
        first = self.modules[0]
        if self.input_var is not None:
            first_input_var = getattr(first, 'input_var', None)
            if first_input_var is None:
                # Set it to match NN's input_var
                if hasattr(first, 'input_var'):
                    first.input_var = self.input_var
            elif first_input_var != self.input_var:
                errors.append(
                    f"NN '{self.name}': first module '{first.name}' has "
                    f"input_var '{first_input_var}' which differs from NN's "
                    f"input_var '{self.input_var}'. They must be the same."
                )

        # Check last module's output_var against NN's return_vars
        last = self.modules[-1]
        last_output_var = getattr(last, 'output_var', None)
        if self.return_vars is not None:
            last_output_var = getattr(last, 'output_var', None)
            if last_output_var is None:
                last_output_var = getattr(last, "output_vars", None)
                if last_output_var:
                    last_output_var = ", ".join(last_output_var)
            if last_output_var is None:
                # Set it to match NN's return_vars
                if (isinstance(last, TensorOp)
                    and last.tns_type == "split"):
                    last.output_vars = [
                        x.strip() for x in self.return_vars.split(",")
                    ]
                elif hasattr(last, 'output_var'):
                    last.output_var = self.return_vars
            elif last_output_var != self.return_vars:
                errors.append(
                    f"NN '{self.name}': last module '{last.name}' has "
                    f"output_var '{last_output_var}' which differs from NN's "
                    f"return_vars '{self.return_vars}'. "
                    "They must be the same."
                )

    def _validate_sub_nn_acyclic(self, errors: list) -> bool:
        """Detect cycles in the sub-NN graph rooted at this NN. 
           Returns True if a cycle was found."""
        def visit(nn: NN, stack: set) -> bool:
            if id(nn) in stack:
                return True
            stack.add(id(nn))
            for child in nn.sub_nns:
                if visit(child, stack):
                    return True
            stack.remove(id(nn))
            return False

        if visit(self, set()):
            errors.append(
                f"NN '{self.name}': sub-NN graph contains a cycle (an NN "
                f"directly or transitively contains itself)."
            )
            return True
        return False

    def _validate_sub_nns_recursive(self, errors: list, warnings: list,
                                    _visited: set):
        for sub in self.sub_nns:
            sub_result = sub.validate(raise_exception=False,
                                      _visited=_visited)
            errors.extend(sub_result["errors"])
            warnings.extend(sub_result["warnings"])

    def _collect_nn_warnings(self, warnings: list):
        if not self.modules:
            warnings.append(f"NN '{self.name}' has no modules.")
        if self.train_data is not None and self.configuration is None:
            warnings.append(
                f"NN '{self.name}' has training data but no configuration."
            )

    def _validate_dataset_consistency(self, warnings: list):
        """Surface mismatches between training and test datasets that 
        usually indicate user error."""
        train, test = self.train_data, self.test_data
        if test is not None and train is None:
            warnings.append(
                f"NN '{self.name}' has a test dataset "
                "but no training dataset."
            )
        if train is None or test is None:
            return
        train_fmt = train.input_format
        test_fmt = test.input_format
        if train_fmt and test_fmt and train_fmt != test_fmt:
            warnings.append(
                f"NN '{self.name}': train input_format '{train_fmt}' differs "
                f"from test input_format '{test_fmt}'."
            )
        if (
            train.image is not None
            and test.image is not None
            and train.image.shape != test.image.shape
        ):
            warnings.append(
                f"NN '{self.name}': train image shape {train.image.shape}"
                f" differs from test image shape {test.image.shape}."
            )

    def _validate_module_names(self, errors: list, warnings: list):
        """Reject names that aren't valid Python identifiers; warn
        (matching ``NamedElement.name``'s warn-not-error stance) on
        Python keywords."""
        def _check(name, label):
            if not name.isidentifier():
                errors.append(
                    f"{label} name '{name}' is not a valid Python identifier."
                )
            elif keyword.iskeyword(name):
                warnings.append(
                    f"{label} name '{name}' is a Python reserved keyword "
                    "and may cause issues in generated code."
                )

        _check(self.name, "NN")
        for module in self.modules:
            _check(module.name, type(module).__name__)

    def _validate_numerical_bounds(self, errors: list):
        """Reject non-positive sizes/rates that would crash the trainer 
        at runtime."""
        cfg = self.configuration
        if cfg is not None:
            if cfg.batch_size <= 0:
                errors.append(
                    f"NN '{self.name}': configuration batch_size "
                    f"must be > 0, got {cfg.batch_size}."
                )
            if cfg.epochs <= 0:
                errors.append(
                    f"NN '{self.name}': configuration epochs must be > 0, "
                    f"got {cfg.epochs}."
                )
            if cfg.learning_rate <= 0:
                errors.append(
                    f"NN '{self.name}': configuration learning_rate "
                    f"must be > 0, got {cfg.learning_rate}."
                )
            if cfg.weight_decay < 0:
                errors.append(
                    f"NN '{self.name}': configuration weight_decay "
                    f"must be >= 0, got {cfg.weight_decay}."
                )

        for layer in self.layers:
            cls_name = type(layer).__name__
            label = f"NN '{self.name}': {cls_name} '{layer.name}'"

            if isinstance(layer, DropoutLayer) and (not 0 <= layer.rate < 1):
                    errors.append(
                        f"{label} rate must be in [0, 1), got {layer.rate}."
                    )
            if isinstance(layer, RNN):
                if layer.hidden_size <= 0:
                    errors.append(
                        f"{label} hidden_size must be > 0, "
                        f"got {layer.hidden_size}."
                    )
                if not 0 <= layer.dropout < 1:
                    errors.append(
                        f"{label} dropout must be in [0, 1), "
                        f"got {layer.dropout}."
                    )
            if isinstance(layer, LinearLayer):
                if layer.out_features <= 0:
                    errors.append(
                        f"{label} out_features must be > 0, "
                        f"got {layer.out_features}."
                    )
                if layer.in_features is not None and layer.in_features <= 0:
                    errors.append(
                        f"{label} in_features must be > 0, "
                        f"got {layer.in_features}."
                    )
            if isinstance(layer, ConvolutionalLayer):
                if layer.out_channels <= 0:
                    errors.append(
                        f"{label} out_channels must be > 0, "
                        f"got {layer.out_channels}."
                    )
                if layer.in_channels is not None and layer.in_channels <= 0:
                    errors.append(
                        f"{label} in_channels must be > 0, "
                        f"got {layer.in_channels}."
                    )
                if any(d <= 0 for d in (layer.kernel_dim or [])):
                    errors.append(
                        f"{label} kernel_dim entries must all be > 0, "
                        f"got {layer.kernel_dim}."
                    )
                if (
                    layer.stride_dim is not None
                    and any(d <= 0 for d in layer.stride_dim)
                ):
                    errors.append(
                        f"{label} stride_dim entries must all be > 0, "
                        f"got {layer.stride_dim}."
                    )
            if isinstance(layer, PoolingLayer):
                if (
                    layer.kernel_dim is not None
                    and any(d <= 0 for d in layer.kernel_dim)
                ):
                    errors.append(
                        f"{label} kernel_dim entries must all be > 0, "
                        f"got {layer.kernel_dim}."
                    )
                if (
                    layer.stride_dim is not None
                    and any(d <= 0 for d in layer.stride_dim)
                ):
                    errors.append(
                        f"{label} stride_dim entries must all be > 0, "
                        f"got {layer.stride_dim}."
                    )
            if isinstance(layer, BatchNormLayer) and layer.num_features <= 0:
                errors.append(
                    f"{label} num_features must be > 0, got "
                    f"{layer.num_features}."
                )
            if (
                isinstance(layer, LayerNormLayer)
                and layer.normalized_shape is not None
                and any(d <= 0 for d in layer.normalized_shape)
            ):
                errors.append(
                    f"{label} normalized_shape entries must all be > 0, "
                    f"got {layer.normalized_shape}."
                )
            if isinstance(layer, EmbeddingLayer):
                if layer.num_embeddings <= 0:
                    errors.append(
                        f"{label} num_embeddings must be > 0, got "
                        f"{layer.num_embeddings}."
                    )
                if layer.embedding_dim <= 0:
                    errors.append(
                        f"{label} embedding_dim must be > 0, got "
                        f"{layer.embedding_dim}."
                    )
        for ds_label, ds in (
            ("train_data", self.train_data),
            ("test_data", self.test_data)
        ):
            if (
                ds is not None
                and ds.image is not None
                and any(d <= 0 for d in ds.image.shape)
            ):
                errors.append(
                    f"NN '{self.name}': {ds_label} image shape entries "
                    f"must all be > 0, got {ds.image.shape}."
                )
