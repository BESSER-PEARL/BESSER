from __future__ import annotations
from besser.BUML.metamodel.structural import BehavioralImplementation
from typing import List, Self, Union

                                                                              
class TensorOp:
    """
    This class represents a tensor operation. It encapsulates attributes such 
    as the name and the type of the tensor operation.
    
    Args:
        name (str): The name of the tensor operation.
        type (str): The type of the tensor operation.
        concatenate_dim (int): The dimension along which the tensors will be 
            concatenated with the cat operation.
        layers_of_tensors (List[Union[str, float]]): The list that defines the
            inputs of the tensor op. Elements of the list can be either names
            of layers from which the tensors originate or scalar values.
        reshape_dim (List[int]): A list specifying the new shape of the tensor
            after the reshape operation.
        transpose_dim (List[int]): A list specifying the transpose dimensions. 
            Only relevant with the transpose operation.
        permute_dim (List[int]): A list containing the desired ordering of 
            dimensions. Only relevant with the permute operation.
        after_activ_func(bool): Whether to perform the tensor operation after 
            applying the activation function.
        input_reused (bool): Whether the input to this tensor op is reused as 
            input to another layer (or tensor op).

    Attributes:
        name (str): The name of the tensor operation.
        type (str): The type of the tensor operation.
        concatenate_dim (int): The dimension along which the tensors will be 
            concatenated with the cat operation.
        layers_of_tensors (List[Union[str, float]]): The list that defines the
            inputs of the tensor op. Elements of the list can be either names
            of layers from which the tensors originate or scalar values.
        reshape_dim (List[int]): A list specifying the new shape of the tensor
            after the reshape operation.
        transpose_dim (List[int]): A list specifying the transpose dimensions.
            Only relevant with the transpose operation.
        permute_dim (List[int]): A list containing the desired ordering of 
            dimensions. Only relevant with the permute operation.
        after_activ_func(bool): Whether to perform the tensor operation after
            applying the activation function.
        input_reused (bool): Whether the input to this tensor op is reused as 
            input to another layer (or tensor op).
    """
    def __init__(self, name: str, type: str, concatenate_dim: int = None, 
                 layers_of_tensors: List[Union[str, float]] = None, 
                 reshape_dim: List[int] = None, 
                 transpose_dim: List[int] = None, 
                 permute_dim: List[int] = None, 
                 after_activ_func: bool = True, input_reused: bool = False):
        self.name: str = name
        self.concatenate_dim: int = concatenate_dim
        self.layers_of_tensors: List[Union[str, float]] = layers_of_tensors
        self.reshape_dim: List[int] = reshape_dim
        self.transpose_dim: List[int] = transpose_dim
        self.permute_dim: List[int] = permute_dim
        self.after_activ_func: bool = after_activ_func
        self.input_reused: bool = input_reused
        self.type: str = type

    @property
    def name(self) -> str:
        """str: Get the name of the tensorOp."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the tensorOp."""
        self.__name = name

    @property
    def type(self) -> str:
        """str: Get the type of the tensorOp."""
        return self.__type

    @type.setter
    def type(self, type: str):
        """
        str: Set the type of the tensorOp.

        Raises:
            ValueError: If the type is not one of the allowed options: 
            'reshape', 'concatenate', 'multiply', 'matmultiply', 'permute' 
            and transpose.
        """
        if type not in [
            'reshape', 'concatenate', 'multiply', 
            'matmultiply', 'permute', 'transpose'
        ]:
            raise ValueError("Invalid value of tensorOp type")
        elif type == 'reshape' and self.reshape_dim == None:
            raise ValueError("reshape_dim parameter cannot be None when type \
                             is 'reshape'")
        elif type == 'concatenate':
            if self.concatenate_dim == None:
                raise ValueError("concatenate_dim parameter cannot be None \
                                  when type is 'concatenate'")
            elif self.layers_of_tensors == None:
                raise ValueError("layers_of_tensors parameter cannot be None \
                                 when type is 'concatenate'")
        elif type == 'multiply' and self.layers_of_tensors == None:
            raise ValueError("layers_of_tensors parameter cannot be None \
                             when type is 'multiply'")
        elif type == 'matmultiply' and self.layers_of_tensors == None:
            raise ValueError("layers_of_tensors parameter cannot be None \
                              when type is 'matmultiply'")
        elif type == 'permute' and self.permute_dim == None:
            raise ValueError("permute_dim parameter cannot be None when \
                             type is 'permute'")
        elif type == 'transpose' and self.transpose_dim == None:
            raise ValueError("transpose_dim parameter cannot be None when \
                             type is 'transpose'")

        self.__type = type

    @property
    def concatenate_dim(self) -> int:
        """
        int: Get the dimension along which the tensors will be concatenated 
            with the cat operation.
        """
        return self.__concatenate_dim

    @concatenate_dim.setter
    def concatenate_dim(self, concatenate_dim: int):
        """
        int: Set the dimension along which the tensors will be concatenated
            with the cat operation.
        """
        self.__concatenate_dim = concatenate_dim

    @property
    def layers_of_tensors(self) -> List[Union[str, float]]:
        """
        List[Union[str, float]]: Get the list that defines the inputs of the 
            tensor op. Elements of the list can be either names of layers 
            from which the tensors originate or scalar values.
        """
        return self.__layers_of_tensors

    @layers_of_tensors.setter
    def layers_of_tensors(self, layers_of_tensors: List[Union[str, float]]):
        """
        List[Union[str, float]]: Set the list of layers names from which the 
            tensors, on which tensor ops are performed, originate.
        """
        self.__layers_of_tensors = layers_of_tensors

    @property
    def reshape_dim(self) -> List[int]:
        """
        List[int]: Get the list specifying the new shape of the tensor after 
            reshaping with the view operation.
        """
        return self.__reshape_dim

    @reshape_dim.setter
    def reshape_dim(self, reshape_dim: List[int]):
        """
        List[int]: Set the list specifying the new shape of the tensor after 
            reshaping with the view operation.
        """
        self.__reshape_dim = reshape_dim

    @property
    def transpose_dim(self) -> List[int]:
        """List[int]: Get the list specifying the transpose dimensions."""
        return self.__transpose_dim

    @transpose_dim.setter
    def transpose_dim(self, transpose_dim: List[int]):
        """List[int]: Set the list specifying the transpose dimensions."""
        self.__transpose_dim = transpose_dim

    @property
    def permute_dim(self) -> List[int]:
        """
        List[int]: Get the list containing the desired ordering of dimensions
            for permute operation.
        """
        return self.__permute_dim

    @permute_dim.setter
    def permute_dim(self, permute_dim: List[int]):
        """
        List[int]: Set the list containing the desired ordering of dimensions 
            for permute operation.
        """
        self.__permute_dim = permute_dim

    @property
    def after_activ_func(self) -> bool:
        """
        bool: Get whether to perform the tensor operation after applying the 
            activation function.
        """
        return self.__after_activ_func

    @after_activ_func.setter
    def after_activ_func(self, after_activ_func: bool):
        """
        bool: Set whether to perform the tensor operation after applying 
            the activation function.
        """
        self.__after_activ_func = after_activ_func

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
        self.__input_reused = input_reused

    def __repr__(self):
        return (
            f'TensorOp({self.name}, {self.type}, {self.concatenate_dim}, '
            f'{self.layers_of_tensors}, {self.reshape_dim}, '
            f'{self.transpose_dim}, {self.permute_dim}, '
            f'{self.after_activ_func}, {self.input_reused})'
        )

class Layer:
    """
    This class represents a layer of the neural network. It encapsulates 
    attributes such as the name of the layer and the activation function.
    
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.

    Attributes:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
    """
    def __init__(self, name: str, actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        self.name: str = name
        self.actv_func: str = actv_func
        self.name_layer_input: str = name_layer_input
        self.input_reused: bool = input_reused
    
    @property
    def name(self) -> str:
        """str: Get the name of the layer."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the layer."""
        self.__name = name

    @property
    def actv_func(self) -> str:
        """str: Get the actv_func."""
        return self.__actv_func

    @actv_func.setter
    def actv_func(self, actv_func: str):
        """
        str: Set the actv_func.

        Raises:
            ValueError: If the actv_func is not one of the allowed 
            options: 'relu', 'leaky_rely', 'sigmoid', 'softmax' and 'tanh'
        """
        if actv_func is not None and actv_func not in [
            'relu', 'leaky_relu', 'sigmoid', 'softmax', 'tanh'
        ]:
            raise ValueError("Invalid value of actv_func")
        self.__actv_func = actv_func

    @property
    def name_layer_input(self) -> str:
        """str: Get the name of the layer from which the inputs originate."""
        return self.__name_layer_input

    @name_layer_input.setter
    def name_layer_input(self, name_layer_input: str):
        """str: Set the name of the layer from which the inputs originate."""
        self.__name_layer_input = name_layer_input

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
        self.__input_reused = input_reused
    
    def __repr__(self):
        return (
            f'Layer({self.name}, {self.actv_func}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )

class CNN(Layer):
    """
    Represents a layer that is generally used in convolutional neural 
    networks.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        kernel_dim (List[int]): A list containing the dimensions of 
            the convolving or pooling kernel (i.e., [depth, height, width]).
        stride_dim (List[int]): A list containing the dimensions of 
            the stride of the convolution or pooling (i.e., [depth, height, 
            width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of the 
            activation function.
        kernel_dim (List[int]): A list containing the dimensions of 
            the convolving or pooling kernel (i.e., [depth, height, width]).
        stride_dim (List[int]): A list containing the dimensions of 
            the stride of the convolution or pooling (i.e., [depth, height, 
            width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """

    def __init__(self, name: str, kernel_dim: List[int], 
                 stride_dim: List[int], padding_amount: int = 0, 
                 padding_type: str = "valid", actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)
        self.kernel_dim: List[int] = kernel_dim
        self.stride_dim: List[int] = stride_dim
        self.padding_amount: int = padding_amount
        self.padding_type: str = padding_type

    @property
    def kernel_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: List[int]):
        """List[int]: Set the list of dimensions of the kernel."""
        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: List[int]):
        """List[int]: Set the list of dimensions of the stride."""
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


    def __repr__(self):
        return (
            f'CNN({self.name}, {self.actv_func}, {self.kernel_dim}, '
            f'{self.stride_dim}, {self.padding_amount}, {self.padding_type}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )

class ConvolutionalLayer(CNN):
    """
    Represents a convolutional layer.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by 
            the convolution.
        kernel_dim (List[int]): A list containing the dimensions of 
            the convolving or pooling kernel (i.e., [depth, height, width]).
        stride_dim (List[int]): A list containing the dimensions of 
            the stride of the convolution or pooling (i.e., [depth, height, 
            width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by 
            the convolution.
        kernel_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the convolving or pooling kernel (i.e., [depth, 
            height, width]).
        stride_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the stride of the convolution or pooling 
            (i.e., [depth, height, width]).
        padding_amount (int): Inherited from CNN. The amount of padding added 
            to the input.
        padding_type (str): Inherited from CNN. The type of padding applied 
            to the input. 
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
        
    def __init__(self, name: str, in_channels: int, out_channels: int, 
                 kernel_dim: List[int], stride_dim: List[int], 
                 padding_amount: int = 0, padding_type: str = "valid",
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, kernel_dim, stride_dim, padding_amount, 
                         padding_type, actv_func, name_layer_input, 
                         input_reused)
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

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
        """int: Get the number of channels produced by the convolution."""
        return self.__out_channels

    @out_channels.setter
    def out_channels(self, out_channels: int):
        """int: Set the number of channels produced by the convolution."""
        self.__out_channels = out_channels


    def __repr__(self):
        return (
            f'ConvolutionaLayer({self.name}, {self.actv_func}, '
            f'{self.in_channels}, {self.out_channels}, {self.kernel_dim}, '
            f'{self.stride_dim}, {self.padding_amount}, {self.padding_type}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )
   
class Conv1D(ConvolutionalLayer):
    """
    Represents a type of convolutional layer that applies a 1D convolution.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by 
            the convolution.
        kernel_dim (List[int]): A list containing the dimensions of 
            the convolving or pooling kernel (i.e., [depth, height, width]).
        stride_dim (List[int]): A list containing the dimensions of the stride
            of the convolution or pooling (i.e., [depth, height, width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        in_channels (int): Inherited from ConvolutionalLayer. It represents 
            the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It represents 
            the number of channels produced by the convolution.
        kernel_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the convolving or pooling kernel 
            (i.e., [depth, height, width]).
        stride_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the stride of the convolution or pooling 
            (i.e., [depth, height, width]).
        padding_amount (int): Inherited from CNN. It represents the amount of
            padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type of 
            padding applied to the input. 
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, in_channels: int, out_channels: int, 
                 kernel_dim: List[int], stride_dim: List[int] = [1], 
                 padding_amount: int = 0, padding_type: str = "valid",
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, in_channels, out_channels, kernel_dim, 
                         stride_dim, padding_amount, padding_type, actv_func,
                         name_layer_input, input_reused)
    
    @property
    def kernel_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: List[int]):
        """
        List[int]: Set the list of dimensions of the kernel.
        An error is raised if the list contains more than 1 element 
        (dimension).
        """
        if len(kernel_dim) != 1:
            raise ValueError("kernel_dim list must have exactly 1 element \
                             (dimension).")

        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: List[int]):
        """
        List[int]: Set the list of dimensions of the stride.
        An error is raised if the list contains more than 1 element 
        (dimension).
        """
        if len(stride_dim) != 1:
            raise ValueError("stride_dim list must have exactly 1 element \
                             (dimension).")
        self.__stride_dim = stride_dim

    def __repr__(self):
        return (
            f'Conv1D({self.name}, {self.actv_func}, {self.in_channels}, '
            f'{self.out_channels}, {self.kernel_dim}, {self.stride_dim}, ' 
            f'{self.padding_amount}, {self.padding_type}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )
        
class Conv2D(ConvolutionalLayer):
    """
    Represents a type of convolutional layer that applies a 2D convolution.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by 
            the convolution.
        kernel_dim (List[int]): A list containing the dimensions of 
            the convolving or pooling kernel (i.e., [depth, height, width]).
        stride_dim (List[int]): A list containing the dimensions of 
            the stride of the convolution or pooling 
            (i.e., [depth, height, width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused 
            as input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        in_channels (int): Inherited from ConvolutionalLayer. It represents 
            the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It represents 
            the number of channels produced by the convolution.
        kernel_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the convolving or pooling kernel 
            (i.e., [depth, height, width]).
        stride_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the stride of the convolution or pooling 
            (i.e., [depth, height, width]).
        padding_amount (int): Inherited from CNN. It represents the amount of 
            padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type of 
            padding applied to the input. 
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, in_channels: int, out_channels: int, 
                 kernel_dim: List[int], stride_dim: List[int] = [1, 1],  
                 padding_amount: int = 0, padding_type: str = "valid", 
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, in_channels, out_channels, kernel_dim, 
                         stride_dim, padding_amount, padding_type,
                         actv_func, name_layer_input, input_reused) 
    
    @property
    def kernel_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: List[int]):
        """
        List[int]: Set the list of dimensions of the kernel.
        An error is raised if the list contains more than 2 elements 
        (dimensions).
        """
        if len(kernel_dim) != 2:
            raise ValueError("kernel_dim list must have exactly 2 elements \
                             (dimensions).")

        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: List[int]):
        """
        List[int]: Set the list of dimensions of the stride.
        An error is raised if the list contains more than 2 elements 
        (dimensions).
        """
        if len(stride_dim) != 2:
            raise ValueError("stride_dim list must have exactly 2 elements \
                             (dimensions).")

        self.__stride_dim = stride_dim


    def __repr__(self):
        return (
            f'Conv2D({self.name}, {self.actv_func}, {self.in_channels}, '
            f'{self.out_channels}, {self.kernel_dim}, {self.stride_dim}, '
            f'{self.padding_amount}, {self.padding_type}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )
            
class Conv3D(ConvolutionalLayer):
    """
    Represents a type of convolutional layer that applies a 3D convolution.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by 
            the convolution.
        kernel_dim (List[int]): A list containing the dimensions of 
            the convolving or pooling kernel (i.e., [depth, height, width]).
        stride_dim (List[int]): A list containing the dimensions of 
            the stride of the convolution or pooling 
            (i.e., [depth, height, width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        in_channels (int): Inherited from ConvolutionalLayer. It represents 
            the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It represents 
            the number of channels produced by the convolution.
        kernel_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the convolving or pooling kernel 
            (i.e., [depth, height, width]).
        stride_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the stride of the convolution or pooling 
            (i.e., [depth, height, width]).
        padding_amount (int): Inherited from CNN. It represents the amount of 
            padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type of 
            padding applied to the input. 
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, in_channels: int, out_channels: int, 
                 kernel_dim: List[int], stride_dim: List[int] = [1, 1, 1], 
                 padding_amount: int = 0, padding_type: str = "valid", 
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, in_channels, out_channels, kernel_dim, 
                         stride_dim, padding_amount, padding_type, actv_func, 
                         name_layer_input, input_reused)
    
    @property
    def kernel_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: List[int]):
        """
        List[int]: Set the list of dimensions of the kernel.
        An error is raised if the list does not contains exactly 3 elements 
        (dimensions).
        """
        if len(kernel_dim) != 3:
            raise ValueError("kernel_dim list must have exactly 3 element \
                             (dimensions).")

        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: List[int]):
        """
        List[int]: Set the list of dimensions of the stride.
        An error is raised if the list does not contains exactly 3 elements 
        (dimensions).
        """
        if len(stride_dim) != 3:
            raise ValueError("stride_dim list must have exactly 3 elements \
                             (dimensions).")
        self.__stride_dim = stride_dim

    def __repr__(self):
        return (
            f'Conv3D({self.name}, {self.actv_func}, {self.in_channels}, '
            f'{self.out_channels}, {self.kernel_dim}, {self.stride_dim}, '
            f'{self.padding_amount}, {self.padding_type}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )


class PoolingLayer(CNN):
    """
    Represents a type of layer that performs a pooling operation.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        type (str): The type of pooling applied.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the pooling 
            operation.
        kernel_dim (List[int]): A list containing the dimensions of 
            the convolving or pooling kernel (i.e., [depth, height, width]).
        stride_dim (List[int]): A list containing the dimensions of 
            the stride of the convolution or pooling 
            (i.e., [depth, height, width]).
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        pooling_type (str): The type of pooling. Either average or max.
        dimension (str): The dimensionality of the pooling. Either 1D, 2D 
            or 3D.
        output_dim (List[int]): The output dimensions of the adaptive 
            pooling operation. Only relevant for adaptive pooling.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        type (str): It represents the type of pooling applied.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the pooling 
            operation.
        kernel_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the convolving or pooling kernel 
            (i.e., [depth, height, width]).
        stride_dim (List[int]): Inherited from CNN. A list containing 
            the dimensions of the stride of the convolution or pooling 
            (i.e., [depth, height, width]).
        padding_amount (int): Inherited from CNN. It represents the amount 
            of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type of 
            padding applied to the input. 
        pooling_type (str): The type of pooling. Either average or max.
        dimension (str): The dimensionality of the pooling. Either 1D, 2D 
            or 3D.
        output_dim (List[int]): The output dimensions of the adaptive 
            pooling operation. Only relevant for adaptive pooling.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, pooling_type: str, dimension: str, 
                 kernel_dim: List[int] = None, stride_dim: List[int] = None, 
                 padding_amount: int = 0, padding_type: str = "valid", 
                 output_dim: List[int] = [], actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        self.pooling_type: str = pooling_type
        self.dimension: str = dimension
        self.output_dim: List[int] = output_dim
        super().__init__(name, kernel_dim, stride_dim, padding_amount, 
                         padding_type, actv_func, name_layer_input, 
                         input_reused)
        
    @property
    def kernel_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the kernel."""
        return self.__kernel_dim

    @kernel_dim.setter
    def kernel_dim(self, kernel_dim: List[int]):
        """
        List[int]: Set the list of dimensions of the kernel.
        An error is raised if the length of the list does not match 
        the dimensionality of the pooling operation.
        """
        if not self.pooling_type.startswith("adaptive"):
            if self.dimension == "1D" and len(kernel_dim) != 1:
                raise ValueError("kernel_dim list must have exactly \
                                 1 element (dimension).")
            elif self.dimension == "2D" and len(kernel_dim) != 2:
                raise ValueError("kernel_dim list must have exactly \
                                 2 elements (dimensions).")
            elif self.dimension == "3D" and len(kernel_dim) != 3:
                raise ValueError("kernel_dim list must have exactly \
                                 3 elements (dimensions).")
        self.__kernel_dim = kernel_dim

    @property
    def stride_dim(self) -> List[int]:
        """List[int]: Get the list of dimensions of the stride."""
        return self.__stride_dim

    @stride_dim.setter
    def stride_dim(self, stride_dim: List[int]):
        """
        List[int]: Set the list of dimensions of the stride.
        An error is raised if the length of the list does not match 
        the dimensionality of the pooling operation.
        """
        if (
            not self.pooling_type.startswith("adaptive") 
            and stride_dim is not None
        ):
            if self.dimension == "1D" and len(stride_dim) != 1:
                raise ValueError("kernel_dim list must have exactly \
                                 1 element (dimension).")
            elif self.dimension == "2D" and len(stride_dim) != 2:
                raise ValueError("kernel_dim list must have exactly \
                                 2 elements (dimensions).")
            elif self.dimension == "3D" and len(stride_dim) != 3:
                raise ValueError("kernel_dim list must have exactly \
                                 3 elements (dimensions).")
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
            ValueError: If the pooling type provided is none of 
            these: 'average', 'adaptive_average', 'max' or 'adaptive_max'.
        """

        if pooling_type not in [
            'average', 'adaptive_average', 'max', 'adaptive_max'
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
    def output_dim(self) -> List[int]:
        """List[int]: Get the output dimensions of the adaptive pooling."""
        return self.__output_dim

    @output_dim.setter
    def output_dim(self, output_dim: List[int]):
        """List[int]: Set the output dimensions of the adaptive pooling."""
        self.__output_dim = output_dim

    def __repr__(self):
        return (
            f'PoolingLayer({self.name}, {self.actv_func}, '
            f'{self.pooling_type}, {self.dimension}, {self.kernel_dim}, '
            f'{self.stride_dim}, {self.padding_amount}, {self.padding_type}, '
            f'{self.output_dim}, {self.name_layer_input}, '
            f'{self.input_reused})'
        )
    
class LayerModifier(Layer):
    """
    Represents a type of layer that applies transformations or adjustments
        to other layers, enhancing their behavior or performance.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to 
            this layer is reused as input to another layer.
    """
    
    def __init__(self, name: str, actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)

    def __repr__(self):
        return (
            f'LayerModifier({self.name}, {self.actv_func}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )
        
class NormalisationLayer(LayerModifier):
    """
    Represents a type of layer that applies normalisation techniques.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)

    def __repr__(self):
        return (
            f'NormalisationLayer({self.name}, {self.actv_func}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )
        
class BatchNormLayer(NormalisationLayer):
    """
    Represents a type of layer that normalises inputs within mini-batches to 
    maintain consistent mean and variance, enhancing training speed and 
    stability.
    
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        num_features (int): The number of channels or features in each input 
            sample.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the input 
            data to be normalised using batch normalisation.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        num_features (int): The number of channels or features in each input 
            sample.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the input 
            data to be normalised using batch normalisation.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, num_features: int, dimension: str,
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)
        self.num_features: int = num_features
        self.dimension: str = dimension

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
        """str: Get the dimensionality of the input data to be normalised."""
        return self.__dimension

    @dimension.setter
    def dimension(self, dimension: str):
        """
        str: Set the dimensionality of the input data to be normalised.
        
        Raises:
            ValueError: If the dimensionality of the input data is none of 
            these: '1D', '2D', or '3D'.
        """

        if dimension not in ['1D', '2D', '3D']:
            raise ValueError ("Invalid data dimensionality")  
        self.__dimension = dimension

    def __repr__(self):
        return (
            f'BatchNormLayer({self.name}, {self.actv_func}, '
            f'{self.num_features}, {self.dimension}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )

class LayerNormLayer(NormalisationLayer):
    """
    Represents a type of layer that normalises the inputs across 
    the features of a single data sample, rather than across 
    the batch, to stabilize and accelerate training by reducing internal 
    covariate shift.
    
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        normalised_shape (List[int]): A list refering to the dimensions or 
            axis indices over which layer normalization is applied, 
            specifying which parts of the tensor are normalized.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        normalised_shape (List[int]): A list refering to the dimensions or 
            axis indices over which layer normalization is applied, 
            specifying which parts of the tensor are normalized.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, normalised_shape: List[int], 
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)
        self.normalised_shape: List[int] = normalised_shape

    @property
    def normalised_shape(self) -> List[int]:
        """
        List[int]: Get the list containing the dimensions or axis indices over
            which layer normalization is applied. 
        """
        return self.__normalised_shape

    @normalised_shape.setter
    def normalised_shape(self, normalised_shape: List[int]):
        """
        List[int]: Set the list containing the dimensions or axis indices over
            which layer normalization is applied. 
        """
        self.__normalised_shape = normalised_shape

    def __repr__(self):
        return (
            f'LayerNormLayer({self.name}, {self.actv_func}, '
            f'{self.normalised_shape}, {self.name_layer_input}, '
            f'{self.input_reused})'
        )

class DropoutLayer(LayerModifier):
    """
    Represents a type of layer that randomly sets a fraction of input 
    units to zero during training to prevent overfitting and improve 
    generalization.
    
    Args:
        name (str): The name of the layer.
        rate (float): It represents a float between 0 and 1. Fraction of 
            the input units to drop. 
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        rate (float): It represents a float between 0 and 1. Fraction of 
            the input units to drop. 
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, rate: float, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, None, name_layer_input, input_reused)
        self.rate: float = rate

    @property
    def rate(self) -> float:
        """float: Get the fraction of the input units to drop."""
        return self.__rate

    @rate.setter
    def rate(self, rate: float):
        """float: Set the fraction of the input units to drop."""
        self.__rate = rate

    def __repr__(self):
        return (
            f'DropoutLayer({self.name}, {self.rate}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )

class RNN(Layer):
    """
    Represents a type of layer used in recurrent neural networks (RNN) for 
    processing sequential data by using memory from previous steps to inform 
    current outputs.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input 
            features.
        hidden_size (int): It represents the number of units in the hidden 
            state, which captures the network's internal representation of 
            the input sequence.
        bidirectional (bool): Whether the layer is bidirectional or not.
        dropout (float): If non-zero, it introduces a Dropout layer on 
            the outputs of the current sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are 
            provided as (batch, seq, feature) instead of (seq, batch, feature).
            Only relevant to PyTorch.
        return_hidden (bool): If True, the layer does not return the output 
            features but the hidden states.
        return_sequences (bool): If True, the layer does not return only the 
            last output in the output sequence but the full sequence.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        input_size (int): It represents the dimensionality of the input 
            features.
        hidden_size (int): It represents the number of units in the hidden 
            state, which captures the network's internal representation of 
            the input sequence.
        bidirectional (bool): Whether the layer is bidirectional or not.
        dropout (float): If non-zero, it introduces a Dropout layer on 
            the outputs of the current sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are 
            provided as (batch, seq, feature) instead of (seq, batch, feature).
            Only relevant to PyTorch.
        return_hidden (bool): If True, the layer does not return the output 
            features but the hidden states.
        return_sequences (bool): If True, the layer does not return only the 
            last output in the output sequence but the full sequence.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, input_size: int, hidden_size: int, 
                 bidirectional: bool = False, dropout: float = 0.0, 
                 batch_first: bool = True, return_hidden: bool = False, 
                 return_sequences: bool = False, actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)
        self.bidirectional: bool = bidirectional
        self.dropout: float = dropout
        self.batch_first: bool = batch_first
        self.return_hidden: bool = return_hidden
        self.return_sequences: bool = return_sequences
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size

    @property
    def input_size(self) -> int:
        """int: Get the dimensionality of the input features of the layer."""
        return self.__input_size

    @input_size.setter
    def input_size(self, input_size: int):
        """int: Set the dimensionality of the input features of the layer."""
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
        """
        bool: Get whether the input and output tensors are provided as 
            (batch, seq, feature).
        """
        return self.__batch_first

    @batch_first.setter
    def batch_first(self, batch_first: bool):
        """
        bool: Set whether the input and output tensors are provided as 
            (batch, seq, feature).
        """
        self.__batch_first = batch_first

    @property
    def return_hidden(self) -> bool:
        """
        bool: Get whether to return the output features or the hidden 
            states.
        """
        return self.__return_hidden

    @return_hidden.setter
    def return_hidden(self, return_hidden: bool):
        """
        bool: Set whether to return the output features or the hidden 
            states.
        """
        self.__return_hidden = return_hidden

    @property
    def return_sequences(self) -> bool:
        """
        bool: Get whether to return the full output sequences or only 
            the last one.
        """
        return self.__return_sequences

    @return_sequences.setter
    def return_sequences(self, return_sequences: bool):
        """
        bool: Set whether to return the full output sequences or only 
            the last one.
        """
        self.__return_sequences= return_sequences

    def __repr__(self):
        return (
            f'RNN({self.name}, {self.actv_func}, {self.input_size}, '
            f'{self.hidden_size}, {self.bidirectional}, {self.dropout}, '
            f'{self.batch_first}, {self.return_hidden}, '
            f'{self.return_sequences}, {self.name_layer_input}, '
            f'{self.input_reused})'
        )

class SimpleRNNLayer(RNN):
    """
    Represents a fully-connected RNN layer where the output is to be fed 
    back as the new input.

    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input 
            features.
        hidden_size (int): It represents the number of units in the hidden 
            state, which captures the network's internal representation of 
            the input sequence.
        bidirectional (bool): Whether the layer is bidirectional or not.
        dropout (float): If non-zero, it introduces a Dropout layer on 
            the outputs of the RNN sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are 
            provided as (batch, seq, feature) instead of (seq, batch, feature).
            Only relevant to PyTorch.
        return_hidden (bool): If True, the layer does not return the output 
            features but the hidden states.
        return_sequences (bool): If True, the layer does not return only the 
            last output in the output sequence but the full sequence.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        input_size (int): Inherited from RNN. It represents the dimensionality
            of the input features.
        hidden_size (int): Inherited from RNN. It represents the number of 
            units in the hidden state, which captures the network's internal 
            representation of the input sequence.
        bidirectional (bool): Inherited from RNN. Whether the layer is 
            bidirectional or not.
        dropout (float): Inherited from RNN. If non-zero, it introduces a 
            Dropout layer on the outputs of the RNN sub layers except 
            the last one.
        batch_first (bool): Inherited from RNN. If True, the input and output 
            tensors are provided as (batch, seq, feature) instead of 
            (seq, batch, feature). Only relevant to PyTorch.
        return_hidden (bool): Inherited from Layer. If True, the layer does 
            not return the output features but the hidden states.
        return_sequences (bool): Inherited from Layer. If True, the layer does
            not return only the last output in the output sequence but the full
            sequence.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, input_size: int, hidden_size: int,
                 bidirectional: bool = False, dropout: float = 0.0, 
                 batch_first: bool = True, return_hidden: bool = False, 
                 return_sequences: bool = False, actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        super().__init__(name, input_size, hidden_size, bidirectional, 
                         dropout, batch_first, return_hidden, 
                         return_sequences, actv_func, name_layer_input, 
                         input_reused)

    def __repr__(self):
        return (
            f'SimpleRNNLayer({self.name}, {self.actv_func}, '
            f'{self.input_size}, {self.hidden_size}, {self.bidirectional}, '
            f'{self.dropout}, {self.batch_first}, {self.return_hidden}, '
            f'{self.return_sequences}, {self.name_layer_input}, '
            f'{self.input_reused})'
        )
    

class LSTMLayer(RNN):
    """
    Represents a Long Short-Term Memory layer.
 
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input 
            features.
        hidden_size (int): It represents the number of units in the hidden 
            state, which captures the network's internal representation of 
            the input sequence.
        bidirectional (bool): Whether the layer is bidirectional or not.
        dropout (float): If non-zero, it introduces a Dropout layer on 
            the outputs of the LSTM sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are 
            provided as (batch, seq, feature) instead of (seq, batch, feature).
            Only relevant to PyTorch.
        return_hidden (bool): If True, the layer does not return the output 
            features but the hidden states.
        return_sequences (bool): If True, the layer does not return only the 
            last output in the output sequence but the full sequence.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        input_size (int): Inherited from RNN. It represents the dimensionality
            of the input features.
        hidden_size (int): Inherited from RNN. It represents the number of 
            units in the hidden state, which captures the network's internal 
            representation of the input sequence.
        bidirectional (bool): Inherited from RNN. Whether the layer is 
            bidirectional or not.
        dropout (float): Inherited from RNN. If non-zero, it introduces a 
            Dropout layer on the outputs of the LSTM sub layers except 
            the last one.
        batch_first (bool): Inherited from RNN. If True, the input and output 
            tensors are provided as (batch, seq, feature) instead of 
            (seq, batch, feature). Only relevant to PyTorch.
        return_hidden (bool): Inherited from Layer. If True, the layer does 
            not return the output features but the hidden states.
        return_sequences (bool): Inherited from Layer. If True, the layer does
            not return only the last output in the output sequence but the full
            sequence.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, input_size: int, hidden_size: int, 
                 bidirectional: bool = False, dropout: float = 0.0, 
                 batch_first: bool = True, return_hidden: bool = False, 
                 return_sequences: bool = False, actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        super().__init__(name, input_size, hidden_size, bidirectional, 
                         dropout, batch_first, return_hidden, 
                         return_sequences, actv_func, name_layer_input, 
                         input_reused)

    def __repr__(self):
        return (
            f'LSTMLayer({self.name}, {self.actv_func}, {self.input_size}, '
            f'{self.hidden_size}, {self.bidirectional}, {self.dropout}, '
            f'{self.batch_first}, {self.return_hidden}, {self.return_sequences}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )
        

class GRULayer(RNN):
    """
    Represents a Gated Recurrent Unit layer.
 
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input 
            features.
        hidden_size (int): It represents the number of units in the hidden 
            state, which captures the network's internal representation of 
            the input sequence.
        bidirectional (bool): Whether the layer is bidirectional or not.
        dropout (float): If non-zero, it introduces a Dropout layer on 
            the outputs of the GRU sub layers except the last one.
        batch_first (bool): If True, the input and output tensors are 
            provided as (batch, seq, feature) instead of (seq, batch, feature).
            Only relevant to PyTorch.
        return_hidden (bool): If True, the layer does not return the output 
            features but the hidden states.
        return_sequences (bool): If True, the layer does not return only the 
            last output in the output sequence but the full sequence.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        input_size (int): Inherited from RNN. It represents the dimensionality
            of the input features.
        hidden_size (int): Inherited from RNN. It represents the number of 
            units in the hidden state, which captures the network's internal 
            representation of the input sequence.
        bidirectional (bool): Inherited from RNN. Whether the layer is 
            bidirectional or not.
        dropout (float): Inherited from RNN. If non-zero, it introduces a 
            Dropout layer on the outputs of the GRU sub layers except the 
            last one.
        batch_first (bool): Inherited from RNN. If True, the input and output 
            tensors are provided as (batch, seq, feature) instead of 
            (seq, batch, feature). Only relevant to PyTorch.
        return_hidden (bool): Inherited from Layer. If True, the layer does 
            not return the output features but the hidden states.
        return_sequences (bool): Inherited from Layer. If True, the layer does
            not return only the last output in the output sequence but the full
            sequence.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, input_size: int, hidden_size: int, 
                 bidirectional: bool = False, dropout: float = 0.0, 
                 batch_first: bool = True, return_hidden: bool = False, 
                 return_sequences: bool = False, actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        super().__init__(name, input_size, hidden_size, bidirectional, 
                         dropout, batch_first, return_hidden, return_sequences, 
                         actv_func, name_layer_input, input_reused)

    def __repr__(self):
        return (
            f'GRULayer({self.name}, {self.actv_func}, {self.input_size}, '
            f'{self.hidden_size}, {self.bidirectional}, {self.dropout}, '
            f'{self.batch_first}, {self.return_hidden}, {self.return_sequences}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )

class GeneralLayer(Layer):
    """
    Represents a layer designed to handle general operations and 
    transformations.
 
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, actv_func: str = None, 
                 name_layer_input: str = None, input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)

    def __repr__(self):
        return (
            f'GeneralLayer({self.name}, {self.actv_func}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )

class LinearLayer(GeneralLayer):
    """
    Represents a densely-connected NN layer that applies a linear 
    transformation to the input data.
 
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        in_features (int): It represents the size of each input sample.
        out_features (int): It represents the size of each output sample.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        in_features (int): It represents the size of each input sample.
        out_features (int): It represents the size of each output sample.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, in_features: int, out_features: int, 
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)
        self.in_features: int = in_features
        self.out_features: int = out_features

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

    def __repr__(self):
        return (
            f'LinearLayer({self.name}, {self.actv_func}, {self.in_features}, ' 
            f'{self.out_features}, {self.name_layer_input}, '
            f'{self.input_reused})'
        )

class FlattenLayer(GeneralLayer):
    """
    Represents a layer that flattens a contiguous range of dims into a tensor.
 
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        start_dim (int): The first dimension to flatten.
        end_dim (int): The last dim to flatten.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        start_dim (int): The first dimension to flatten.
        end_dim (int): The last dim to flatten.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, start_dim: int = 1, end_dim: int = -1, 
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)
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
            f'{self.end_dim}, {self.name_layer_input}, {self.input_reused})'
        )

        
class EmbeddingLayer(GeneralLayer):
    """
    Represents a layer that learns dense vector representations of the input 
    data.
 
    Args:
        name (str): The name of the layer.
        actv_func (str): The type of the activation function.
        num_embeddings (int): The size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        name_layer_input (str): The name of the layer from which the inputs 
            originate.
        input_reused (bool): Whether the input to this layer is reused as 
            input to another layer.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        actv_func (str): Inherited from Layer. It represents the type of 
            the activation function.
        num_embeddings (int): The size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        name_layer_input (str): Inherited from Layer. The name of the layer 
            from which the inputs originate.
        input_reused (bool): Inherited from Layer. Whether the input to this 
            layer is reused as input to another layer.
    """
    def __init__(self, name: str, num_embeddings: int, embedding_dim: int, 
                 actv_func: str = None, name_layer_input: str = None, 
                 input_reused: bool = False):
        super().__init__(name, actv_func, name_layer_input, input_reused)
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim

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

    def __repr__(self):
        return (
            f'EmbeddingLayer({self.name}, {self.actv_func}, '
            f'{self.num_embeddings}, {self.embedding_dim}, '
            f'{self.name_layer_input}, {self.input_reused})'
        )


class Feature:
    """
    A feature is a measurable property or characteristic of an object used 
    to represent and describe it within a dataset.
 
    Args:
        name (str): The name of the feature.
        
    Attributes:
        name (str): The name of the feature.
    """
    def __init__(self, name: str):
        self.name: str = name

    @property
    def name(self) -> str:
        """str: Get the name of the feature."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the feature."""
        self.__name = name

    def __repr__(self):
        return f'Feature({self.name})'        
    

class Label:
    """
    A label is a value assigned to an observation, representing the target 
    variable for prediction.
 
    Args:
        col_name (str): The name of the column containing the labels.
        label_name (str): The name of a label in the dataset. If 
            the prediction task is regression, it can be omitted.
        
    Attributes:
        col_name (str): The name of the column containing the labels.
        label_name (str): The name of a label in the dataset. If 
            the prediction task is regression, it can be omitted.
    """
    def __init__(self, col_name: str, label_name: str = None):
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
    characteristics, typically including attributes such as height and width.

    Args:
        shape (list[int]): The shape of the image in the form 
            [height, width, channels].
        
    Attributes:
        shape (list[int]): The shape of the image in the form 
            [height, width, channels].

    """
    def __init__(self, shape: List[int]):
        self.shape: List[int] = shape

    @property
    def shape(self) -> List[int]:
        """List[int]: Get the shape of the image."""
        return self.__shape

    @shape.setter
    def shape(self, shape: List[int]):
        """List[int]: Set the shape of the image."""
        self.__shape = shape


    def __repr__(self):
        return f'Image({self.shape})' 

class Structured(Feature):
    """
    Represents features organized in a systematic manner, typically with 
    well-defined columns and rows, often found in tabular datasets.

    Args:
        name (str): The name of the feature.
        
    Attributes:
        name (str): Inherited from Feature. It represents the name of 
            the feature.
    """
    def __init__(self, name: str):
        super().__init__(name)
        
    def __repr__(self):
        return f'Structured({self.name})'



class Dataset:
    """
    Represents the collection of data instances used for training or 
    evaluation, where each instance comprises features and corresponding 
    labels.

    Args:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing 
            the dataset.
        task_type (str): The type of prediction task associated with 
            the dataset.
        input_format (str): The format of the input dataset.
        image (Image): An image instance that contains the shape desired for 
            the images if input_format parameter is set to 'images'.
        labels (set[Label]): The set of labels in the dataset.
        
    Attributes:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing 
            the dataset.
        task_type (str): The type of prediction task associated with 
            the dataset.
        input_format (str): The format of the input dataset.
        image (Image): An image instance that contains the shape desired for 
            the images if input_format parameter is set to 'images'.
        labels (set[Label]): The set of labels in the dataset.
    """
    def __init__(self, name: str, path_data: str, task_type: str = None, 
                 input_format: str = None, image: Image = None, 
                 labels: set[Label] = set()):
        self.name: str = name
        self.path_data: str = path_data
        if task_type is not None:
            self.task_type: str = task_type
        if input_format is not None:
            self.input_format: str = input_format
        self.image: Image = image
        self.labels: set[Label] = labels

    @property
    def name(self) -> str:
        """str: Get the name of the dataset."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the dataset."""
        self.__name = name

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
        """
        str: Get the type of prediction task associated with the dataset.
        """
        return self.__task_type

    @task_type.setter
    def task_type(self, task_type: str):
        """
        str: Set the type of prediction task associated with the dataset.
        
        Raises:
            ValueError: If task_type is not one of the allowed 
            options: 'binary', 'multi_class', and 'regression'
        """

        if task_type not in ['binary', 'multi_class', 'regression']:
            raise ValueError("Invalid value of task_type")

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
            raise ValueError("Invalid value of input_format")
        
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

    def load_data(self):
        pass

    def split_features_labels(self):
        pass

    def __repr__(self):
        return (
            f'Dataset({self.name}, {self.path_data}, {self.task_type}, '
            f'{self.input_format}, {self.image}, {self.labels})'
        )  
    
class TrainingDataset(Dataset):
    """
    Represents a Dataset used for training the NN model.

    Args:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing 
            the dataset.
        task_type (str): The type of prediction task associated with 
            the dataset.
        input_format (str): The format of the input dataset.
        image (Image): An image instance that contains the shape desired for 
            the images if input_format parameter is set to 'images'.
        labels (set[Label]): The set of labels in the dataset.
        
    Attributes:
        name (str): Inherited from Dataset. It represents the name of 
            the dataset.
        path_data (str): Inherited from Dataset. It represents the file path 
            containing the dataset.
        task_type (str): Inherited from Dataset. It represents the type of 
            prediction task associated with the dataset.
        input_format (str): Inherited from Dataset. The format of the input 
            dataset.
        image (Image): Inherited from Dataset. An image instance that 
            contains the shape desired for the images if input_format 
            parameter is set to 'images'.
        labels (set[Label]): Inherited from Dataset. It represents the set 
            of labels in the dataset.
    """

    def __init__(self, name: str, path_data: str, task_type: str, 
                 input_format: str, image: Image = None, 
                 labels: set[Label] = set()):
        super().__init__(name, path_data, task_type, input_format, image, 
                         labels)
    
    def __repr__(self):
        return (
            f'TrainingDataset({self.name}, {self.path_data}, '
            f'{self.task_type}, {self.input_format}, {self.image}, '
            f'{self.labels})'
        )

class TestDataset(Dataset):
    """
    Represents a Dataset used for evaluating the performance of the NN model.

    Args:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing 
            the dataset.
        
    Attributes:
        name (str): Inherited from Dataset. It represents the name of 
            the dataset.
        path_data (str): Inherited from Dataset. It represents the file path 
            containing the dataset.
    """

    def __init__(self, name: str, path_data: str):
        super().__init__(name=name, path_data=path_data)
    
    def __repr__(self):
        return f'TestDataset({self.name}, {self.path_data})' 


class Parameters:
    """
    Represents a collection of parameters essential for training and 
    evaluating neural networks.

    Args:
        batch_size (int): The number of data samples processed in each 
            iteration during training or inference in a neural network.
        epochs (int): It refers to the number of complete passes through 
            the entire dataset during the training, with each epoch 
            consisting of one iteration through all data samples.
        learning_rate (float): The step size used to update the model 
            parameters during optimisation.
        optimiser (str): The method or algorithm used to adjust the model 
            parameters iteratively during training to minimize the loss 
            function and improve model performance.
        loss_function (str): The method used to calculate the difference 
            between predicted and actual values, guiding the model towards 
            better predictions.
        metrics List[str]: Quantitative measures used to evaluate 
            the performance of NN models.
        weight_decay (float): It represents the strength of L2 regularisation 
            applied to the model's parameters during optimisation.
        
    Attributes:
        batch_size (int): The number of data samples processed in each 
            iteration during training or inference in a neural network.
        epochs (int): It refers to the number of complete passes through 
            the entire dataset during the training, with each epoch 
            consisting of one iteration through all data samples.
        learning_rate (float): The step size used to update the model 
            parameters during optimisation.
        optimiser (str): The method or algorithm used to adjust the model 
            parameters iteratively during training to minimize the loss 
            function and improve model performance.
        loss_function (str): The method used to calculate the difference 
            between predicted and actual values, guiding the model towards 
            better predictions.
        metrics List[str]: Quantitative measures used to evaluate 
            the performance of NN models.
        weight_decay (float): It represents the strength of L2 regularisation 
            applied to the model's parameters during optimisation.
    """
    def __init__(self, batch_size: int, epochs: int, learning_rate: float, 
                 optimiser: str, loss_function: str, metrics: List[str], 
                 weight_decay: float = 0):
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.optimiser: str = optimiser
        self.loss_function: str = loss_function
        self.metrics: List[str] = metrics
        self.weight_decay: float = weight_decay

    @property
    def batch_size(self) -> int:
        """
        int: Get the number of data samples processed in each iteration 
            during training or inference in a neural network.
        """
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """
        int: Set the number of data samples processed in each iteration 
           during training or inference in a neural network.
        """
        self.__batch_size = batch_size

    @property
    def epochs(self) -> int:
        """
        int: Get the number of complete passes through the entire dataset 
            during the training.
        """
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs: int):
        """
        int: Set the number of complete passes through the entire dataset 
            during the training.
        """
        self.__epochs = epochs

    @property
    def learning_rate(self) -> float:
        """
        float: Get the step size used to update the model parameters during 
            optimisation.
        """
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        """
        float: Set step size used to update the model parameters during 
            optimisation.
        """
        self.__learning_rate = learning_rate

    @property
    def optimiser(self) -> str:
        """
        str: Get the algorithm used to adjust the model parameters 
            iteratively during training to minimize the loss function.
        """
        return self.__optimiser

    @optimiser.setter
    def optimiser(self, optimiser: str):
        """
        str: Set the algorithm used to adjust the model parameters 
            iteratively during training to minimize the loss function.
        
        Raises:
            ValueError: If optimiser is not one of the allowed 
            options: 'sgd', 'adam', 'adamW' and 'adagrad'
        """

        if optimiser not in ['sgd', 'adam', 'adamW', 'adagrad']:
            raise ValueError("Invalid value of optimiser")
        self.__optimiser = optimiser

    @property
    def loss_function(self) -> str:
        """
        str: Get the method used to calculate the difference between 
            predicted and actual values, guiding the model towards better 
            predictions.
        """
        return self.__loss_function


    @loss_function.setter
    def loss_function(self, loss_function: str):
        """
        str: Set the method used to calculate the difference between 
            predicted and actual values, guiding the model towards better 
            predictions.
        
        Raises:
            ValueError: If loss_function is not one of the allowed 
            options: 'crossentropy', 'binary_crossentropy' and 'mse'
        """

        if loss_function not in [
            'crossentropy', 'binary_crossentropy', 'mse'
        ]:
            raise ValueError("Invalid value of loss_function")
        self.__loss_function = loss_function

    @property
    def metrics(self) -> List[str]:
        """
        List[str]: Get the measures for evaluating the performance of 
            the model.
        """
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """
        List[str]: Set the measures for evaluating the performance of 
            the model.
        
        Raises:
            ValueError: If metrics is not one of the allowed 
            options: accuracy', 'precision', 'recall', 'f1-score' and 'mae'
        """
        if isinstance(metrics, list) and \
            all(isinstance(metric, str) for metric in metrics):
            if all(metric in ['accuracy', 'precision', 'recall', 
                              'f1-score', 'mae'] for metric in metrics):
                self.__metrics = metrics
            else:
                invalid_metrics = [
                    metric for metric in metrics 
                    if metric not in ['accuracy', 'precision', 'recall', 
                                      'f1-score', 'mae']
                ]
                raise ValueError(
                    f"Invalid metric(s) provided: {invalid_metrics}"
                )
        else:
            raise ValueError("'metrics' must be a list of strings.")

    @property
    def weight_decay(self) -> float:
        """
        float: Get the strength of L2 regularisation applied during 
            optimisation.
        """
        return self.__weight_decay

    @weight_decay.setter
    def weight_decay(self, weight_decay: float):
        """
        float: Set the strength of L2 regularisation applied during 
            optimisation.
        """
        self.__weight_decay = weight_decay


    def __repr__(self):
        return (
            f'Parameters({self.batch_size}, {self.epochs}, '
            f'{self.learning_rate}, {self.optimiser}, {self.loss_function}, '
            f'{self.metrics}, {self.weight_decay})'
        )

class NN(BehavioralImplementation):
    """
    It is a subclass of the BehaviorImplementation class and comprises 
    the fundamental properties and behaviors of a neural network model.

    Args:
        name (str): The name of the neural network model.
        parameters (Parameters): The parameters related to the NN training 
            and evaluation. 
        
    Attributes:
        name (str): The name of the neural network model.
        parameters (Parameters): The parameters related to the NN training 
            and evaluation. 
    """
    def __init__(self, name: str, parameters: Parameters = None):
        super().__init__(name)
        self.parameters: Parameters = parameters
        self.__sub_nns: List[NN] = []
        self.__layers: List[Layer] = []
        self.__tensor_ops: List[TensorOp] = []
        self.__modules: List[Union[NN, Layer, TensorOp]] = []

    @property
    def sub_nns(self) -> List[NN]:
        """List[NN]: Get the sub NN models list of the main model."""
        return self.__sub_nns
            
    @sub_nns.setter
    def sub_nns(self, sub_nns: List[NN]):
        """List[NN]: Set the sub NN models list of the main model."""
        raise AttributeError("sub_nns attribute is read-only")
        
        
    def add_sub_nn(self, sub_nn: NN):
        if isinstance(sub_nn, NN):
            self.__sub_nns.append(sub_nn)
            self.__modules.append(sub_nn)
        else:
            raise ValueError("'sub_nn' must be of type NN.")
        return self
    
    @property
    def layers(self) -> List[Layer]:
        """List[Layer]: Get the list of layers."""
        return self.__layers
            
    @layers.setter
    def layers(self, layers: List[Layer]):
        """List[Layer]: Set the list of layers."""
        raise AttributeError("layers attribute is read-only")

    def add_layer(self, layer: Layer) -> Self:
        """Self: Add a layer to the NN model."""
        if isinstance(layer, Layer):
            self.__layers.append(layer)
            self.__modules.append(layer)
        else:
            raise ValueError("'layer' must be of type Layer.")
        return self

    @property
    def tensor_ops(self) -> List[TensorOp]:
        """List[TensorOp]: Get the list of tensor Ops."""
        return self.__tensor_ops
            
    @tensor_ops.setter
    def tensor_ops(self, tensor_ops: List[TensorOp]):
        """List[TensorOp]: Set the list of tensor Ops ."""
        raise AttributeError("tensor_ops attribute is read-only")

    def add_tensor_op(self, tensor_op: TensorOp) -> Self:
        """Self: Add a tensor Op to the NN model."""
        if isinstance(tensor_op, TensorOp):
            self.__tensor_ops.append(tensor_op)
            self.__modules.append(tensor_op)
        else:
            raise ValueError("'tensor_op' must be of type TensorOp.")
        return self
    
    @property
    def modules(self) -> List[Union[NN, Layer, TensorOp]]:
        """
        List[Union[NN, Layer, TensorOp]]: Get the modules list of the main 
            model.
        """
        return self.__modules
            
    @modules.setter
    def modules(self, modules: List[Union[NN, Layer, TensorOp]]):
        """
        List[Union[NN, Layer, TensorOp]]: Set the modules list of the main 
            model.
        """
        raise AttributeError("modules attribute is read-only")

    @property
    def parameters(self) -> Parameters:
        """
        Parameters: Get the parameters related to the NN training and 
            evaluation.
        """
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters: Parameters):
        """
        Parameters: Set the parameters related to the NN training and 
            evaluation.
        """
        self.__parameters = parameters

    def add_parameters(self, parameters: Parameters) -> Self:
        """Self: Add the parameters to the NN model."""
        self.__parameters = parameters
        return self