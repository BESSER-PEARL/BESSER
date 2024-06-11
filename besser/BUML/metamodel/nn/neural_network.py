from besser.BUML.metamodel.structural import Model, BehavioralImplementation
from typing import List
from abc import ABC, abstractmethod


class Layer:
    """This class represents a layer of the neural network. It encapsulates 
    attributes such as the name of the layer and the activation function.
    
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.

    Attributes:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
    """
    def __init__(self, name: str, activation_function: str = None):
        self.name: str = name
        self.activation_function: str = activation_function

    
    @property
    def name(self) -> str:
        """str: Get the name of the layer."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the layer."""
        self.__name = name


    @property
    def activation_function(self) -> str:
        """str: Get the activation_function."""
        return self.__activation_function


    @activation_function.setter
    def activation_function(self, activation_function: str):
        """str: Set the activation_function.

        Raises:
            ValueError: If the activation_function is not one of the allowed 
            options: 'relu', 'leaky_rely', 'sigmoid', 'softmax', and 'tanh'
        """

        if activation_function not in ['relu', 'leaky_relu', 'sigmoid', 
                                       'softmax', 'tanh', None]:
            raise ValueError("Invalid value of activation_function")
        self.__activation_function = activation_function

    
    def __repr__(self):
        return f'Layer({self.name}, {self.activation_function})'


class CNN(Layer):
    """Represents a layer that is generally used in convolutional neural networks.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        kernel_height (int): The height of the convolving or pooling kernel.
        kernel_width (int): The width of the convolving  or pooling kernel.
        kernel_depth (int): The depth of the convolving  or pooling kernel.
        stride_height (int): The height of the stride of the convolution or pooling .
        stride_width (int): The width of the stride of the convolution or pooling .
        stride_depth (int): The depth of the stride of the convolution or pooling .
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
        kernel_height (int): The height of the convolving or pooling kernel.
        kernel_width (int): The width of the convolving  or pooling kernel.
        kernel_depth (int): The depth of the convolving  or pooling kernel.
        stride_height (int): The height of the stride of the convolution or pooling .
        stride_width (int): The width of the stride of the convolution or pooling .
        stride_depth (int): The depth of the stride of the convolution or pooling .
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 
    """

    def __init__(self, name: str, activation_function: str, kernel_height: int, 
                 kernel_width: int, kernel_depth: int, stride_height: int, stride_width: int, 
                 stride_depth: int, padding_amount: int = 0, padding_type: str = "valid"):
        super().__init__(name, activation_function)
        self.kernel_height: int = kernel_height
        self.kernel_width: int = kernel_width
        self.kernel_depth: int = kernel_depth
        self.stride_height: int = stride_height
        self.stride_width: int = stride_width
        self.stride_depth: int = stride_depth
        self.padding_amount: int = padding_amount
        self.padding_type: str = padding_type

    @property
    def kernel_height(self) -> int:
        """int: Get the height of the kernel."""
        return self.__kernel_height

    @kernel_height.setter
    def kernel_height(self, kernel_height: int):
        """int: Set the height of the kernel."""
        self.__kernel_height = kernel_height

    @property
    def kernel_width(self) -> int:
        """int: Get the width of the kernel."""
        return self.__kernel_width

    @kernel_width.setter
    def kernel_width(self, kernel_width: int):
        """int: Set the width of the kernel."""
        self.__kernel_width = kernel_width

    @property
    def kernel_depth(self) -> int:
        """int: Get the depth of the kernel."""
        return self.__kernel_depth

    @kernel_depth.setter
    def kernel_depth(self, kernel_depth: int):
        """int: Set the depth of the kernel."""
        self.__kernel_depth = kernel_depth

    @property
    def stride_height(self) -> int:
        """int: Get the height of the stride."""
        return self.__stride_height

    @stride_height.setter
    def stride_height(self, stride_height: int):
        """int: Set the height of the stride."""
        self.__stride_height = stride_height

    @property
    def stride_width(self) -> int:
        """int: Get the width of the stride."""
        return self.__stride_width

    @stride_width.setter
    def stride_width(self, stride_width: int):
        """int: Set the width of the stride."""
        self.__stride_width = stride_width

    @property
    def stride_depth(self) -> int:
        """int: Get the depth of the stride."""
        return self.__stride_depth

    @stride_depth.setter
    def stride_depth(self, stride_depth: int):
        """int: Set the depth of the stride."""
        self.__stride_depth = stride_depth

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
        return (f'CNN({self.name}, {self.activation_function}, {self.kernel_height}, '
                f'{self.kernel_width}, {self.kernel_depth}, {self.stride_height}, '
                f'{self.stride_width}, {self.stride_depth}, {self.padding_amount}, '
                f'{self.padding_type})')


    
class ConvolutionalLayer(CNN):
    """Represents a convolutional layer.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by the convolution.
        kernel_height (int): The height of the convolving kernel.
        kernel_width (int): The width of the convolving kernel.
        kernel_depth (int): The depth of the convolving kernel.
        stride_height (int): The height of the stride of the convolution.
        stride_width (int): The width of the stride of the convolution.
        stride_depth (int): The depth of the stride of the convolution.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by the convolution.
        kernel_height (int): Inherited from CNN. The height of the convolving kernel.
        kernel_width (int): Inherited from CNN. The width of the convolving kernel.
        kernel_depth (int): Inherited from CNN. The depth of the convolving kernel.
        stride_height (int): Inherited from CNN. The height of the stride of the convolution.
        stride_width (int): Inherited from CNN. The width of the stride of the convolution.
        stride_depth (int): Inherited from CNN. The depth of the stride of the convolution.
        padding_amount (int): Inherited from CNN. The amount of padding added to the input.
        padding_type (str): Inherited from CNN. The type of padding applied to the input. 
    """
        
    def __init__(self, name: str, activation_function: str, in_channels: int, 
                 out_channels: int, kernel_height: int, kernel_width: int, 
                 kernel_depth: int, stride_height: int, stride_width: int, 
                 stride_depth: int, padding_amount: int = 0, padding_type: str = "valid"):
        super().__init__(name, activation_function, kernel_height, kernel_width, kernel_depth, 
                         stride_height, stride_width, stride_depth, padding_amount, padding_type)
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
        return (f'ConvolutionaLayer({self.name}, {self.activation_function},  {self.in_channels}, '
                f'{self.out_channels}, {self.kernel_height}, {self.kernel_width}, '
                f'{self.kernel_depth}, {self.stride_height}, {self.stride_width}, '
                f'{self.stride_depth}, {self.padding_amount}, {self.padding_type})')
   

class Conv1D(ConvolutionalLayer):
    """Represents a type of convolutional layer that applies a 1D convolution.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by the convolution.
        kernel_height (int): The height of the convolving kernel.
        stride_height (int): The height of the stride of the convolution.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
        in_channels (int): Inherited from ConvolutionalLayer. It represents the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It represents the number of channels produced by the convolution.
        kernel_height (int): Inherited from CNN. It represents the height of the convolving kernel.
        stride_height (int): Inherited from CNN. It represents the height of the stride of the convolution.
        padding_amount (int): Inherited from CNN. It represents the amount of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type of padding applied to the input. 
    """
    def __init__(self, name: str, activation_function: str, in_channels: int, 
                 out_channels: int, kernel_height: int, stride_height: int, 
                 padding_amount: int = 0, padding_type: str = "valid"):
        super().__init__(name, activation_function, in_channels, out_channels, 
                         kernel_height, None, None, stride_height, None, None, 
                         padding_amount, padding_type)
        
    def __repr__(self):
        return (f'Conv1D({self.name}, {self.activation_function}, {self.in_channels}, '
                f'{self.out_channels}, {self.kernel_height}, {self.stride_height}, ' 
                f'{self.padding_amount}, {self.padding_type})')
        
        

class Conv2D(ConvolutionalLayer):
    """Represents a type of convolutional layer that applies a 2D convolution.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by the convolution.
        kernel_height (int): The height of the convolving kernel.
        kernel_width (int): The width of the convolving kernel.
        stride_height (int): The height of the stride of the convolution.
        stride_width (int): The width of the stride of the convolution.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
        in_channels (int): Inherited from ConvolutionalLayer. It represents the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It represents the number of channels produced by the convolution.
        kernel_height (int): Inherited from CNN. It represents the height of the convolving kernel.
        kernel_width (int): Inherited from CNN. It represents the width of the convolving kernel.
        stride_height (int): Inherited from CNN. It represents the height of the stride of the convolution.
        stride_width (int): Inherited from CNN. It represents the width of the stride of the convolution.
        padding_amount (int): Inherited from CNN. It represents the amount of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type of padding applied to the input. 
    """
    def __init__(self, name: str, activation_function: str, in_channels: int, 
                 out_channels: int, kernel_height: int, kernel_width: int, 
                 stride_height: int, stride_width: int, padding_amount: int = 0, 
                 padding_type: str = "valid"):
        super().__init__(name, activation_function, in_channels, out_channels, 
                         kernel_height, kernel_width, None, stride_height, 
                         stride_width, None, padding_amount, padding_type)
    
    def __repr__(self):
        return (f'Conv2D({self.name}, {self.activation_function}, {self.in_channels}, '
                f'{self.out_channels}, {self.kernel_height}, {self.kernel_width}, '
                f'{self.stride_height}, {self.stride_width}, {self.padding_amount}, '
                f'{self.padding_type})')
            
class Conv3D(ConvolutionalLayer):
    """Represents a type of convolutional layer that applies a 3D convolution.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of channels produced by the convolution.
        kernel_height (int): The height of the convolving kernel.
        kernel_width (int): The width of the convolving kernel.
        kernel_depth (int): The depth of the convolving kernel.
        stride_height (int): The height of the stride of the convolution.
        stride_width (int): The width of the stride of the convolution.
        stride_depth (int): The depth of the stride of the convolution.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input. 

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
        in_channels (int): Inherited from ConvolutionalLayer. It represents the number of channels in the input image.
        out_channels (int): Inherited from ConvolutionalLayer. It represents the number of channels produced by the convolution.
        kernel_height (int): Inherited from CNN. It represents the height of the convolving kernel.
        kernel_width (int): Inherited from CNN. It represents the width of the convolving kernel.
        kernel_depth (int): Inherited from CNN. It represents the depth of the convolving kernel.
        stride_height (int): Inherited from CNN. It represents the height of the stride of the convolution.
        stride_width (int): Inherited from CNN. It represents the width of the stride of the convolution.
        stride_depth (int): Inherited from CNN. It represents the depth of the stride of the convolution.
        padding_amount (int): Inherited from CNN. It represents the amount of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type of padding applied to the input. 
    """
    def __init__(self, name: str, activation_function: str, in_channels: int, out_channels: int, 
                 kernel_height: int, kernel_width: int, kernel_depth: int, stride_height: int, 
                 stride_width: int, stride_depth: int, padding_amount: int = 0, padding_type: str = "valid"):
        super().__init__(name, activation_function, in_channels, out_channels, kernel_height, 
                         kernel_width, kernel_depth, stride_height, stride_width, stride_depth, 
                         padding_amount, padding_type)
        
    def __repr__(self):
        return (f'Conv3D({self.name}, {self.activation_function},  {self.in_channels}, '
                f'{self.out_channels}, {self.kernel_height}, {self.kernel_width}, '
                f'{self.kernel_depth}, {self.stride_height}, {self.stride_width}, '
                f'{self.stride_depth}, {self.padding_amount}, {self.padding_type})')


class PoolingLayer(CNN):
    """Represents a type of layer that performs a pooling operation.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        type (str): The type of pooling applied.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the pooling operation.
        kernel_height (int): The height of the pooling kernel.
        kernel_width (int): The width of the pooling kernel.
        kernel_depth (int): The depth of the pooling kernel.
        stride_height (int): The height of the stride of the pooling.
        stride_width (int): The width of the stride of the pooling.
        stride_depth (int): The depth of the stride of the pooling.
        padding_amount (int): The amount of padding added to the input.
        padding_type (str): The type of padding applied to the input.
        pooling_type (str): The type of pooling. Either average or max.
        dimension (str): The dimensionality of the pooling. Either 1D, 2D or 3D.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
        type (str): It represents the type of pooling applied.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the pooling operation.
        kernel_height (int): Inherited from CNN. It represents the height of the pooling kernel.
        kernel_width (int): Inherited from CNN. It represents the width of the pooling kernel.
        kernel_depth (int): Inherited from CNN. It represents the depth of the pooling kernel.
        stride_height (int): Inherited from CNN. It represents the height of the stride of the pooling.
        stride_width (int): Inherited from CNN. It represents the width of the stride of the pooling.
        stride_depth (int): Inherited from CNN. It represents the depth of the stride of the pooling.
        padding_amount (int): Inherited from CNN. It represents the amount of padding added to the input.
        padding_type (str): Inherited from CNN. It represents the type of padding applied to the input. 
        pooling_type (str): The type of pooling. Either average or max.
        dimension (str): The dimensionality of the pooling. Either 1D, 2D or 3D.

    """
    def __init__(self, name: str, activation_function: str, pooling_type: str, dimension: str, 
                 kernel_height: int, stride_height: int, kernel_width: int = None, 
                 kernel_depth: int = None, stride_width: int = None, stride_depth: int = None, 
                 padding_amount: int = 0, padding_type: str = "valid"):
        super().__init__(name, activation_function, kernel_height, kernel_width, kernel_depth, 
                         stride_height, stride_width, stride_depth, padding_amount, padding_type)
        self.pooling_type: str = pooling_type
        self.dimension: str = dimension

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
            these: 'average' or 'max'.
        """

        if pooling_type not in ['average', 'max']:
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


    def __repr__(self):
        return (f'PoolingLayer({self.name}, {self.activation_function}, {self.pooling_type}, '
                f'{self.dimension}, {self.kernel_height}, {self.stride_height}, '
                f'{self.kernel_width}, {self.kernel_depth}, {self.stride_width}, '
                f'{self.stride_depth}, {self.padding_amount}, {self.padding_type})')
    

class LayerModifier(Layer):
    """Represents a type of layer that adjusts preceding layer outputs, incorporating 
       techniques such as normalization and dropout.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
    """
    def __init__(self, name: str, activation_function: str):
        super().__init__(name, activation_function)

    def __repr__(self):
        return f'LayerModifier({self.name}, {self.activation_function})'
        
class NormalizationLayer(LayerModifier):
    """Represents a type of layer that applies normalization techniques.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
    """
    def __init__(self, name: str, activation_function: str):
        super().__init__(name, activation_function)

    def __repr__(self):
        return f'NormalizationLayer({self.name}, {self.activation_function})'
        
class BatchNormLayer(NormalizationLayer):
    """Represents a type of layer that applies Batch Normalization.
    
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        num_features (int): The number of channels or features in each input sample.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the input data to be 
                         normalized using batch normalization.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
        num_features (int): The number of channels or features in each input sample.
        dimension (str): The dimensionality (1D, 2D, or 3D) of the input data to be 
                         normalized using batch normalization.
    """
    def __init__(self, name: str, activation_function: str, num_features: int, dimension: str):
        super().__init__(name, activation_function)
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
        """str: Get the dimensionality of the input data to be normalized."""
        return self.__dimension

    @dimension.setter
    def dimension(self, dimension: str):
        """
        str: Set the dimensionality of the input data to be normalized.
        
        Raises:
            ValueError: If the dimensionality of the input data is none of 
            these: '1D', '2D', or '3D'.
        """

        if dimension not in ['1D', '2D', '3D']:
            raise ValueError ("Invalid data dimensionality")  
        self.__dimension = dimension


    def __repr__(self):
        return (f'BatchNormLayer({self.name}, {self.activation_function}, '
                f'{self.num_features}, {self.dimension})')

class LayerNormLayer(NormalizationLayer):
    """Represents a type of layer that applies Layer Normalization.
    
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        norm_channels (int): It represents the number of channels or features in the input 
                             data, guiding per-channel normalization across samples.
        norm_height (int): It represents the height of each input data, guiding normalization
                           along the height dimension.
        norm_width (int): It represents the width of each input data, guiding normalization
                          along the width dimension.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type 
                                   of the activation function.
        norm_channels (int): It represents the number of channels or features in each input 
                             sample, guiding per-channel normalization across samples.
        norm_height (int): It represents the height of the input, guiding normalization
                           along the height dimension.
        norm_width (int):  It represents the width of the input sample, guiding normalization
                           along the width dimension.
    """
    def __init__(self, name: str, activation_function: str, norm_channels: int,
                 norm_height: int = None, norm_width: int = None):
        super().__init__(name, activation_function)
        self.norm_channels: int = norm_channels
        self.norm_height: int = norm_height
        self.norm_width: int = norm_width

    @property
    def norm_channels(self) -> int:
        """int: Get the number of channels or features."""
        return self.__norm_channels

    @norm_channels.setter
    def norm_channels(self, norm_channels: int):
        """int: Set the number of channels or features."""
        self.__norm_channels = norm_channels

    @property
    def norm_height(self) -> int:
        """int: Get the height dimension used for normalization."""
        return self.__norm_height

    @norm_height.setter
    def norm_height(self, norm_height: int):
        """int: Set the height dimension used for normalization."""
        self.__norm_height = norm_height

    @property
    def norm_width(self) -> int:
        """int: Get the width dimension used for normalization."""
        return self.__norm_width

    @norm_width.setter
    def norm_width(self, norm_width: int):
        """int: Set the width dimension used for normalization."""
        self.__norm_width = norm_width

    def __repr__(self):
        return (f'LayerNormLayer({self.name}, {self.activation_function}, '
                f'{self.norm_channels}, {self.norm_height}, ' 
                f'{self.norm_width})')

class DropoutLayer(LayerModifier):
    """Represents a type of layer that applies dropout to the input.
    
    Args:
        name (str): The name of the layer.
        rate (float): It represents a float between 0 and 1. Fraction of the input units to drop. 

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        rate (float): It represents a float between 0 and 1. Fraction of the input units to drop. 
    """
    def __init__(self, name: str, rate: float):
        super().__init__(name, None)
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
        return f'DropoutLayer({self.name}, {self.rate})'


class RNN(Layer):
    """Represents a type of layer generally used in recurrent neural networks (RNN) for processing sequential data.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input features.
        hidden_size (int): It represents the number of units in the hidden state, which captures the network's 
                           internal representation of the input sequence.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
        input_size (int): It represents the dimensionality of the input features.
        hidden_size (int): It represents the number of units in the hidden state, which captures the network's 
                           internal representation of the input sequence.
    """
    def __init__(self, name: str, activation_function: str, input_size: int, hidden_size: int):
        super().__init__(name, activation_function)
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


    def __repr__(self):
        return (f'RNN({self.name}, {self.activation_function}, {self.input_size}, '
                f'{self.hidden_size})')

class SimpleRNNLayer(RNN):
    """Represents a fully-connected RNN layer where the output is to be fed back as the new input.

    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input features.
        hidden_size (int): It represents the number of units in the hidden state, which captures the network's 
                           internal representation of the input sequence.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
        input_size (int): Inherited from RNN. It represents the dimensionality of the input features.
        hidden_size (int): Inherited from RNN. It represents the number of units in the hidden state, which captures 
                           the network's internal representation of the input sequence.
    """
    def __init__(self, name: str, activation_function: str, input_size: int, hidden_size: int):
        super().__init__(name, activation_function, input_size, hidden_size)

    def __repr__(self):
        return f'SimpleRNNLayer({self.name}, {self.activation_function}, {self.input_size}, {self.hidden_size})'
    

class LSTMLayer(RNN):
    """Represents a Long Short-Term Memory layer.
 
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input features.
        hidden_size (int): It represents the number of units in the hidden state, which captures the network's 
                           internal representation of the input sequence.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
        input_size (int): Inherited from RNN. It represents the dimensionality of the input features.
        hidden_size (int): Inherited from RNN. It represents the number of units in the hidden state, which captures 
                           the network's internal representation of the input sequence.
    """
    def __init__(self, name: str, activation_function: str, input_size: int, hidden_size: int):
        super().__init__(name, activation_function, input_size, hidden_size)

    def __repr__(self):
        return f'LSTMLayer({self.name}, {self.activation_function}, {self.input_size}, {self.hidden_size})'
        

class GRULayer(RNN):
    """Represents a Gated Recurrent Unit layer.
 
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        input_size (int): It represents the dimensionality of the input features.
        hidden_size (int): It represents the number of units in the hidden state, which captures the network's 
                           internal representation of the input sequence.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
        input_size (int): Inherited from RNN. It represents the dimensionality of the input features.
        hidden_size (int): Inherited from RNN. It represents the number of units in the hidden state, which captures 
                           the network's internal representation of the input sequence.
    """
    def __init__(self, name: str, activation_function: str, input_size: int, hidden_size: int):
        super().__init__(name, activation_function, input_size, hidden_size)

    def __repr__(self):
        return f'GRULayer({self.name}, {self.activation_function}, {self.input_size}, {self.hidden_size})'

class GeneralLayer(Layer):
    """Represents a layer that encapsulates common functionalities utilized across diverse layer types.".
 
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
    """
    def __init__(self, name: str, activation_function: str):
        super().__init__(name, activation_function)

    def __repr__(self):
        return f'GeneralLayer({self.name}, {self.activation_function})'

class InputLayer(GeneralLayer):
    """Represents the initial layer of an NN architecture, serving as the entry point for input data.
 
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        input_dim1 (int): It represents the size of the first axis of the input tensor.
        input_dim2 (int): It represents the size of the second axis of the input tensor.
        input_dim3 (int): It represents the size of the third axis of the input tensor.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
        input_dim1 (int): It represents the size of the first axis of the input tensor.
        input_dim2 (int): It represents the size of the second axis of the input tensor.
        input_dim3 (int): It represents the size of the third axis of the input tensor.
    """
    def __init__(self, name: str, activation_function: str, input_dim1: int, input_dim2: int, input_dim3: int):
        super().__init__(name, activation_function)
        self.input_dim1: int = input_dim1
        self.input_dim2: int = input_dim2
        self.input_dim3: int = input_dim3

    @property
    def input_dim1(self) -> int:
        """int: Get the size of the first axis of the input tensor."""
        return self.__input_dim1

    @input_dim1.setter
    def input_dim1(self, input_dim1: int):
        """int: Set the size of the first axis of the input tensor."""
        self.__input_dim1 = input_dim1

    @property
    def input_dim2(self) -> int:
        """int: Get the size of the second axis of the input tensor."""
        return self.__input_dim2

    @input_dim2.setter
    def input_dim2(self, input_dim2: int):
        """int: Set the size of the second axis of the input tensor."""
        self.__input_dim2 = input_dim2

    @property
    def input_dim3(self) -> int:
        """int: Get the size of the third axis of the input tensor."""
        return self.__input_dim3

    @input_dim3.setter
    def input_dim3(self, input_dim3: int):
        """int: Set the size of the third axis of the input tensor."""
        self.__input_dim3 = input_dim3

    def __repr__(self):
        return (f'InputLayer({self.name}, {self.activation_function}, {self.input_dim1}, '
                f'{self.input_dim2}, {self.input_dim3})')


class LinearLayer(GeneralLayer):
    """Represents a densely-connected NN layer that applies a linear transformation to the input data.
 
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        in_features (int): It represents the size of each input sample.
        out_features (int): It represents the size of each output sample.

    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
        in_features (int): It represents the size of each input sample.
        out_features (int): It represents the size of each output sample.
    """
    def __init__(self, name: str, activation_function: str, in_features: int, out_features: int):
        super().__init__(name, activation_function)
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
        return (f'LinearLayer({self.name}, {self.activation_function}, ' 
                f'{self.in_features}, {self.out_features})')

class FlattenLayer(GeneralLayer):
    """Represents a layer that flattens a contiguous range of dims into a tensor..
 
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
    """
    def __init__(self, name: str, activation_function: str):
        super().__init__(name, activation_function)

    def __repr__(self):
        return f'FlattenLayer({self.name}, {self.activation_function})'

        
class EmbeddingLayer(GeneralLayer):
    """Represents a layer that learns dense vector representations of the input data.
 
    Args:
        name (str): The name of the layer.
        activation_function (str): The type of the activation function.
        num_embeddings (int): The size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        
    Attributes:
        name (str): Inherited from Layer. It represents the name of the layer.
        activation_function (str): Inherited from Layer. It represents the type of the activation
                                   function.
        num_embeddings (int): The size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
    """
    def __init__(self, name: str, activation_function: str, num_embeddings: int, embedding_dim: int):
        super().__init__(name, activation_function)
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
        return (f'EmbeddingLayer({self.name}, {self.activation_function}, ' 
                f'{self.num_embeddings}, {self.embedding_dim})')


class Feature:
    """A feature is a measurable property or characteristic of an object used to represent and describe
       it within a dataset.
 
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
    """A label is a value assigned to an observation, representing the target variable for prediction.
 
    Args:
        col_name (str): The name of the column containing the labels.
        label_name (str): The name of a label in the dataset. If the prediction task 
                          is regression, it can be omitted.
        
    Attributes:
        col_name (str): The name of the column containing the labels.
        label_name (str): The name of a label in the dataset. If the prediction task 
                          is regression, it can be omitted.
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
    """Image represents features designed for handling data with spatial characteristics, typically 
       including attributes such as height and width.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        channels (int): The number of channels in the image.
        
    Attributes:
        height (int): The height of the image.
        width (int): The width of the image.
        channels (int): The number of channels in the image.
    """
    def __init__(self, height: int, width: int, channels: int):
        self.height: int = height
        self.width: int = width
        self.channels: int = channels

    @property
    def height(self) -> int:
        """int: Get the height of the image."""
        return self.__height

    @height.setter
    def height(self, height: int):
        """int: Set the height of the image."""
        self.__height = height

    @property
    def width(self) -> int:
        """int: Get the width of the image."""
        return self.__width

    @width.setter
    def width(self, width: int):
        """int: Set the width of the image."""
        self.__width = width

    @property
    def channels(self) -> int:
        """int: Get the channels of the image."""
        return self.__channels

    @channels.setter
    def channels(self, channels: int):
        """int: Set the channels of the image."""
        self.__channels = channels

    def __repr__(self):
        return f'Image({self.height}, {self.width}, {self.channels})' 

class Structured(Feature):
    """Represents features organized in a systematic manner, typically with well-defined columns and rows, 
       often found in tabular datasets.

    Args:
        name (str): The name of the feature.
        
    Attributes:
        name (str): Inherited from Feature. It represents the name of the feature.
    """
    def __init__(self, name: str):
        super().__init__(name)
        
    def __repr__(self):
        return f'Structured({self.name})'



class Dataset:
    """Represents the collection of data instances used for training or evaluation, where each 
       instance comprises features and corresponding labels.

    Args:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing the dataset.
        task_type (str): The type of prediction task associated with the dataset.
        has_images (bool): Indicates whether the dataset contains images.
        features (set[Feature]): The set of features in the dataset.
        labels (set[Label]): The set of labels in the dataset.
        
    Attributes:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing the dataset.
        task_type (str): The type of prediction task associated with the dataset.
        has_images (bool): Indicates whether the dataset contains images.
        features (set[Feature]): The set of features in the dataset.
        labels (set[Label]): The set of labels in the dataset.
    """
    def __init__(self, name: str, path_data: str, task_type: str, has_images: bool, 
                 features: set[Feature] = None, labels: set[Label] = None):
        self.name: str = name
        self.path_data: str = path_data
        self.task_type: str = task_type
        self.has_images: bool = has_images
        self.features: set[Feature] = features
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
        """str: Get the type of prediction task associated with the dataset."""
        return self.__task_type

    @task_type.setter
    def task_type(self, task_type: str):
        """str: Set the type of prediction task associated with the dataset.
        
        Raises:
            ValueError: If task_type is not one of the allowed 
            options: 'binary', 'multi_class', and 'regression'
        """

        if task_type not in ['binary', 'multi_class', 'regression', None]:
            raise ValueError("Invalid value of task_type")

        self.__task_type = task_type

    @property
    def has_image(self) -> bool:
        """bool: Get whether the dataset contains images."""
        return self.__has_image


    @has_image.setter
    def has_image(self, has_image: bool):
        """bool: Set whether the dataset contains images."""
        self.__has_image = has_image


    @property
    def features(self) -> set[Label]:
        """set[Feature]: Get the set of features."""
        return self.__features
    
    @features.setter
    def features(self, features: set[Label]):
        """set[Feature]: Set the set of features."""
        self.__features = features
        
        
    @property
    def labels(self) -> set[Label]:
        """set[Label]: Get the set of labels."""
        return self.__labels
    
    @labels.setter
    def labels(self, labels: set[Label]):
        """set[Label]: Set the set of labels."""
        self.__labels = labels

    def load_data(self):
        pass

    def split_features_labels(self):
        pass

    def __repr__(self):
        return (f'Dataset({self.name}, {self.path_data}, {self.task_type}, '
                f'{self.has_images}, {self.labels})')  
    
class TrainingDataset(Dataset):
    """Represents a Dataset used for training the NN model.

    Args:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing the dataset.
        task_type (str): The type of prediction task associated with the dataset.
        has_images (bool): Indicates whether the dataset contains images.
        features (set[Feature]): The set of features in the dataset.
        labels (set[Label]): The set of labels in the dataset.
        
    Attributes:
        name (str): Inherited from Dataset. It represents the name of the dataset.
        path_data (str): Inherited from Dataset. It represents the file path containing the dataset.
        task_type (str): Inherited from Dataset. It represents the type of prediction task 
                        associated with the dataset.
        has_images (bool): Inherited from Dataset. It indicates whether the dataset contains images.
        features (set[Feature]): Inherited from Dataset. It represents the set of features in the dataset.
        labels (set[Label]): Inherited from Dataset. It represents the set of labels in the dataset.
    """

    def __init__(self, name: str, path_data: str, task_type: str, has_images: bool, 
                 features: set[Feature] = None, labels: set[Label] = None):
        super().__init__(name, path_data, task_type, has_images, features, labels)
    
    def __repr__(self):
        return (f'TrainingDataset({self.name}, {self.path_data}, {self.task_type}, '
                f'{self.has_images}, {self.features}, {self.labels})')

class TestDataset(Dataset):
    """Represents a Dataset used for evaluating the performance of the NN model.

    Args:
        name (str): The name of the dataset.
        path_data (str): The file path or directory location containing the dataset.
        
    Attributes:
        name (str): Inherited from Dataset. It represents the name of the dataset.
        path_data (str): Inherited from Dataset. It represents the file path containing the dataset.
    """

    def __init__(self, name: str, path_data: str):
        super().__init__(name, path_data, None, None, None, None)
    
    def __repr__(self):
        return f'TestDataset({self.name}, {self.path_data})' 


class Parameters:
    """Represents a collection of parameters essential for training and evaluating neural networks.

    Args:
        batch_size (int): The number of data samples processed in each iteration during training 
                          or inference in a neural network.
        epochs (int): It refers to the number of complete passes through the entire dataset during 
                      the training, with each epoch consisting of one iteration through all data samples.
        learning_rate (float): The step size used to update the model parameters during optimization.
        optimizer (str): The method or algorithm used to adjust the model parameters iteratively 
                         during training to minimize the loss function and improve model performance.
        loss_function (str): The method used to calculate the difference between predicted and actual 
                             values, guiding the model towards better predictions.
        metrics List[str]: Quantitative measures used to evaluate the performance of NN models.
        regularization (str): The regularization method used in training the model. It can be either
                              'l1' or 'l2' or None.
        weight_decay (float): It represents the strength of L2 regularization applied to the model's 
                              parameters during optimization.
        
    Attributes:
        batch_size (int): The number of data samples processed in each iteration during training 
                          or inference in a neural network.
        epochs (int): It refers to the number of complete passes through the entire dataset during 
                      the training, with each epoch consisting of one iteration through all data samples.
        learning_rate (float): The step size used to update the model parameters during optimization.
        optimizer (str): The method or algorithm used to adjust the model parameters iteratively 
                         during training to minimize the loss function and improve model performance.
        loss_function (str): The method used to calculate the difference between predicted and actual 
                             values, guiding the model towards better predictions.
        metrics List[str]: Quantitative measures used to evaluate the performance of NN models.
        regularization (str): The regularization method used in training the model. It can be either
                              'l1' or 'l2' or None.
        weight_decay (float): It represents the strength of L2 regularization applied to the model's 
                              parameters during optimization.
    """
    def __init__(self, batch_size: int, epochs: int, learning_rate: float, optimizer: str, 
                 loss_function: str, metrics: List[str], regularization: str = None,
                 weight_decay: float = 0):
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.optimizer: str = optimizer
        self.loss_function: str = loss_function
        self.metrics: List[str] = metrics
        self.regularization = regularization
        self.weight_decay: float = weight_decay

    @property
    def batch_size(self) -> int:
        """int: Get the number of data samples processed in each iteration during training or inference 
           in a neural network."""
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """int: Set the number of data samples processed in each iteration during training or inference 
           in a neural network."""
        self.__batch_size = batch_size

    @property
    def epochs(self) -> int:
        """int: Get the number of complete passes through the entire dataset during the training."""
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs: int):
        """int: Set the number of complete passes through the entire dataset during the training."""
        self.__epochs = epochs

    @property
    def learning_rate(self) -> float:
        """float: Get the step size used to update the model parameters during optimization."""
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        """float: Set step size used to update the model parameters during optimization."""
        self.__learning_rate = learning_rate

    @property
    def optimizer(self) -> str:
        """str: Get the algorithm used to adjust the model parameters iteratively during 
           training to minimize the loss function."""
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: str):
        """str: Set the algorithm used to adjust the model parameters iteratively during 
           training to minimize the loss function.
        
        Raises:
            ValueError: If optimizer is not one of the allowed 
            options: 'sgd', 'adam', 'adamW' and 'adagrad'
        """

        if optimizer not in ['sgd', 'adam', 'adamW', 'adagrad']:
            raise ValueError("Invalid value of optimizer")
        self.__optimizer = optimizer

    @property
    def loss_function(self) -> str:
        """str: Get the method used to calculate the difference between predicted and actual 
           values, guiding the model towards better predictions."""
        return self.__loss_function


    @loss_function.setter
    def loss_function(self, loss_function: str):
        """str: Set the method used to calculate the difference between predicted and actual 
           values, guiding the model towards better predictions.
        
        Raises:
            ValueError: If loss_function is not one of the allowed 
            options: 'crossentropy', 'binary_crossentropy' and 'mse'
        """

        if loss_function not in ['crossentropy', 'binary_crossentropy', 'mse']:
            raise ValueError("Invalid value of loss_function")
        self.__loss_function = loss_function

    @property
    def metrics(self) -> List[str]:
        """List[str]: Get the measures for evaluating the performance of the model."""
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """List[str]: Set the measures for evaluating the performance of the model.
        
        Raises:
            ValueError: If metrics is not one of the allowed 
            options: accuracy', 'precision', 'recall', 'f1-score' and 'mae'
        """
        if isinstance(metrics, list) and all(isinstance(metric, str) for metric in metrics):
            if all(metric in ['accuracy', 'precision', 'recall', 'f1-score', 'mae'] for metric in metrics):
                self.__metrics = metrics
            else:
                invalid_metrics = [metric for metric in metrics if metric not in ['accuracy', 'precision', 
                                                                                  'recall', 'f1-score', 'mae']]
                raise ValueError(f"Invalid metric(s) provided: {invalid_metrics}")
        else:
            raise ValueError("'metrics' must be a list of strings.")

    @property
    def regularization(self) -> str:
        """str: Get the method used to calculate the difference between predicted and actual 
           values, guiding the model towards better predictions."""
        return self.__regularization


    @regularization.setter
    def regularization(self, regularization: str):
        """str: Set the regularization method used in training the model.
        
        Raises:
            ValueError: If regularization is not one of the allowed 
            options: 'l1', 'l2', and None
        """

        if regularization not in ['l1', 'l2', None]:
            raise ValueError("Invalid value of regularization")
        self.__regularization = regularization

    @property
    def weight_decay(self) -> float:
        """float: Get the strength of L2 regularization applied during optimization."""
        return self.__weight_decay

    @weight_decay.setter
    def weight_decay(self, weight_decay: float):
        """float: Set the strength of L2 regularization applied during optimization."""
        self.__weight_decay = weight_decay


    def __repr__(self):
        return (f'Parameters({self.batch_size}, {self.epochs}, {self.learning_rate}, ' 
                f'{self.optimizer}, {self.loss_function}, {self.metrics}, '
                f'{self.regularization}, {self.weight_decay})')

class NN(BehavioralImplementation):
    """It is a subclass of the NamedElement class and comprises the fundamental properties 
        and behaviors of a neural network model.

    Args:
        name (str): The name of the neural network model.
        layers (List[Layer]): The list of layers composing the neural network.
        parameters (Parameters): The parameters related to the NN training and evaluation. 
        
    Attributes:
        name (str): The name of the neural network model.
        layers (List[Layer]): The list of layers composing the neural network.
        parameters (Parameters): The parameters related to the NN training and evaluation. 
    """
    def __init__(self, name: str, layers: List[Layer] = [], parameters: Parameters = None):
        super().__init__(name)
        self.layers: List[Layer] = layers
        self.parameters: Parameters = parameters

    @property
    def layers(self) -> List[Layer]:
        return self.__layers
            
    @layers.setter
    def layers(self, layers: List[Layer]):
        if isinstance(layers, list):
            if all(isinstance(layer, Layer) for layer in layers):
                self.__layers = layers
            else:
                raise ValueError("All the elements of the list must be of type Layer")
        else:
            raise ValueError("'layers' must be a list.")
        
    @property
    def parameters(self) -> Parameters:
        """Parameters: Get the parameters related to the NN training and evaluation."""
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters: Parameters):
        """Parameters: Set the parameters related to the NN training and evaluation."""
        self.__parameters = parameters

    def add_parameters(self, parameters: Parameters):
        self.parameters = parameters

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def compile(self):
        pass

    def summary(self):
        pass

    def train(self, training_data: TrainingDataset):
        pass

    def predict(self, feature: Feature):
        pass
    
    def evaluate(self, test_dataset: TestDataset):
        pass
