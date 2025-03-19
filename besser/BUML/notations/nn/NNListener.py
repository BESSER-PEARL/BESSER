# Generated from ./NN.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .NNParser import NNParser
else:
    from NNParser import NNParser

# This class defines a complete listener for a parse tree produced by NNParser.
class NNListener(ParseTreeListener):

    # Enter a parse tree produced by NNParser#neuralNetwork.
    def enterNeuralNetwork(self, ctx:NNParser.NeuralNetworkContext):
        pass

    # Exit a parse tree produced by NNParser#neuralNetwork.
    def exitNeuralNetwork(self, ctx:NNParser.NeuralNetworkContext):
        pass


    # Enter a parse tree produced by NNParser#parameters.
    def enterParameters(self, ctx:NNParser.ParametersContext):
        pass

    # Exit a parse tree produced by NNParser#parameters.
    def exitParameters(self, ctx:NNParser.ParametersContext):
        pass


    # Enter a parse tree produced by NNParser#layer.
    def enterLayer(self, ctx:NNParser.LayerContext):
        pass

    # Exit a parse tree produced by NNParser#layer.
    def exitLayer(self, ctx:NNParser.LayerContext):
        pass


    # Enter a parse tree produced by NNParser#layerParams.
    def enterLayerParams(self, ctx:NNParser.LayerParamsContext):
        pass

    # Exit a parse tree produced by NNParser#layerParams.
    def exitLayerParams(self, ctx:NNParser.LayerParamsContext):
        pass


    # Enter a parse tree produced by NNParser#generalLayer.
    def enterGeneralLayer(self, ctx:NNParser.GeneralLayerContext):
        pass

    # Exit a parse tree produced by NNParser#generalLayer.
    def exitGeneralLayer(self, ctx:NNParser.GeneralLayerContext):
        pass


    # Enter a parse tree produced by NNParser#linear.
    def enterLinear(self, ctx:NNParser.LinearContext):
        pass

    # Exit a parse tree produced by NNParser#linear.
    def exitLinear(self, ctx:NNParser.LinearContext):
        pass


    # Enter a parse tree produced by NNParser#flatten.
    def enterFlatten(self, ctx:NNParser.FlattenContext):
        pass

    # Exit a parse tree produced by NNParser#flatten.
    def exitFlatten(self, ctx:NNParser.FlattenContext):
        pass


    # Enter a parse tree produced by NNParser#embedding.
    def enterEmbedding(self, ctx:NNParser.EmbeddingContext):
        pass

    # Exit a parse tree produced by NNParser#embedding.
    def exitEmbedding(self, ctx:NNParser.EmbeddingContext):
        pass


    # Enter a parse tree produced by NNParser#rnn.
    def enterRnn(self, ctx:NNParser.RnnContext):
        pass

    # Exit a parse tree produced by NNParser#rnn.
    def exitRnn(self, ctx:NNParser.RnnContext):
        pass


    # Enter a parse tree produced by NNParser#cnn.
    def enterCnn(self, ctx:NNParser.CnnContext):
        pass

    # Exit a parse tree produced by NNParser#cnn.
    def exitCnn(self, ctx:NNParser.CnnContext):
        pass


    # Enter a parse tree produced by NNParser#cnnParams.
    def enterCnnParams(self, ctx:NNParser.CnnParamsContext):
        pass

    # Exit a parse tree produced by NNParser#cnnParams.
    def exitCnnParams(self, ctx:NNParser.CnnParamsContext):
        pass


    # Enter a parse tree produced by NNParser#convolutional.
    def enterConvolutional(self, ctx:NNParser.ConvolutionalContext):
        pass

    # Exit a parse tree produced by NNParser#convolutional.
    def exitConvolutional(self, ctx:NNParser.ConvolutionalContext):
        pass


    # Enter a parse tree produced by NNParser#pooling.
    def enterPooling(self, ctx:NNParser.PoolingContext):
        pass

    # Exit a parse tree produced by NNParser#pooling.
    def exitPooling(self, ctx:NNParser.PoolingContext):
        pass


    # Enter a parse tree produced by NNParser#layerModifier.
    def enterLayerModifier(self, ctx:NNParser.LayerModifierContext):
        pass

    # Exit a parse tree produced by NNParser#layerModifier.
    def exitLayerModifier(self, ctx:NNParser.LayerModifierContext):
        pass


    # Enter a parse tree produced by NNParser#dropout.
    def enterDropout(self, ctx:NNParser.DropoutContext):
        pass

    # Exit a parse tree produced by NNParser#dropout.
    def exitDropout(self, ctx:NNParser.DropoutContext):
        pass


    # Enter a parse tree produced by NNParser#normalisation.
    def enterNormalisation(self, ctx:NNParser.NormalisationContext):
        pass

    # Exit a parse tree produced by NNParser#normalisation.
    def exitNormalisation(self, ctx:NNParser.NormalisationContext):
        pass


    # Enter a parse tree produced by NNParser#lossFunction.
    def enterLossFunction(self, ctx:NNParser.LossFunctionContext):
        pass

    # Exit a parse tree produced by NNParser#lossFunction.
    def exitLossFunction(self, ctx:NNParser.LossFunctionContext):
        pass


    # Enter a parse tree produced by NNParser#sub_nn.
    def enterSub_nn(self, ctx:NNParser.Sub_nnContext):
        pass

    # Exit a parse tree produced by NNParser#sub_nn.
    def exitSub_nn(self, ctx:NNParser.Sub_nnContext):
        pass


    # Enter a parse tree produced by NNParser#trainingDataset.
    def enterTrainingDataset(self, ctx:NNParser.TrainingDatasetContext):
        pass

    # Exit a parse tree produced by NNParser#trainingDataset.
    def exitTrainingDataset(self, ctx:NNParser.TrainingDatasetContext):
        pass


    # Enter a parse tree produced by NNParser#testDataset.
    def enterTestDataset(self, ctx:NNParser.TestDatasetContext):
        pass

    # Exit a parse tree produced by NNParser#testDataset.
    def exitTestDataset(self, ctx:NNParser.TestDatasetContext):
        pass


    # Enter a parse tree produced by NNParser#label.
    def enterLabel(self, ctx:NNParser.LabelContext):
        pass

    # Exit a parse tree produced by NNParser#label.
    def exitLabel(self, ctx:NNParser.LabelContext):
        pass


    # Enter a parse tree produced by NNParser#tensorOp.
    def enterTensorOp(self, ctx:NNParser.TensorOpContext):
        pass

    # Exit a parse tree produced by NNParser#tensorOp.
    def exitTensorOp(self, ctx:NNParser.TensorOpContext):
        pass


    # Enter a parse tree produced by NNParser#modules.
    def enterModules(self, ctx:NNParser.ModulesContext):
        pass

    # Exit a parse tree produced by NNParser#modules.
    def exitModules(self, ctx:NNParser.ModulesContext):
        pass


    # Enter a parse tree produced by NNParser#intList.
    def enterIntList(self, ctx:NNParser.IntListContext):
        pass

    # Exit a parse tree produced by NNParser#intList.
    def exitIntList(self, ctx:NNParser.IntListContext):
        pass


    # Enter a parse tree produced by NNParser#strList.
    def enterStrList(self, ctx:NNParser.StrListContext):
        pass

    # Exit a parse tree produced by NNParser#strList.
    def exitStrList(self, ctx:NNParser.StrListContext):
        pass


    # Enter a parse tree produced by NNParser#intStrList.
    def enterIntStrList(self, ctx:NNParser.IntStrListContext):
        pass

    # Exit a parse tree produced by NNParser#intStrList.
    def exitIntStrList(self, ctx:NNParser.IntStrListContext):
        pass


    # Enter a parse tree produced by NNParser#activityFuncType.
    def enterActivityFuncType(self, ctx:NNParser.ActivityFuncTypeContext):
        pass

    # Exit a parse tree produced by NNParser#activityFuncType.
    def exitActivityFuncType(self, ctx:NNParser.ActivityFuncTypeContext):
        pass


    # Enter a parse tree produced by NNParser#returnTypeRRN.
    def enterReturnTypeRRN(self, ctx:NNParser.ReturnTypeRRNContext):
        pass

    # Exit a parse tree produced by NNParser#returnTypeRRN.
    def exitReturnTypeRRN(self, ctx:NNParser.ReturnTypeRRNContext):
        pass


    # Enter a parse tree produced by NNParser#tensorOpType.
    def enterTensorOpType(self, ctx:NNParser.TensorOpTypeContext):
        pass

    # Exit a parse tree produced by NNParser#tensorOpType.
    def exitTensorOpType(self, ctx:NNParser.TensorOpTypeContext):
        pass


    # Enter a parse tree produced by NNParser#taskType.
    def enterTaskType(self, ctx:NNParser.TaskTypeContext):
        pass

    # Exit a parse tree produced by NNParser#taskType.
    def exitTaskType(self, ctx:NNParser.TaskTypeContext):
        pass


    # Enter a parse tree produced by NNParser#inputFormat.
    def enterInputFormat(self, ctx:NNParser.InputFormatContext):
        pass

    # Exit a parse tree produced by NNParser#inputFormat.
    def exitInputFormat(self, ctx:NNParser.InputFormatContext):
        pass


    # Enter a parse tree produced by NNParser#paddingType.
    def enterPaddingType(self, ctx:NNParser.PaddingTypeContext):
        pass

    # Exit a parse tree produced by NNParser#paddingType.
    def exitPaddingType(self, ctx:NNParser.PaddingTypeContext):
        pass


    # Enter a parse tree produced by NNParser#poolingType.
    def enterPoolingType(self, ctx:NNParser.PoolingTypeContext):
        pass

    # Exit a parse tree produced by NNParser#poolingType.
    def exitPoolingType(self, ctx:NNParser.PoolingTypeContext):
        pass


    # Enter a parse tree produced by NNParser#dimensionality.
    def enterDimensionality(self, ctx:NNParser.DimensionalityContext):
        pass

    # Exit a parse tree produced by NNParser#dimensionality.
    def exitDimensionality(self, ctx:NNParser.DimensionalityContext):
        pass



del NNParser