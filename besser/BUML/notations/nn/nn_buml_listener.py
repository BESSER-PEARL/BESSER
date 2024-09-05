from .NNParser import NNParser
from .NNListener import NNListener

class NeuralNetworkASTListener(NNListener):

    def __init__(self, output):
        self.output = output
        self.__nn_name = ""
        self.__sub_nn = {}
        self.__layers = []
        self.__tensor_op = []

    def enterNeuralNetwork(self, ctx: NNParser.NeuralNetworkContext):
        text = "from besser.BUML.metamodel.nn import *\n\n"
        self.__nn_name = ctx.ID().getText()
        text += "# Neural network definition\n"
        text += "nn_model: NN = NN(name=\"" + self.__nn_name + "\")\n\n"
        self.output.write(text)

    def enterParameters(self, ctx: NNParser.ParametersContext):
        text = "# Parameters of the NN\n"
        text += "parameters = Parameters(batch_size=" + ctx.INT(0).getText() \
        + ", epochs=" + ctx.INT(1).getText() + ", learning_rate=" + ctx.DOUBLE(0).getText() \
        + ", optimiser=" + ctx.STRING().getText() + ", metrics=" + ctx.strList().getText() \
        + ", loss_function=\"" + ctx.lossFunction().getText() + "\", weight_decay=" \
        + ctx.DOUBLE(1).getText() + ")\n\n"
        self.output.write(text)

    def enterLayer(self, ctx: NNParser.LayerContext):
        n_layer = ctx.ID().getText()
        if isinstance(ctx.parentCtx, NNParser.Sub_nnContext):
            sub_nn = ctx.parentCtx.ID().getText()
            if sub_nn in self.__sub_nn:
                self.__sub_nn[sub_nn].append(n_layer)
            else:
                self.__sub_nn[sub_nn] = [n_layer]
        self.__layers.append(n_layer)

    def exitLayer(self, ctx: NNParser.LayerContext):
        self.output.write(")\n")

    def enterConvolutional(self, ctx: NNParser.ConvolutionalContext):
        l_name = ctx.parentCtx.parentCtx.ID().getText()
        text = l_name + " = " + ctx.conv_type.text + "(name=\"" + l_name +"\""
        text += ", in_channels=" + ctx.INT(0).getText()
        text += ", out_channels=" + ctx.INT(1).getText()
        self.output.write(text)

    def enterLayerParams(self, ctx: NNParser.LayerParamsContext):
        text = ""
        if ctx.activityFuncType():
            if ctx.activityFuncType().getText() != "None":
                text += ", actv_func=\"" + ctx.activityFuncType().getText() + "\""
            else:
                text += ", actv_func=None"
        if ctx.STRING():
            text += ", name_layer_input=\"" + ctx.STRING().getText() + "\""
        if ctx.BOOL():
            text += ", input_reused=\"" + ctx.BOOL().getText() + "\""
        self.output.write(text)

    def enterCnnParams(self, ctx: NNParser.CnnParamsContext):
        text=""
        if ctx.kernel:
            text = ", kernel_dim=" + ctx.kernel.getText()
        if ctx.stride:
            text += ", stride_dim=" + ctx.stride.getText()
        if ctx.paddingType():
            text += ", paddingType=\"" + ctx.paddingType().getText() + "\""
        if ctx.INT():
            text += ", padding_amount=" + ctx.INT().getText()
        self.output.write(text)

    def enterPooling(self, ctx: NNParser.PoolingContext):
        l_name = ctx.parentCtx.parentCtx.ID().getText()
        text = l_name + " = PoolingLayer(name=\"" + l_name +"\""
        text += ", pooling_type=\"" + ctx.poolingType().getText() + "\""
        text += ", dimension=\"" + ctx.dimensionality().getText() + "\""
        if ctx.intList():
            text += ", output_dim=" + ctx.intList().getText()
        self.output.write(text)

    def enterFlatten(self, ctx: NNParser.FlattenContext):
        l_name = ctx.parentCtx.parentCtx.ID().getText()
        text = l_name + " = FlattenLayer(name=\"" + l_name +"\""
        if ctx.INT(0):
            text += ", start_dim=" + ctx.INT(0).getText()
        if ctx.INT(1):
            text += ", end_dim=" + ctx.INT(1).getText()
        self.output.write(text)

    def enterDropout(self, ctx: NNParser.DropoutContext):
        l_name = ctx.parentCtx.parentCtx.ID().getText()
        text = l_name + " = DropoutLayer(name=\"" + l_name +"\", rate=" + ctx.DOUBLE().getText()
        self.output.write(text)

    def enterLinear(self, ctx: NNParser.LinearContext):
        l_name = ctx.parentCtx.parentCtx.ID().getText()
        text = l_name + " = LinearLayer(name=\"" + l_name +"\""
        text += ", in_features=" + ctx.INT(0).getText()
        text += ", out_features=" + ctx.INT(1).getText()
        self.output.write(text)

    def enterSub_nn(self, ctx: NNParser.Sub_nnContext):
        l_name = ctx.ID().getText()
        text = l_name + ": NN = NN(name=\"" + l_name +"\")\n"
        self.output.write(text)

    def enterModules(self, ctx: NNParser.ModulesContext):
        text = "\n# Adding layers, sub-layers, and parameters to NNs\n"
        for sub_nn, sub_layers in self.__sub_nn.items():
            for sub_layer in sub_layers:
                text += sub_nn + ".add_layer(" + sub_layer +')\n'
        for module in ctx.ID():
            if str(module) in self.__layers:
                text += "nn_model.add_layer(" + str(module) +')\n'
            elif str(module) in self.__sub_nn:
                text += "nn_model.add_sub_nn(" + str(module) +')\n'
            elif str(module) in self.__tensor_op:
                text += "nn_model.add_tensor_op(" + str(module) +')\n'
        text += "nn_model.add_parameters(parameters)\n"
        self.output.write(text)

    def enterTrainingDataset(self, ctx: NNParser.TrainingDatasetContext):
        text = "\n# Training dataset definition\n"
        tab = "\n" + ("\t" * 7)
        text += "train_data = TrainingDataset(name=\"" + ctx.ID().getText() + "\","\
                + tab + "path_data=r" + ctx.STRING().getText() + ","\
                + tab + "task_type=\"" + ctx.taskType().getText() + "\","\
                + tab + "input_format=\"" + ctx.inputFormat().getText() + "\","\
                + tab + "image=Image(shape=" + ctx.intList().getText() + ")," + tab + "labels={"
        for label in ctx.label():
            text += "Label(col_name=" + label.STRING(0).getText() + ", label_name=" + label.STRING(1).getText() + "), "
        text = text[:-2]
        text += "})\n"
        self.output.write(text)

    def enterTestDataset(self, ctx: NNParser.TestDatasetContext):
        text = "\n# Test dataset definition\n"
        tab = "\n" + ("\t" * 6)
        text += "test_data = TestDataset(name=\"" + ctx.ID().getText() + "\","\
                + tab + "path_data=r" + ctx.STRING().getText() + ")\n"
        self.output.write(text)

    def enterTensorOp(self, ctx: NNParser.TensorOpContext):
        t_name = ctx.ID().getText()
        text = t_name + " = TensorOp(name=\"" + t_name + "\", type=\"" + ctx.tensorOpType().getText() + "\""
        if ctx.INT():
            text += ", concatenate_dim=" + ctx.INT().getText()
        if ctx.intStrList():
            text += ", layers_of_tensors=" + ctx.intStrList.getText()
        if ctx.reshape:
            text += ", reshape_dim=" + ctx.reshape.getText()
        if ctx.transpose:
            text += ", transpose_dim=" + ctx.transpose.getText()
        if ctx.permute:
            text += ", permute_dim=" + ctx.permute.getText()
        if ctx.after_ativ:
            text += ", after_activ_func=" + ctx.after_ativ.getText()
        if ctx.input_ref:
            text += ", input_reused=" + ctx.input_ref.getText()
        text += ")\n"
        self.output.write(text)
        self.__tensor_op.append(t_name)
