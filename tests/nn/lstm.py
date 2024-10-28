from besser.BUML.metamodel.nn import NN, LinearLayer, DropoutLayer, EmbeddingLayer, LSTMLayer
from besser.generators.pytorch.pytorch_code_generator import PytorchGenerator
from besser.generators.tf.tf_code_generator import TFGenerator

"""Paper: LSTM and GRU neural network performance comparison study
   https://github.com/g9g99g9g/basic/blob/master/ANN/LSTM_vs_GRU/yelp_reviews_emotion_prediction.py"""

lstm_model: NN = NN(name="lstm")
lstm_model.add_layer(EmbeddingLayer(name="l1", actv_func=None, num_embeddings=10000, embedding_dim=326))
lstm_model.add_layer(LSTMLayer(name="l2", actv_func=None, return_type="full", input_size=326, hidden_size=40, bidirectional=True, dropout=0.5))
lstm_model.add_layer(DropoutLayer(name="l3", rate=0.2))
lstm_model.add_layer(LSTMLayer(name="l4", actv_func=None, return_type="last", input_size=2*40, hidden_size=40, dropout=0.2))
lstm_model.add_layer(LinearLayer(name="l5", actv_func="relu", in_features=40, out_features=40))
lstm_model.add_layer(LinearLayer(name="l6", actv_func="softmax", in_features=40, out_features=2))


pytorch_model = PytorchGenerator(model=lstm_model, output_dir="output/lstm")
pytorch_model.generate() 
tf_model = TFGenerator(model=lstm_model, output_dir="output/lstm")
tf_model.generate() 