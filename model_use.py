from MLP import *
import parameter_data as param
from image_convert import image_convert

def main():
    
    # load data
    data = image_convert("pictures\\three2.png")
    
    # initialize weights and biases
    layer_1_weights = param.layer_1_weights
    layer_1_biases = param.layer_1_biases

    layer_2_weights = param.layer_2_weights
    layer_2_biases = param.layer_2_biases
    
    # initialize ANN model
    model = MLP(list([layer_1_weights, layer_2_weights]), list([layer_1_biases, layer_2_biases]))
    
    # predict number
    model.forward_propagation(data)
    
    print(*enumerate([round(x, 2) for x in model.get_predict()]))
    
if __name__ == "__main__":
    main()
