import random
import math
import time
import sys

from torchvision import datasets, transforms

from MLP import *
import parameter_data as param

def main():
    
    # load MNIST dataset
    transform = transforms.ToTensor()
    
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    print("MNIST dataset loaded")
    
    
    # transform dataset to list
    test_data_list = list()
    for image, label in test_dataset:
        image_flatten = image.view(-1).tolist()
        test_data_list.append((image_flatten, label))
    print("test data transformed")
    
    
    # initialize weights and biases
    layer_1_weights = param.layer_1_weights
    layer_1_biases = param.layer_1_biases

    layer_2_weights = param.layer_2_weights
    layer_2_biases = param.layer_2_biases
    
    # initialize ANN model
    model = MLP(list([layer_1_weights, layer_2_weights]), list([layer_1_biases, layer_2_biases]))
    
    # test model
    total_loss = 0
    correct_num = 0
    loop_count = -1
    start_time = time.time()
    for input_d, label in test_data_list:
        loop_count += 1

        model.forward_propagation(input_d)
        total_loss += model.calculate_loss(label)
        
        if model.output_datas.index(max(model.output_datas)) == label:
            correct_num += 1
        
        if loop_count % 100 == 0:
                sys.stdout.write(f"\rprogress: {round(loop_count / len(test_data_list) * 100, 2)} %  duration: {round(time.time()-start_time, 2)} second")
                sys.stdout.flush()
        
    print(f"test\ntotal loss: {total_loss}\ncorrect rate: {round(correct_num / len(test_data_list), 2) * 100} %")
    
main()
