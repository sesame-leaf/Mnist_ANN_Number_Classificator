import random
import math
import time
import sys

from torchvision import datasets, transforms

from MLP import *
    
def main():
    
    # load MNIST dataset
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    print("MNIST dataset loaded")
    
    
    # transform dataset to list
    train_data_list = list() # list[list[float], int]
    for image, label in train_dataset:
        image_flatten = image.view(-1).tolist()
        train_data_list.append((image_flatten, label))
    print("train data transformed")
        
    test_data_list = list()
    for image, label in test_dataset:
        image_flatten = image.view(-1).tolist()
        test_data_list.append((image_flatten, label))
    print("test data transformed")
    
    
    # initialize weights and biases
    layer_1_weight_init = list([[random.gauss(0, math.sqrt(2 / 784)) for _ in range(28**2)] for _ in range(128)])
    layer_1_bias_init = list([0.0 for _ in range(128)])
    
    layer_2_weight_init = list([[random.gauss(0, math.sqrt(1 / 128)) for _ in range(128)] for _ in range(10)])
    layer_2_bias_init = list([0.0 for _ in range(10)])
    
    
    # initialize ANN model
    model = MLP(list([layer_1_weight_init, layer_2_weight_init]), list([layer_1_bias_init, layer_2_bias_init]))
    
    # train model
    epoch_num = 5
    for epoch in range(1, epoch_num+1):
        print(f"epoch {epoch} start")
        start_time = time.time()
        
        total_loss = 0
        for i in range(len(train_data_list)):
            model.forward_propagation(train_data_list[i][0])
            total_loss += model.calculate_loss(train_data_list[i][1])
            model.backward_propagation()
            model.update_weights_biases()
            
            if i % 100 == 0:
                sys.stdout.write(f"\rprogress: {round(i / len(train_data_list) * 100, 2)} %  duration: {round(time.time()-start_time, 2)} second")
                sys.stdout.flush()
        
        end_time = time.time()
        print(f"\nepoch {epoch} end\nduration: {round((end_time-start_time) / 60), 2} minutes\ntotal loss: {total_loss}\naverage loss: {total_loss / len(train_data_list)}\n")
    
    # test model
    total_loss = 0
    correct_num = 0
    for data, label in test_data_list:
        model.forward_propagation(data)
        total_loss += model.calculate_loss(label)
        
        if model.output_datas.index(max(model.output_datas)) == label:
            correct_num += 1
        
    print(f"test\ntotal loss: {total_loss}\ncorrect rate: {round(correct_num / len(test_data_list), 2) * 100} %")
    
main()