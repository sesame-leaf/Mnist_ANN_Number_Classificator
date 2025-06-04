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
    layer_1_weights = list()
    layer_1_biases = list()

    layer_2_weights = list()
    layer_2_biases = list()
    
    # initialize ANN model
    model = MLP(list([layer_1_weights, layer_1_biases]), list([layer_2_weights, layer_2_biases]))
    
    # train model
    # epoch_num = 5
    # for epoch in range(1, epoch_num+1):
    #     print(f"epoch {epoch} start")
    #     start_time = time.time()
        
    #     total_loss = 0
    #     for loop_count in range(len(train_data_list)):
    #         model.forward_propagation(train_data_list[loop_count][0])
    #         total_loss += model.calculate_loss(train_data_list[loop_count][1])
    #         model.backward_propagation()
    #         model.update_weights_biases()
            
    #         if loop_count % 100 == 0:
    #             sys.stdout.write(f"\rprogress: {round(loop_count / len(train_data_list) * 100, 2)} %  duration: {round(time.time()-start_time, 2)} second")
    #             sys.stdout.flush()
        
    #     end_time = time.time()
    #     print(f"\nepoch {epoch} end\nduration: {round((end_time-start_time) / 60), 2} minutes\ntotal loss: {total_loss}\naverage loss: {total_loss / len(train_data_list)}\n")
    
    # test model
    total_loss = 0
    correct_num = 0
    loop_count = -1
    start_time = time.time()
    for data, label in test_data_list:
        loop_count += 1
        model.forward_propagation(data)
        total_loss += model.calculate_loss(label)
        
        if model.output_datas.index(max(model.output_datas)) == label:
            correct_num += 1
        
        if loop_count % 100 == 0:
                sys.stdout.write(f"\rprogress: {round(loop_count / len(train_data_list) * 100, 2)} %  duration: {round(time.time()-start_time, 2)} second")
                sys.stdout.flush()
        
    print(f"test\ntotal loss: {total_loss}\ncorrect rate: {round(correct_num / len(test_data_list), 2) * 100} %")
    
main()