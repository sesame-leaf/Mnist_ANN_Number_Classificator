from math import exp, log

# data: float
def ReLU(data) -> float:
    return max(0, data)

# data: float
def identity(data) -> float:
    return data

# datas: list[float]
def softmax(datas) -> list[float]:
    max_data = max(datas)
    exps = [exp(x - max_data) for x in datas]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

# answer: int, datas: list[float]
def cross_entropy(answer, datas) -> float:
    return -log(datas[answer] + 1e-15)


class Perceptron:
    # weights: list[float], bias: float, activation_func(float) -> float
    def __init__(self, weights, bias, activation_func) -> None:
        self.weights = weights
        self.bias = bias
        self.__activation_func = activation_func
    
    # datas: list[float]
    def __call__(self, input_datas) -> float:
        self.z = sum([weight * data for weight, data in zip(self.weights, input_datas)]) + self.bias
        return self.__activation_func(self.z)


class Layer:
    # weights: list[list[float]], biases: list[float]
    def __init__(self, weights, biases, activation_func) -> None:
        self.perceptrons = [Perceptron(weight, bias, activation_func) for weight, bias in zip(weights, biases)]
    
    # input_datas: list[float]
    def __call__(self, input_datas) -> list[float]:
        return list([perceptron(input_datas) for perceptron in self.perceptrons])


class MLP:
    # weights: list[list[list[float]]], biases: list[list[float]]
    def __init__(self, weights, biases) -> None:
        self.layer1 = Layer(weights[0], biases[0], ReLU)
        self.layer2 = Layer(weights[1], biases[1], identity)
        self.learning_rate = 0.01
    
    # input_datas: list[float]
    def forward_propagation(self, input_datas) -> None:
        self.input_datas = input_datas
        self.hidden_datas = self.layer1(self.input_datas)
        self.output_datas = softmax(self.layer2(self.hidden_datas))
    
    # answer: int
    def calculate_loss(self, answer) -> float:
        self.answer = answer
        self.loss_value = cross_entropy(answer, self.output_datas)
        
        return self.loss_value
    
    def backward_propagation(self) -> None:
        self.__dL_dw2 = [[(self.output_datas[i] - (1 if i == self.answer else 0)) * self.hidden_datas[j] for j in range(128)] for i in range(10)]
        self.__dL_db2 = [(self.output_datas[i] - (1 if i == self.answer else 0)) for i in range(10)]
        
        self.__dL_db1 = [sum([(self.layer2.perceptrons[i].weights[k]) * self.__dL_db2[i] for i in range(10)]) * (1 if self.layer1.perceptrons[k].z > 0 else 0) for k in range(128)]
        self.__dL_dw1 = [[self.__dL_db1[k] * self.input_datas[j] for j in range(28**2)] for k in range(128)]
    
    def update_weights_biases(self) -> None:
        for i in range(10):
            for j in range(128):
                self.layer2.perceptrons[i].weights[j] -= self.learning_rate * self.__dL_dw2[i][j]
            
            self.layer2.perceptrons[i].bias -= self.learning_rate * self.__dL_db2[i]
        
        for i in range(128):
            for j in range(28**2):
                self.layer1.perceptrons[i].weights[j] -= self.learning_rate * self.__dL_dw1[i][j]
            
            self.layer1.perceptrons[i].bias -= self.learning_rate * self.__dL_db1[i]


def main():
    pass

if __name__ == "__main__":
    main()
    