"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import random

import minitorch

import time

os = []
sides = [0, 0]
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)


    def forward(self, x):
        middle = [h.relu() for h in self.layer1.forward(x)]
        #print('middle:', middle)
        end = [h.relu() for h in self.layer2.forward(middle)]
        #print('end:', end)
        #print('ret val:', self.layer3.forward(end)[0].sigmoid())
        return self.layer3.forward(end)[0].sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        #print('-'*10)
        output = [0.0 for x in range(len(self.bias))]
        """
        print('len(bias):', len(self.bias))
        print('len(weights):', len(self.weights))
        print('inputs:', inputs)
        """
        #print('in:', inputs)
        #print('bias:', self.bias)
        #print('inputs:', inputs)
        #print('weights:', self.weights)
        #print('bias:', self.bias)
        #print('shape:', [len(x) for x in inputs])
        for i in range(len(self.bias)):
            my_bias = self.bias[i].value
            for j in range(len(self.weights)):
                output[i] += self.weights[j][i].value * inputs[j]
            output[i] += self.bias[i].value        
        #os.append(tuple([x.data for x in output]))
        #print('out:', output)
        return output
    
def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []

        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                #print('='*50)
                x_1, x_2 = data.X[i]
                y = data.y[i]
                #print('y:', y)
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))
                #print('out is:', out)
                os.append(out.data)
                if out.data > 0.5:
                    sides[0] += 1
                else:
                    sides[1] +=1
                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data
            
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            #if epoch % 10 == 0 or epoch == max_epochs or epoch in [1, 2, 3, 4, 5]:
            log_fn(epoch, total_loss, correct, losses)

if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    ScalarTrain(HIDDEN).train(data, RATE)


def rounddown(x):
    if x < 1e-10:
        return 0
    else:
        return x

def count(mi):
    md = {}
    for item in mi:
        if item in list(md.keys()):
            md[item] += 1
        else:
            md.update({item:1})
    return md
    
def main():
    PTS = 50
    HIDDEN = 2
    RATE = 0.01
    data = minitorch.datasets["Simple"](PTS)
    ScalarTrain(HIDDEN).train(data, RATE)

    """
    temp = [rounddown(y) for y in os]
    md = count(temp)

    print('counts:', md)
    x = [rounddown(y) for y in list(set(os))]
    x = list(set(x))
    print('unique outputs:', x)
    print('unique outputs:', len(x))
    print('total outputs:', len(os))
    print('cases: [>0.5, <0.5]', sides)
    """
