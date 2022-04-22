import numpy as np
import mnist
import math


class MLP(object):
    def __init__(self, learning_rate=0.001, momentum=0.001, epoch=50, batch_size=50):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epoch = epoch
        self.batch_size = batch_size


    def init_weights(self, input, layer_1, layer_2, output, batch_size):
        w1 = np.random.randn(layer_1, input+1)
        w2 = np.random.randn(layer_2, layer_1+1)
        w3 = np.random.randn(output, layer_2+1)

        return w1, w2, w3


    def one_hot_enc(self, y, num_labels=10):
        one_hot = np.zeros((num_labels, y.shape[0]), dtype=np.float32)

        for i, val in enumerate(y):
            one_hot[val, i] = 1.0

        return one_hot


    def add_bias_unit(self, layer, orientation):
        if orientation == 'row':
            updated_layer = np.ones((layer.shape[0]+1, layer.shape[1]))
            updated_layer[1:, :] = layer
        elif orientation == 'col':
            updated_layer = np.ones((layer.shape[0], layer.shape[1]+1))
            updated_layer[:, 1:] = layer

        return updated_layer


    def forward_pass(self, input):
        a1 = self.add_bias_unit(input, orientation='col')

        z2 = np.matmul(self.w1, a1.transpose(1, 0))
        a2 = 1/(1+np.exp(-z2))
        a2 = self.add_bias_unit(a2, orientation='row')

        z3 = np.matmul(self.w2, a2)
        a3 = 1/(1+np.exp(-z3))
        a3 = self.add_bias_unit(a3, orientation='row')

        z4 = np.matmul(self.w3, a3)
        a4 = 1/(1+np.exp(-z4))

        return a1, z2, a2, z3, a3, z4, a4


    def compute_loss(self, prediction, label):
        term_1 = -1*label*np.log(prediction)
        term_2 = (1-label)*(np.log(1-prediction))

        loss = np.sum(term_1-term_2)
        return loss


    def back_pass(self, outputs, label):
        a1, z2, a2, z3, a3, z4, a4 = outputs

        delta_4 = a4-label
        sig_z3 = np.array(1/(1+np.exp(-z3)))
        delta_3 = np.matmul(self.w3[:, 1:].transpose(), delta_4)*sig_z3*(1-sig_z3)

        sig_z2 = np.array(1/(1+np.exp(-z2)))
        delta_2 = np.matmul(self.w2[:, 1:], delta_3)*(sig_z2)*(1-(sig_z2))

        grad_w1 = np.matmul(delta_2, a1)
        grad_w2 = np.matmul(delta_3, a2.transpose())
        grad_w3 = np.matmul(delta_4, a3.transpose())

        return grad_w1, grad_w2, grad_w3


    def norm(self, X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0)-X.min(axis=0)
        denom[denom == 0] = 1
        return x_min+nom/denom


    def batch_norm(self, X, y):
        X_ = []
        y_ = []

        itr = int(len(y)/self.batch_size)+1
        for j in range(1, itr):
            rng = j*self.batch_size
            X_.append(X[rng-self.batch_size:rng, :])
            y_.append(y[rng-self.batch_size:rng])

        X, y = np.array(X_), np.array(y_)
        X = self.norm(X, 0, 1)

        return X, y


    def reshaper(self, x, x1, shape):
        x = np.reshape(x, (x.shape[0],shape[0],shape[1]))
        x1 = np.reshape(x1, (x1.shape[0],shape[0],shape[1]))

        return x, x1


    def predict(self, a4):
        prediction = np.argmax(a4, axis=0)
        return prediction


    def fit(self, X, y):
        input = len(X[0, 0, :])  # returns the flattened image size (28*28 = 784)

        layer_1, layer_2, output = 100, 100, 10
        self.w1, self.w2, self.w3 = self.init_weights(input, layer_1, layer_2, output, self.batch_size)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)
        delta_w3_prev = np.zeros(self.w3.shape)

        train_losses = []
        train_acc = []

        for i in range(self.epoch):
            for j, (input, label) in enumerate(zip(X, y)):
                one_hot_label = self.one_hot_enc(label, num_labels=10)

                a1, z2, a2, z3, a3, z4, a4 = self.forward_pass(input)
                loss = self.compute_loss(a4, one_hot_label)
                grad1, grad2, grad3 = self.back_pass([a1, z2, a2, z3, a3, z4, a4], one_hot_label)

                delta_w1, delta_w2, delta_w3 = self.learning_rate*grad1, self.learning_rate*grad2, self.learning_rate*grad3

                self.w1 -= delta_w1+delta_w1_prev*self.momentum
                self.w2 -= delta_w2+delta_w2_prev*self.momentum
                self.w3 -= delta_w3+delta_w3_prev*self.momentum

                delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3

                train_losses.append(loss)
                predictions = self.predict(a4)

                wrong = np.where(predictions != label, np.matrix([1.]), np.matrix([0.]))

                accuracy = 1-np.sum(wrong)/self.batch_size

                train_acc.append(accuracy)

            print('epoch ', i, 'training accuracy %.2f'%np.mean(np.matrix(train_acc)).item())


if __name__ == '__main__':
    mlp = MLP()
    x_train, y_train, x_test, y_test = mnist.load()

    # w = h = int(math.sqrt(x_train.shape[1]))
    w, h = 28, 28

    # x_train, y_train = mlp.reshaper(x_train, x_test, (w,h))

    x_train, y_train = mlp.batch_norm(x_train, y_train)

    mlp.fit(x_train, y_train)