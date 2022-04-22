import numpy as np


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

