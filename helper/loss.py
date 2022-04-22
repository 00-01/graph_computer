def compute_loss(self, prediction, label):
    term_1 = -1*label*np.log(prediction)
    term_2 = (1-label)*(np.log(1-prediction))

    loss = np.sum(term_1-term_2)
    return loss

