mlp = MLP()
x_train, y_train, x_test, y_test = mnist.load()

# w = h = int(math.sqrt(x_train.shape[1]))
w, h = 28, 28

# x_train, y_train = mlp.reshaper(x_train, x_test, (w,h))

x_train, y_train = mlp.batch_norm(x_train, y_train)

mlp.fit(x_train, y_train)