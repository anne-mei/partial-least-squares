import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def generate_sample_data():
    P = np.array([[0.5765, 0.2856, 0.1614],
                [0.3660, 0.0458, 0.9060],
                [0.5889, 0.4645, 0.9942],
                [0.3572, 0.3450, 0.7396],
                [0.4036, 0.6851, 0.2262]])
    c = np.array([[0.7451, 0.4928, 0.7320, 0.4738, 0.4652],
                  np.random.randn(1, 5)[0],
                  np.random.randn(1, 5)[0],
                  np.random.randn(1, 5)[0],
                  np.random.randn(1, 5)[0]])
    train_size = 500
    test_size = 400
    x_train = np.zeros((train_size, 5))
    y_train = np.zeros((train_size, np.shape(c)[0]))
    x_test = np.zeros((test_size, 5))
    y_test = np.zeros((test_size, np.shape(c)[0]))

    for i in range(train_size+test_size):
        x = np.zeros((5, 1))
        y = np.zeros((1, 1))
        #generate random data for e, t, v
        t = 2 * np.random.randn(3, 1)
        e = 0.5 * np.random.randn(5, 1)
        # v = 0.5 * np.random.randn(np.shape(c)[0], 1)
        v = np.zeros((np.shape(c)[0],1)) #array of zeros to account for the added columns in c
        x = P.dot(t) + e
        y = c.dot(x) + v
        if i < train_size:
            x_train[i] = np.transpose(x)
            y_train[i] = np.transpose(y)
        else:
            x_test[i-train_size] = np.transpose(x)
            y_test[i-train_size] = np.transpose(y)
    return x_train, y_train, x_test, y_test


def autoscale_train(x, y):
    # scale X to 0 mean and unit variance
    x_m = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    for i in range(x.shape[1]):
        x[:,i] = x[:,i] - x_m[i]
        x[:,i] = x[:,i] / x_std[i]

    # scale Y to 0 mean and unit variance
    y_m = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    for i in range(y.shape[1]):
        y[:, i] = y[:, i] - y_m[i]
        y[:, i] = y[:, i] / y_std[i]

    return x_m, x_std, x, y_m, y_std, y


#normalizes test data with m and std of training set
def autoscale_test(data, m, std):
    # scale test data to 0 mean and unit variance of training data
    for i in range(data.shape[1]):
        data[:, i] = data[:, i] - m[i]
        data[:, i] = data[:, i] / std[i]
    return data


def pls_nipals(x, y, imax):
    x_size = np.shape(x)
    y_size = np.shape(y)

    # final latent variable matrices
    U = np.zeros((x_size[0], imax))
    W = np.zeros((x_size[1], imax))
    T = np.zeros((x_size[0], imax))
    Q = np.zeros((y_size[1], imax))
    B = np.zeros((1, imax))
    P = np.zeros((x_size[1], imax))

    threshold = 0.001
    i = 0

    while i < imax:
        #outer modelling
        u_prev = y[:,0].reshape(-1, 1)
        t_prev = u_prev
        max_iter = 100
        iter = 0
        error = 10
        while linalg.norm(error) > threshold and iter < max_iter:
            w = np.dot(np.transpose(x),u_prev) / linalg.norm(np.dot(np.transpose(x), u_prev))
            t = np.dot(x, w)
            q = np.dot(np.transpose(y), t) / linalg.norm(np.dot(np.transpose(y), t))
            u = np.dot(y, q)
            error = t_prev - t
            t_prev = t
            u_prev = u
            iter = iter+1

        #inner modelling
        b_matrix = np.transpose(u).dot(t) / (np.transpose(t).dot(t))
        b = b_matrix[0][0]

        #residual deflation
        p = np.dot(np.transpose(x), t) / np.dot(np.transpose(t), t)
        x = x - np.dot(t, np.transpose(p))
        y = y - (b * np.dot(t, np.transpose(q)))
        U[:, i] = u[:, 0]
        W[:, i] = w[:, 0]
        T[:, i] = t[:, 0]
        Q[:, i] = q[:, 0]
        B[:, i] = b
        P[:, i] = p[:, 0]
        i = i+1
    return x, y, T, P, Q, B, W


def prediction(x, y, T, P, Q, B, W, imax):
    B_diag = np.diag(B[0])
    R = W.dot(np.linalg.inv(np.transpose(P).dot(W)))
    x_prediction = x.dot(R).dot(np.transpose(P))
    y_prediction = x.dot(R).dot(B_diag).dot(np.transpose(Q))
    print('x1_mse: ', mean_squared_error(x[:, 0], x_prediction[:, 0]))
    print('x2_mse: ', mean_squared_error(x[:, 1], x_prediction[:, 1]))
    print('x3_mse: ', mean_squared_error(x[:, 2], x_prediction[:, 2]))
    print('x4_mse: ', mean_squared_error(x[:, 3], x_prediction[:, 3]))
    print('x5_mse: ', mean_squared_error(x[:, 4], x_prediction[:, 4]))
    print('y_mse: ', mean_squared_error(y[:, 0], y_prediction[:, 0]))
    return x_prediction, y_prediction


def plot_graphs(x, x_prediction, y, y_prediction, blocking):    
    fig = plt.figure(figsize=(20,10))
    fig.suptitle('PLS Results')

    # Plot X data
    for i in range(5):
        plt.subplot(231 + i)
        plt.plot(x[:, i], c='b', label='Original Data')
        plt.plot(x_prediction[:, i], c='r', label='Prediction Data')
        plt.xlabel('Data Point')
        plt.ylabel('X')
        plt.title('X Original vs. Prediction Data of Feature ' + str(i + 1))
        plt.legend(loc='upper left')

    # Plot Y data
    plt.subplot(236)
    plt.plot(y[:, 0], c='b', label='Y Original Data')
    plt.plot(y_prediction[:, 0], c='r', label='Y Prediction Data')
    plt.xlabel('Data Point')
    plt.ylabel('Y')
    plt.title('Y Original vs. Prediction Data')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("sample_result.png")
    plt.show(block=blocking)
    


x_train, y_train, x_test, y_test = generate_sample_data()
imax = 5

#normalize data
x_train_m, x_train_std, x_train_scaled, y_train_m, y_train_std, y_train_scaled = autoscale_train(x_train, y_train)
x_test_scaled = autoscale_test(x_test, x_train_m, x_train_std)
y_test_scaled = autoscale_test(y_test, y_train_m, y_train_std)

#training data
x_train, y_train, T, P, Q, B, W = pls_nipals(x_train_scaled, y_train_scaled, imax)
x_test_prediction, y_test_prediction = prediction(x_test_scaled, y_test_scaled, T, P, Q, B, W, imax)

plot_graphs(x_test_scaled, x_test_prediction, y_test_scaled, y_test_prediction,"False")