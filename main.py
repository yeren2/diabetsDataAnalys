import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

train=pd.read_csv("diabetes.csv")

X_features = train.drop("Outcome", axis=1)
y_train=train["Outcome"]

mu = np.mean(X_features, axis=0)
sigma = np.std(X_features, axis=0)
X_train_scaled = (X_features - mu) / sigma

X_train = X_train_scaled.values
y_train=y_train.values

a=X_train.shape[1]
initial_w=np.zeros(a)
initial_b=0.0

def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g


def compute_cost(X,y,w,b,lambda_=0):
    m,n=X.shape
    loss_sum=0
    for i in range(m):
        z_wb=np.dot(w,X[i]) + b
        f=sigmoid(z_wb)
        loss=(-y[i]*np.log(f))-(1-y[i])*np.log(1-f)
        loss_sum+=loss
    loss_sum=loss_sum/m
    return loss_sum

def compute_gradient(X, y, w, b,lambda_=0):
    m,n=X.shape
    dj_dw=np.zeros(w.shape)
    dj_db=0.
    for i in range(m):
        z_wb=0
        for j in range(n):
            z_wbij = w[j] * X[i][j]
            z_wb += z_wbij
        z_wb+=b
        f=sigmoid(z_wb)
        dj_db_i=f-y[i]
        dj_db+=dj_db_i
        for j in range(n):
            dj_dw_ij = (f - y[i]) * X[i][j]
            dj_dw[j] += dj_dw_ij
        dj_dw=dj_dw/m
        dj_db=dj_db/m
    return dj_db,dj_dw
print(compute_gradient(X_train,y_train,initial_w,initial_b))

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, iteration, lambda_):
    m=len(X)
    j_total=[]
    w_total=[]

    for i in range(iteration):
        dj_db,dj_dw=gradient_function(X,y,w_in,b_in,lambda_)

        w_in=w_in-alpha*dj_dw
        b_in=b_in-alpha*dj_db

        if i<10000:
            cost=cost_function(X,y,w_in,b_in,lambda_)
            j_total.append(cost)
        if i % math.ceil(iteration / 10) == 0 or i == (iteration - 1):
            w_total.append(w_in)
            print(f"Iteration-> {i}: Cost ->{float(j_total[-1])}")
    return w_in,b_in,j_total,w_total


def predict(X, w, b):
    z = np.dot(X, w) + b
    f_wb = 1 / (1 + np.exp(-z))
    return f_wb >= 0.5

iterations = 10000
alpha = 0.3

w_final, b_final, j_history, w_history = gradient_descent(
    X_train, y_train, initial_w, initial_b,
    compute_cost, compute_gradient, alpha, iterations, lambda_=0
)

X_a = X_train[:, [1, 5]]
y_a = y_train

tahminler = predict(X_train, w_final, b_final)
basari_orani = np.mean(tahminler == y_train) * 100

print(f"Modelin Doğruluk Oranı: %{basari_orani:.2f}")

positive = y_a == 1
negative = y_a == 0
plt.figure(figsize=(10,6))

plt.scatter(X_a[positive, 0], X_a[positive, 1], marker='x', c='red', label='Diyabetli (1)')

plt.scatter(X_a[negative, 0], X_a[negative, 1], marker='o', c='blue', edgecolors='k', label='Sağlıklı (0)')

plt.xlabel("Glikoz")
plt.ylabel("BMI")
plt.legend()
plt.title("Diyabet Veri Seti Dağılımı")
plt.show()
