import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randint(100)

model: LinearRegression = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Интерцепт (β0):", model.intercept_)
print("Коэффициент (β1):", model.coef_[0])

mse = mean_squared_error(y, y_pred)
print("MSE:", mse)

plt.scatter(X, y, color='blue', label='Исходные данные')
plt.plot(X, y_pred, color='red', label='Линия регрессии')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Линейная регрессия: исходные данные и предсказания")
plt.legend()

plt.savefig("regression_plot.png")

# Попытка показать график в зависимости от среды выполнения
try:
    from PIL import Image
    img = Image.open("regression_plot.png")
    img.show()
except Exception as e:
    print("График сохранён как regression_plot.png, но не может быть открыт автоматически:", e)