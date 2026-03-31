import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def create_function(expr):
    return lambda x, y: eval(expr, {'x': x, 'y': y,
                                    'cos': np.cos, 'sin': np.sin, 'tan': np.tan,
                                    'exp': np.exp, 'log': np.log, 'log10': np.log10,
                                    'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e})


def euler_simple(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        x[i + 1] = x[i] + h
    return x, y


def euler_modified(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h / 2 * (k1 + k2)
        x[i + 1] = x[i] + h
    return x, y


def rk2(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        y[i + 1] = y[i] + h * k2
        x[i + 1] = x[i] + h
    return x, y


def rk4(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i + 1] = x[i] + h
    return x, y


def adams(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0

    for i in range(3):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i + 1] = x[i] + h

    for i in range(3, n):
        y[i + 1] = y[i] + h / 24 * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) +
                                    37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3], y[i - 3]))
        x[i + 1] = x[i] + h
    return x, y


def milne(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0

    for i in range(3):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i + 1] = x[i] + h

    for i in range(3, n):
        y_pred = y[i - 3] + 4 * h / 3 * (2 * f(x[i - 2], y[i - 2]) - f(x[i - 1], y[i - 1]) + 2 * f(x[i], y[i]))
        y[i + 1] = y[i - 1] + h / 3 * (f(x[i - 1], y[i - 1]) + 4 * f(x[i], y[i]) + f(x[i] + h, y_pred))
        x[i + 1] = x[i] + h
    return x, y


def get_exact_solution_func(expr_str, x0_val, y0_val):
    x_s = sp.Symbol('x', real=True, positive=True)
    y_s = sp.Function('y')(x_s)
    expr_sp = sp.sympify(expr_str)
    ode = sp.Eq(y_s.diff(x_s), expr_sp.subs({'x': x_s, 'y': y_s}))
    sol = sp.dsolve(ode, y_s)
    c1 = sp.Symbol('C1')
    eq_c = sp.Eq(sol.rhs.subs(x_s, x0_val), y0_val)
    c1_val = sp.solve(eq_c, c1)[0]
    final_expr = sp.re(sol.rhs.subs(c1, c1_val))
    return sp.lambdify(x_s, final_expr, 'numpy')


expr = input("f(x, y) = ")
x0 = float(input("x0 = "))
x_end = float(input("x_end = "))
y0 = float(input("y0 = "))
h = float(input("h = "))
n = int(round((x_end - x0) / h))
print(f"n = {n}")

f = create_function(expr)
f_exact = get_exact_solution_func(expr, x0, y0)

methods = [
    ("Эйлер (простой)", euler_simple),
    ("Эйлер (модифицированный)", euler_modified),
    ("Рунге-Кутта 2 порядка", rk2),
    ("Рунге-Кутта 4 порядка", rk4),
    ("Адамс", adams),
    ("Милн", milne)
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

results = []
x_fine = np.linspace(x0, x_end, 200)
y_exact_fine = f_exact(x_fine)

for idx, (name, method) in enumerate(methods):
    x, y = method(f, x0, y0, h, n)
    y_exact_nodes = f_exact(x)
    errors = np.abs(y - y_exact_nodes)
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    results.append((name, y[-1], mean_error, max_error))

    axes[idx].plot(x, y, 'o-', markersize=2, label='Численное решение', alpha=0.7)
    axes[idx].plot(x_fine, y_exact_fine, 'k--', label='Точное решение', alpha=0.8)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True)
    axes[idx].set_title(f'{name}\nСр. ошибка: {mean_error},\n Макс. ошибка: {max_error}')

for idx in range(len(methods), 6):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

print("=====================================================")
print(f"Точное решение y({x_end}) = {f_exact(x_end)}")
for name, y_val, mean_err, max_err in results:
    print(f"{name}: y({x_end}) = {y_val}, Ср. ошибка = {mean_err}, Макс. ошибка = {max_err}")

"""
(y**2 - 1)/x
0.1
1
0
0.1
"""


