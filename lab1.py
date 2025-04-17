import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm

numbers = [
    8.0, 12.5, 15.4, 6.9, 11.4, 7.2, 10.5, 11.5, 17.7, 13.6, 15.1, 13.4, 17.9, 18.6,
    9.8, 12.6, 14.9, 7.3, 16.5, 15.5, 12.9, 11.0, 16.8, 18.4, 12.8, 11.4, 13.5, 16.2,
    14.3, 12.1, 12.2, 18.1, 10.9, 7.9, 17.9, 18.6, 10.5, 13.7, 10.3, 17.2, 13.5,
    17.7, 6.7, 17.1, 16.4, 7.1, 16.9, 14.2, 11.3, 15.2, 15.8, 12.3, 9.9, 15.6,
    18.9, 14.2, 8.2, 11.5, 18.6, 19.0
]

"""1. Составить вариационный ряд"""
variation_series = sorted(numbers)
print(variation_series)

"""2. Построить интервальный статистический ряд"""
# нахождение размаха выборки
X_min = min(numbers)
X_max = max(numbers)
R = X_max - X_min

# Нахождение числа интервалов
N = len(numbers)
k = round(1 + 3.322 * math.log10(N))

# Нахождение длины частичного интервала
h = round(R / k, 1)

# Границы частичных интервалов
bins = np.arange(X_min, X_max + h, h)

# Группировка по частичтным интервалам
hist, bin_edges = np.histogram(numbers, bins)

for i in range(len(hist)):
    print(f"{bin_edges[i]:.1f} - {bin_edges[i + 1]:.1f}: {hist[i]}")

"""3. По сгруппированным данным построить полигон и гистограмму"""

# Расчет относительных частот
relative_freq = hist / N

# Середины интервалов для полигона
mid_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

# Построение графиков
plt.figure(figsize=(12, 5))

# Полигон относительных частот
plt.subplot(1, 2, 1)
plt.plot(mid_bins, relative_freq, 'bo-')
plt.title('Полигон относительных частот')
plt.xlabel('Середины интервалов')
plt.ylabel('Относительная частота')

# Гистограмма относительных частот
plt.subplot(1, 2, 2)
plt.bar(bin_edges[:-1], relative_freq, width=h, align='edge',
        edgecolor='black', alpha=0.7)
plt.title('Гистограмма относительных частот')
plt.xlabel('Интервалы')
plt.ylabel('Относительная частота')
plt.xticks(bin_edges)
plt.grid(True)
plt.tight_layout()
plt.show()

"""4. График эмпирической функции распределения"""
# Сортировка данных и расчет накопленных частот
x = np.sort(numbers)
y = np.arange(1, len(x) + 1) / len(x)

plt.figure(figsize=(10, 5))
plt.step(x, y, where='post')
plt.title('Эмпирическая функция распределения')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True)
plt.show()

"""5. Числовые характеристики"""
# Выборочное среднее
mean = np.mean(numbers)

# Исправленная дисперсия
var = np.var(numbers, ddof=1)

# Мода (для интервального ряда)
mode_interval = bin_edges[np.argmax(hist)]

# Медиана
median = np.median(numbers)

# Асимметрия и эксцесс
skewness = skew(numbers, bias=False)
kurtosis_val = kurtosis(numbers, bias=False)

print(f'''
Числовые характеристики:
Выборочное среднее: {mean:.4f}
Исправленная дисперсия: {var:.4f}
Мода: {mode_interval:.1f}
Медиана: {median:.4f}
Асимметрия: {skewness:.4f}
Эксцесс: {kurtosis_val:.4f}
''')

"""6-8. Анализ распределения и теоретические кривые"""
# Гипотеза о нормальном распределении
print("Гипотеза: нормальное распределение N(μ,σ²)")

# Оценки параметров
mu, sigma = mean, np.sqrt(var)
print(f"Параметры: μ={mu:.2f}, σ={sigma:.2f}")

# Теоретические кривые
plt.figure(figsize=(12, 5))

# Гистограмма с теоретической плотностью
plt.subplot(1, 2, 1)
x_range = np.linspace(X_min, X_max, 100)
plt.bar(bin_edges[:-1], relative_freq, width=h, align='edge',
        edgecolor='black', alpha=0.7)
plt.plot(x_range, norm.pdf(x_range, mu, sigma) * h, 'r-', lw=2)
plt.title('Теоретическая плотность и гистограмма')

# ЭФР с теоретической функцией
plt.subplot(1, 2, 2)
plt.step(x, y, where='post')
plt.plot(x_range, norm.cdf(x_range, mu, sigma), 'r-', lw=2)
plt.title('Теоретическая ФР и ЭФР')
plt.tight_layout()
plt.show()

"""9. Проверка правила трёх сигм"""
three_sigma_low = mean - 3 * sigma
three_sigma_high = mean + 3 * sigma
within_3sigma = len([x for x in numbers if three_sigma_low <= x <= three_sigma_high])
percentage = within_3sigma / N * 100

print(f'''
Правило трёх сигм:
Границы: [{three_sigma_low:.2f}, {three_sigma_high:.2f}]
Данных в пределах: {within_3sigma}/{N} ({percentage:.2f}%)
Теоретическое значение: 99.73%
''')

"""10. Критерий согласия Пирсона"""
from scipy.stats import chi2

# Теоретические частоты для нормального распределения
theoretical_probs = [norm.cdf(bin_edges[i + 1], mu, sigma) -
                     norm.cdf(bin_edges[i], mu, sigma)
                     for i in range(len(bin_edges) - 1)]

theoretical_freq = np.array(theoretical_probs) * N

# Объединение интервалов с частотой <5
observed = hist.copy()
expected = theoretical_freq.copy()

# Расчет критерия
chi2_stat = np.sum((observed - expected) ** 2 / expected)
dof = len(observed) - 3  # степени свободы (k - r - 1)
p_value = 1 - chi2.cdf(chi2_stat, dof)

print(f'''
Критерий Пирсона:
Хи-квадрат = {chi2_stat:.3f}
Степени свободы = {dof}
p-value = {p_value:.4f}
Гипотеза {"не отвергается" if p_value > 0.05 else "отвергается"}
''')

"""11. Доверительные интервалы"""
from scipy.stats import t, chi2

gamma = 0.95
alpha = 1 - gamma
n = N

# Для среднего
t_crit = t.ppf(1 - alpha / 2, df=n - 1)
mean_ci_low = mean - t_crit * sigma / np.sqrt(n)
mean_ci_high = mean + t_crit * sigma / np.sqrt(n)

# Для СКО
chi2_low = chi2.ppf(1 - alpha / 2, df=n - 1)
chi2_high = chi2.ppf(alpha / 2, df=n - 1)
sigma_ci_low = np.sqrt((n - 1) * var / chi2_low)
sigma_ci_high = np.sqrt((n - 1) * var / chi2_high)

print(f'''
Доверительные интервалы (γ={gamma}):
Среднее: [{mean_ci_low:.3f}, {mean_ci_high:.3f}]
СКО: [{sigma_ci_low:.3f}, {sigma_ci_high:.3f}]
''')

"""12. Критерий Колмогорова"""
from scipy.stats import kstest

# Эмпирическая F(x) vs теоретическая F(x)
D, p_kolmogorov = kstest(numbers, 'norm', args=(mu, sigma))

print(f'''
Критерий Колмогорова:
D = {D:.4f}
Критическое значение (α=0.05): {1.36 / np.sqrt(N):.4f}
p-value = {p_kolmogorov:.4f}
Гипотеза {"не отвергается" if p_kolmogorov > 0.05 else "отвергается"}
''')


print(f'''
8. Параметры распределения: μ = {mu:.2f}, σ = {sigma:.2f}

9. Правило трёх сигм: 
   Границы: [{three_sigma_low:.2f}, {three_sigma_high:.2f}]
   Попало: {within_3sigma} из {N} ({percentage:.1f}%)

10. Критерий Пирсона: 
    χ² = {chi2_stat:.2f}, df = {dof}, крит.значение = {chi2.ppf(0.95, dof):.2f}, p = {p_value:.4f}

11. Доверительные интервалы (γ=0.95):
    Среднее: [{mean_ci_low:.2f}, {mean_ci_high:.2f}]
    СКО: [{sigma_ci_low:.2f}, {sigma_ci_high:.2f}]

12. Критерий Колмогорова: 
    D = {D:.4f}, крит.значение = {1.36/np.sqrt(N):.4f}, p = {p_kolmogorov:.4f}
''')