import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def embed_series(series, window_size):
    """
    Создает траекторное пространство ряда путем вложения с использованием оконного метода.
    """
    # Получаем размер вложенной матрицы
    n = len(series)

    # Создаем матрицу для траекторного пространства ряда
    embedded = np.zeros((n - window_size + 1, window_size))

    # Заполняем матрицу траекторного пространства
    for i in range(n - window_size + 1):
        embedded[i] = series[i:i + window_size]

    # Возвращаем созданное траекторное пространство ряда
    return embedded


def decompose(embedded_series):
    """
    Разложение вложенного временного ряда на сингулярные компоненты.
    """
    # Вычисляем ковариационную матрицу на основе вложенного временного ряда
    cov_matrix = np.cov(embedded_series.T)

    # Вычисляем собственные значения и собственные векторы ковариационной матрицы
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Сортируем собственные значения по убыванию и переупорядочиваем собственные векторы
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Возвращаем отсортированные собственные значения и собственные векторы
    return eigenvalues, eigenvectors


def forecast(time_series, eigenvalues, eigenvectors, r, M):
    """
    Выполняет прогноз на M значений вперед, используя первые r сингулярных компонент.
    """
    trend_matrix = np.dot(eigenvectors[:, :r], np.diag(eigenvalues[:r]))
    forecast = np.zeros(M)
    for i in range(M):
        if i < r:
            # Для начальных значений, когда не хватает данных для построения тренда
            forecast[i] = time_series[-(r - i)]
        else:
            # Используем метод наименьших квадратов для построения тренда
            X = np.ones((r, 2))
            X[:, 1] = np.arange(1, r + 1)
            y = time_series[-r:]
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            forecast[i] = np.dot([1, r + i], beta)
    return forecast


def read_file(filename, column_name, N, k):
    """
    Читает из файла Excel N чисел начиная с k-го числа заданного столбца и возвращает их как список.
    """
    # Чтение данных из файла Excel
    df = pd.read_excel(filename)

    # Проверяем наличие столбца с заданным именем
    if column_name not in df.columns:
        return "Столбец с именем '{}' не найден.".format(column_name)

    # Считываем N чисел начиная с k-го числа столбца
    column_values = df.iloc[k-1:k-1+N][column_name].tolist()

    # Преобразование значений в тип float и фильтрация пустых значений
    column_values = [float(val) for val in column_values if pd.notnull(val)]

    return column_values


def get_value_prediction(time_series, L, r, M):
    # Построение траекторного пространства
    embedded_series = embed_series(time_series, L)

    # Разложение временного ряда на компоненты
    eigenvalues, eigenvectors = decompose(embedded_series)

    # Прогноз на M значений вперед
    forecast_values = forecast(time_series, eigenvalues, eigenvectors, r, M)

    # Выводим результаты
    print("Прогноз на следующие {} значений:".format(M))
    return forecast_values


def plot_graph(x_values, y_values, plot_name, plot_color, plot_linewidth):
    """
    Рисует график на основе переданных списков значений x и y.
    """
    plt.plot(x_values, y_values, color=plot_color, linewidth=plot_linewidth)
    plt.xlabel("Значения временного ряда")
    plt.ylabel("Номер значения")
    plt.title(plot_name)
    plt.grid(True)


# Заданные параметры
start_N_2 = 4000
N_2 = 3650
L_2 = 365
r_2 = 50
M_2 = 100

start_N_3 = 1500
N_3 = 2400
L_3 = 240
r_3 = 5
M_3 = 60

time_series_2 = read_file("data2.xlsx", "x_pole", N_2, start_N_2)

v2_time_series_2 = read_file("data2.xlsx", "x_pole", N_2 + M_2, start_N_2)
v2_time_series_2_x = [i for i in range(N_2 + M_2)]
plot_graph(v2_time_series_2_x, v2_time_series_2, "График \"Данные 2\"", "red", 4)

result_2 = get_value_prediction(time_series_2, L_2, r_2, M_2)
print(result_2)
time_series_2.extend(result_2)
time_series_2_x = [i for i in range(N_2 + M_2)]
plot_graph(time_series_2_x, time_series_2, "График \"Данные 2\"", "black", 1)
plt.show()

remainders_2 = [y1 - y2 for y1, y2 in zip(v2_time_series_2, time_series_2)]
plot_graph(v2_time_series_2_x, remainders_2, "Остатки \"Данные 2\"", "black", 2)
plt.show()

time_series_3 = read_file("data3.xlsx", "data", N_3, start_N_3)

v2_time_series_3 = read_file("data3.xlsx", "data", N_3 + M_3, start_N_3)
v2_time_series_3_x = [i for i in range(N_3 + M_3)]
plot_graph(v2_time_series_3_x, v2_time_series_3, "График \"Данные 3\"", "red", 4)

result_3 = get_value_prediction(time_series_3, L_3, r_3, M_3)
print(result_3)
time_series_3.extend(result_3)
time_series_3_x = [i for i in range(N_3 + M_3)]
plot_graph(time_series_3_x, time_series_3, "График \"Данные 3\"", "black", 1)
plt.show()

remainders_3 = [y1 - y2 for y1, y2 in zip(v2_time_series_3, time_series_3)]
plot_graph(v2_time_series_3_x, remainders_3, "Остатки \"Данные 3\"", "black", 2)
plt.show()
