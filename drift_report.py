
import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


# Фиксируем генератор для воспроизводимости результатов
rng = np.random.default_rng(42)

SAMPLE_SIZE = 1000

# Эталонные данные: обычное распределение признаков Iris
reference_data = pd.DataFrame({
    "sepal_length": rng.normal(5.8, 0.8, SAMPLE_SIZE),
    "sepal_width": rng.normal(3.0, 0.4, SAMPLE_SIZE),
    "petal_length": rng.normal(3.8, 1.7, SAMPLE_SIZE),
    "petal_width": rng.normal(1.2, 0.7, SAMPLE_SIZE),
})

# Текущие данные: моделируем изменение распределений
current_data = pd.DataFrame({
    "sepal_length": rng.normal(8.5, 0.8, SAMPLE_SIZE),
    "sepal_width": rng.normal(2.0, 0.4, SAMPLE_SIZE),
    "petal_length": rng.normal(6.5, 1.7, SAMPLE_SIZE),
    "petal_width": rng.normal(2.8, 0.7, SAMPLE_SIZE),
})

# Создаем отчет о дрейфе данных
report = Report([
    DataDriftPreset(method="psi")
], include_tests=True)

# Сравниваем текущую выборку с эталонной
snapshot = report.run(
    current_data=current_data,
    reference_data=reference_data,
)

# Сохраняем визуальный отчет
snapshot.save_html("data_drift_report.html")

print("Отчет Data Drift успешно создан!")
print("Файл: data_drift_report.html")
print(f"Размер эталонной выборки: {len(reference_data)}")
print(f"Размер текущей выборки: {len(current_data)}")
