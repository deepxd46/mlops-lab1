# === ЭТАП 1: Сборщик (Builder) ===
FROM python:3.10-slim AS builder
WORKDIR /build

# Устанавливаем системные инструменты компиляции
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Обновляем pip, чтобы избежать бага JSONDecodeError
RUN pip install --upgrade pip
RUN pip wheel --default-timeout=100 --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt

# === ЭТАП 2: Финальный продакшен-образ ===
FROM python:3.10-slim
WORKDIR /app

# Извлекаем только готовые скомпилированные библиотеки
COPY --from=builder /build/wheels /wheels

# Обновляем pip и устанавливаем собранные пакеты
RUN pip install --upgrade pip
COPY --from=builder /build/requirements.txt .

RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY iris_model.onnx .
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]