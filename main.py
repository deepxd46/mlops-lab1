
import time
from typing import List
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)


MODEL_VERSION = "iris_onnx_v1.0"


class IrisRequestData(BaseModel):
    sepal_length: float = Field(
        ..., gt=0, description="Длина чашелистика в см", example=5.1
    )
    sepal_width: float = Field(
        ..., gt=0, description="Ширина чашелистика в см", example=3.5
    )
    petal_length: float = Field(
        ..., gt=0, description="Длина лепестка в см", example=1.4
    )
    petal_width: float = Field(
        ..., gt=0, description="Ширина лепестка в см", example=0.2
    )


class IrisResponseData(BaseModel):
    predicted_class: int = Field(
        ..., description="Индекс предсказанного класса (0, 1 или 2)"
    )
    probabilities: List[float] = Field(
        ..., description="Вектор вероятностей распределения по классам"
    )
    model_status: str = Field(
        ..., description="Системный статус и идентификатор версии инференса"
    )


PREDICTIONS_COUNTER = Counter(
    "predictions_total",
    "Суммарное количество обработанных запросов на инференс",
    labelnames=["predicted_class", "model_version"],
)

INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Время выполнения полного цикла инференса модели в секундах",
    buckets=[
        0.001, 0.005, 0.01, 0.025, 0.05,
        0.1, 0.25, 0.5, 1.0, float("inf")
    ],
)


onnx_session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_session
    print("Инициализация сервера: загрузка графа ONNX в оперативную память...")
    onnx_session = ort.InferenceSession("iris_model.onnx")
    yield
    print("Остановка сервера: освобождение ресурсов памяти...")
    onnx_session = None


app = FastAPI(lifespan=lifespan)


@app.post("/v1/predict", response_model=IrisResponseData)
def predict(data: IrisRequestData):
    start_time = time.perf_counter()

    input_array = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], dtype=np.float32)

    input_name = onnx_session.get_inputs()[0].name
    raw_predictions = onnx_session.run(None, {input_name: input_array})

    predicted_class = int(raw_predictions[0][0])
    probabilities = list(raw_predictions[1][0].values())

    PREDICTIONS_COUNTER.labels(
        predicted_class=str(predicted_class),
        model_version=MODEL_VERSION,
    ).inc()

    duration = time.perf_counter() - start_time
    INFERENCE_LATENCY.observe(duration)

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "model_status": MODEL_VERSION,
    }


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
