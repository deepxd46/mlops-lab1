
import random

from locust import HttpUser, between, task


class IrisNormalUser(HttpUser):
    """Обычные запросы к модели Iris."""

    wait_time = between(0.5, 1.5)
    weight = 4

    @task
    def predict_normal(self):
        samples = [
            [5.1, 3.5, 1.4, 0.2],
            [6.3, 2.5, 5.0, 1.9],
            [5.9, 3.0, 4.2, 1.5],
        ]

        values = random.choice(samples)

        payload = {
            "sepal_length": values[0],
            "sepal_width": values[1],
            "petal_length": values[2],
            "petal_width": values[3],
        }

        self.client.post(
            "/v1/predict",
            json=payload,
            name="/v1/predict [normal]",
        )


class IrisAnomalyUser(HttpUser):
    """Аномальные, но допустимые входные признаки."""

    wait_time = between(0.5, 1.5)
    weight = 1

    @task
    def predict_anomaly(self):
        payload = {
            "sepal_length": random.uniform(9.0, 12.0),
            "sepal_width": random.uniform(0.1, 1.0),
            "petal_length": random.uniform(7.0, 10.0),
            "petal_width": random.uniform(3.0, 5.0),
        }

        self.client.post(
            "/v1/predict",
            json=payload,
            name="/v1/predict [anomaly]",
        )
