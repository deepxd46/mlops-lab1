import mlflow
import mlflow.sklearn
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# 1. Настройка подключения к серверу MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def main():
    # Загружаем модель со стадии Staging
    model_uri = "models:/iris_production_model/Staging"
    print(f"Загрузка модели из реестра: {model_uri}...")
    
    # Скачиваем живой Python-объект модели из Model Registry
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    # --- Конвертация в формат ONNX ---
    # ИСПРАВЛЕНО: Используем FloatTensorType, который оборачивает размерность
    initial_type = [('float_input', FloatTensorType([None, 4]))] 
    
    print("Трансляция графа вычислений в формат ONNX...")
    
    # Преобразуем модель в универсальное ONNX-представление
    onnx_model = to_onnx(loaded_model, initial_types=initial_type, target_opset=15)
    
    # Сохраняем скомпилированный бинарный файл на диск
    onnx_filename = "iris_model.onnx"
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"Экспорт успешно завершен! Файл сохранен как: '{onnx_filename}'")

if __name__ == "__main__":
    main()