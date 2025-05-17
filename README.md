# Build and Run docker image

docker build -t model_service . && docker run --name model_service --network mlflow-net -p 8000:8000 -v "D:/mlflow/mlruns:/mlflow/mlruns" model_service
