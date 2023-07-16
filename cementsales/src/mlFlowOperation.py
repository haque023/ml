import asyncio
from datetime import datetime
import mlflow
from mlflow import MlflowClient
import matplotlib.pyplot as plt
from cementsales.src.model.train import train_cement_data, get_test_datas, get_metrics
from cementsales.src.modelRegistry import create_experiment, register_model

mlflow.set_tracking_uri("http://127.0.0.1:5000")


def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


async def init_experiment():
    model = await train_cement_data()
    experiment_name = "sales_01" + str(datetime.now().strftime("%d-%m-%y"))
    run_name = "sales_01" + str(datetime.now().strftime("%d-%m-%y"))
    test = await get_test_datas()
    y_predict = model.predict(test['x_train'])
    metrics = get_metrics(test['y_train'], y_predict)
    print(metrics)
    run_id = create_experiment(experiment_name, run_name, metrics, model)
    return run_id


async def fetch_model_version(model_name, model_version, test):
    import mlflow.pyfunc
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    print(test)
    y_predict = model.predict(test)
    print(y_predict)
    plt.plot(test, y_predict)
    plt.show()


async def transition_model(model_name, model_version, stage):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name=model_name, version=model_version, stage=stage)


if __name__ == "__main__":
    # run_id = asyncio.run(init_experiment())
    # print(run_id)
    # asyncio.run(register_model(run_id))
    test = asyncio.run(get_test_datas())
    #
    asyncio.run(fetch_model_version("linear", 1, test["x_test"]))
