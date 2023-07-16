import mlflow


def create_experiment(experiment_name, run_name, run_metrics, model, confusion_matrix_path=None,
                      roc_auc_plot_path=None, run_params=None):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])

        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')

        if not roc_auc_plot_path == None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
        mlflow.sklearn.log_model(model, "model")
    print('Run - %s is logged to Experiment - %s' % (run_name, experiment_name))
    return run.info.run_id


def create_experiment_and_register(experiment_name, run_name, run_metrics, model, confusion_matrix_path=None,
                                   roc_auc_plot_path=None, run_params=None):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])

        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')

        if not roc_auc_plot_path == None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
        mlflow.sklearn.log_model(model, "model")
    print('Run - %s is logged to Experiment - %s' % (run_name, experiment_name))
    return run.info.run_id

async def register_model(run_id):
    result = mlflow.register_model("runs:/"+run_id+"/model", "linear")
    print(result)