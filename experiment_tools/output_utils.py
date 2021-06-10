import mlflow.pytorch
from mlflow.tracking import MlflowClient


def get_mlflow_meta(experiment_id, order_by=["metrics.loss"], filter_string=""):
    # get the details of all FINISHED runs under experiment_id
    client = MlflowClient()
    filter_string = f"{filter_string} attribute.status='FINISHED'"
    return client.search_runs(
        experiment_id, order_by=order_by, filter_string=filter_string
    )


def load_all_models(experiment_id, params_filter_string=""):
    """Load all models in experiment_id
    - can filter by passing a params_filter_string of the form:
            # filter_string = (
            #     f"params.num_sources='{num_sources}' "
            #     f"params.physical_dim='{p}' "
            #     f"params.num_experiments='{num_experiments}'
            # )
    """

    meta = get_mlflow_meta(experiment_id, filter_string=params_filter_string)
    model_locs = [f"{x.info.artifact_uri}/model" for x in meta]
    return {model_loc: mlflow.pytorch.load_model(model_loc) for model_loc in model_locs}
