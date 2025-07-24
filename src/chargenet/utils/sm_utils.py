from typing import Optional
from tempfile import TemporaryDirectory
from pathlib import Path
from loguru import logger

import sagemaker
from sagemaker.estimator import Estimator
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial

import pandas as pd

from protest.base.utils import download_output_artifacts_from_s3
from protest.sagemaker.utils import TrainJob, is_s3_uri, upload_data
from protest.sagemaker.constants import REGISTRY
from protest.sagemaker.definitions.metrics import ModelMetricsDefinitions

from apbsconv.runs import APBSPipelineParameters


class ChargeNetEstimator(Estimator):
    metric_definitions = ModelMetricsDefinitions().to_list()

    def __init__(
        self,
        role: str,
        hyperparameters: Optional[dict] = None,
        image_uri=f"{REGISTRY}/chargenet:latest",
        instance_type: str = "ml.g5.2xlarge",
        instance_count: int = 1,
        **kwargs,
    ):
        super().__init__(
            image_uri=image_uri,
            instance_type=instance_type,
            instance_count=instance_count,
            role=role,
            tags=[
                {
                    "Key": "Application",
                    "Value": "Protein property prediction model training job",
                },
                # {"Key": "ITLT_Owner",
                #  "Value": "pauline.t.christensen@iff.com"}
            ],
            hyperparameters=hyperparameters,
            metric_definitions=self.metric_definitions,
            script_mode=True,
            **kwargs,
        )


REGISTRY = "118749263921.dkr.ecr.us-east-1.amazonaws.com"


def start_training_job(
    data_path: str,
    reference_sequence: str,
    target: str,
    pdb_file_path: str,
    mutagenesis_tool: str,
    channel_configuration: str,
    use_ph: int = 0,
    use_temp: int = 0,
    kernel_edge_length: int = 3,
    pooling_edge_length: int = 3,
    n_blocks: int = 2,
    out_channels: int = 7,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    loss: str = "mse",
    optimizer: str = "adam",
    epochs: int = 300,
    batch_size: int = 8,
    n_cores: int = 1,
    intermediate_data_path: str = None,
    electrostatics_path: str = None,
    write_electrostatics_path: str = None,
    instance_type: str = "ml.g5.2xlarge",
    sagemaker_role: str = None,
    wait: bool = False,
) -> ChargeNetEstimator:
    job = TrainJob(
        "chargenet",
        data_path,
        sagemaker_role=sagemaker_role,
        instance_type=instance_type,
    )
    if not is_s3_uri(data_path):
        data_path = upload_data(
            session=sagemaker.Session(),
            base_job_name=job.training_job_name,
            path=data_path,
            channel="train",
        )

    # Specifying the sagemaker training job configuration
    estimator = ChargeNetEstimator(
        role=job.sagemaker_role,
        instance_type=job.instance_type,
        instance_count=1,
        max_run=10 * 60 * 60,
        hyperparameters=APBSPipelineParameters(
            prepared_dataset_uri=data_path,
            reference_sequence=reference_sequence,
            target=target,
            pdb_file_path=pdb_file_path,
            mutagenesis_tool=mutagenesis_tool,
            use_ph=use_ph,
            use_temp=use_temp,
            channel_configuration=channel_configuration,
            kernel_edge_length=kernel_edge_length,
            pooling_edge_length=pooling_edge_length,
            n_blocks=n_blocks,
            out_channels=out_channels,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss=loss,
            batch_size=batch_size,
            epochs=epochs,
            patience=20,
            device="cuda",
            n_cores=n_cores,
            intermediate_data_path=intermediate_data_path,
            electrostatics_path=electrostatics_path,
            write_electrostatics_path=write_electrostatics_path,
        ).dict(exclude_none=True),
    )
    # Starting the training job, either locally or on AWS
    estimator.fit(job.train_location, job_name=job.training_job_name, wait=wait)
    return estimator


def start_tuning_job(
    data_path: str,
    target: str,
    pdb_file_path: str,
    instance_type: str = "ml.g5.16xlarge",
    device: str = "cuda",
    electrostatics_path: str = None,
    sagemaker_role: str = None,
    wait: bool = False,
):
    job = TrainJob(
        "chargenet",
        data_path,
        sagemaker_role=sagemaker_role,
        instance_type=instance_type,
    )

    hyperparameter_ranges = {
        "out_channels": sagemaker.tuner.IntegerParameter(4, 32),
        "n_blocks": sagemaker.tuner.IntegerParameter(1, 5),
        "kernel_edge_length": sagemaker.tuner.IntegerParameter(3, 9),
        "pooling_edge_length": sagemaker.tuner.IntegerParameter(3, 9),
        "batch_size": sagemaker.tuner.IntegerParameter(8, 32),
        "optimizer": sagemaker.tuner.CategoricalParameter(["adam", "adamw"]),
        "weight_decay": sagemaker.tuner.ContinuousParameter(1e-6, 0.1),
        "loss": sagemaker.tuner.CategoricalParameter(
            ["mse", "mae", "huber", "spearman"]
        ),
        "learning_rate": sagemaker.tuner.ContinuousParameter(1e-5, 1e-3),
    }

    estimator = ChargeNetEstimator(
        role=job.sagemaker_role,
        instance_type=job.instance_type,
        instance_count=1,
        hyperparameters=APBSPipelineParameters(
            prepared_dataset_uri=data_path,
            target=target,
            pdb_file_path=pdb_file_path,
            channel_configuration="all",
            epochs=300,
            patience=20,
            device=device,
            n_cores=60,
            electrostatics_path=electrostatics_path,
        ).dict(exclude_none=True),
    )

    tuner = sagemaker.tuner.HyperparameterTuner(
        estimator=estimator,
        metric_definitions=estimator.metric_definitions,
        objective_metric_name="test_spearman",
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type="Maximize",
        max_jobs=100,
        max_parallel_jobs=10,
        strategy="Bayesian",
    )

    tuner.fit(job_name=job.training_job_name, wait=wait)
    return tuner


def run_experiment(
    estimator: Estimator,
    job: TrainJob,
    experiment_name: str,
    trial_name: str,
    wait: bool = False,
) -> Estimator:
    try:
        trial = Trial.create(trial_name=trial_name, experiment_name=experiment_name)
    except ValueError:
        trial = Trial.load(trial_name=trial_name)
    estimator.fit(
        job.train_location,
        job_name=job.training_job_name,
        experiment_config={"TrialName": trial.trial_name},
        wait=wait,
    )
    return estimator


def delete_sm_experiment(experiment_name):
    exp = Experiment.load(experiment_name)
    for trial_sum in exp.list_trials():
        trial = Trial.load(trial_sum.trial_name)
        for trial_component in trial.list_trial_components():
            trial.remove_trial_component(trial_component)
        trial.delete()
    exp.delete()


def fetch_predictions(uri):
    with TemporaryDirectory() as temp_dir:
        download_output_artifacts_from_s3(uri, temp_dir)
        predictions = pd.read_csv(Path(temp_dir) / "predictions.csv")
    return predictions


class TrainingJobManager:
    def __init__(self, experiment_dir, folds, model_limits, client, default_bucket):
        self.experiment_dir = experiment_dir
        self.folds = folds
        self.model_limits = model_limits
        self.client = client
        self.default_bucket = default_bucket
        self.experiment_status_file = (
            f"{self.experiment_dir}/experiment_status_info_{self.folds}_folds.csv"
        )
        self.dataframe = None

    def load(self):
        if Path(self.experiment_status_file).is_file():
            logger.info("Reading existing status file.")
            self.dataframe = pd.read_csv(self.experiment_status_file, index_col=None)
        else:
            logger.info("No existing status file found, creating new.")
            self.dataframe = pd.DataFrame(
                columns=["test_fold", "run_type", "model", "job_name", "status"]
            )

    def add_run(self, test_fold, run_type, model, job_name, status):
        new_row = {
            "test_fold": test_fold,
            "run_type": run_type,
            "model": model,
            "job_name": job_name,
            "status": status,
        }
        self.dataframe = pd.concat(
            [self.dataframe, pd.DataFrame(new_row, index=[0])], axis=0
        ).reset_index(drop=True)

    def update(self):
        jobs_to_update = self.dataframe[self.dataframe["status"] == "InProgress"][
            "job_name"
        ]
        for job_name in jobs_to_update:
            status = check_training_job_status(job_name, self.client)
            self.dataframe.loc[self.dataframe["job_name"] == job_name, "status"] = (
                status
            )
        self.dataframe.to_csv(self.experiment_status_file, index=False)

    def running_jobs_at_maximum(self, model):
        limit = self.model_limits[model]
        running_jobs = self.dataframe[
            (self.dataframe["model"] == model)
            & (self.dataframe["status"] == "InProgress")
        ]
        return len(running_jobs) == limit

    def run_already_started(self, test_fold, run_type, model):
        match = self.dataframe[
            (self.dataframe["test_fold"] == test_fold)
            & (self.dataframe["run_type"] == run_type)
            & (self.dataframe["model"] == model)
        ]
        if match.empty:
            return False
        else:
            return any(
                [
                    status in ["InProgress", "Completed"]
                    for status in match["status"].values
                ]
            )

    def fetch_test_predictions(self, test_fold, run_type, model):
        job_name = self.dataframe[
            (self.dataframe["test_fold"] == test_fold)
            & (self.dataframe["run_type"] == run_type)
            & (self.dataframe["model"] == model)
            & (self.dataframe["status"] == "Completed")
        ]["job_name"].values[0]
        uri = f"s3://{self.default_bucket}/{job_name}/output/output.tar.gz"
        with TemporaryDirectory() as temp_dir:
            download_output_artifacts_from_s3(uri, temp_dir)
            predictions = pd.read_csv(Path(temp_dir) / "predictions.csv")
        return predictions[predictions["split"] == "test"]["y_pred"]


def check_training_job_status(job_name, sagemaker_client):
    response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    status = response["TrainingJobStatus"]
    return status
