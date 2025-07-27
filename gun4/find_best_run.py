import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("sms_model_compare")
runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                          order_by=["metrics.f1_score DESC"],
                          max_results=1)

best_run = runs[0]
run_id = best_run.info.run_id

# best_run.txt dosyasına yaz
with open("best_run.txt", "w") as f:
    f.write(run_id)

print(f"En iyi run_id: {run_id}")