import mlflow 
import os 
import mlflow.sagemaker

ifile = open("setup_mlflow.txt", "r").readlines()
mlflow_tracking_uri = ifile[0].split("=")[1].strip()
mlflow_tracking_username = ifile[1].split("=")[1].strip()
mlflow_tracking_password = ifile[2].split("=")[1].strip()
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_tracking_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_tracking_password
print(os.environ.get("MLFLOW_TRACKING_URI"))
print(os.environ.get("MLFLOW_TRACKING_USERNAME"))
print(os.environ.get("MLFLOW_TRACKING_PASSWORD"))

image_uri = "230178520806.dkr.ecr.eu-west-1.amazonaws.com/mlflow-pyfunc:1.28.0"
model_uri = "models:/SageMaker1/Staging"
region = "eu-west-1"
aws_id = "230178520806"
arn = "arn:aws:iam::230178520806:role/awssagemakerdeployment"

mlflow.sagemaker.deploy(
    mode='create',
    app_name="NaiveBayesTest",
    model_uri=model_uri,
    image_url=image_uri,
    execution_role_arn=arn,
    instance_type="ml.t2.medium",
    instance_count=1,
    region_name=region
)
