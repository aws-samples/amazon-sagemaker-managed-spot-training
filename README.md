## Amazon SageMaker Managed Spot Training Examples

This repository contains examples and related resources regarding Amazon SageMaker Managed Spot Training. 

For full details on how this works, read the Machine Learning Blog post at: https://aws.amazon.com/blogs/machine-learning/implement-checkpointing-with-tensorflow-for-amazon-sagemaker-managed-spot-training/

## Overview

Amazon SageMaker makes it easy to train machine learning models using managed Amazon EC2 Spot instances. Managed spot training can optimize the cost of training models up to 90% over on-demand instances. SageMaker manages the Spot interruptions on your behalf.

Managed Spot Training uses Amazon EC2 Spot instance to run training jobs instead of on-demand instances. You can specify which training jobs use spot instances and a stopping condition that specifies how long SageMaker waits for a job to run using Amazon EC2 Spot instances. Metrics and logs generated during training runs are available in CloudWatch.

Spot instances can be interrupted, causing jobs to take longer to start or finish. You can configure your managed spot training job to use checkpoints. SageMaker copies checkpoint data from a local path to Amazon S3. When the job is restarted, SageMaker copies the data from Amazon S3 back into the local path. The training can then resume from the last checkpoint instead of restarting. For more information about checkpointing, see Use Checkpoints in Amazon SageMaker.

### Repository Structure

The repository contains the following resources:

- **TensorFlow resources:**  

  - [**TensorFlow Training and using checkpointing on SageMaker Managed Spot Training**](tensorflow_managed_spot_training_checkpointing):  This example shows a complete workflow for TensorFlow, showing how to train locally, on the SageMaker Notebook, to verify the training completes successfully. Then you train using SageMaker script mode, using on demand training instances. You continue training using SageMaker Managed Spot Training, simulating a spot interruption, and see how model training resumes from the latest epoch, based on the checkpoints saved in S3. Finally, you deploy the model and produce a confusion matrix.
  - [**TensorFlow 2.x Training and using checkpointing on SageMaker Managed Spot Training**](tensorflow_2_managed_spot_training_checkpointing):  This example shows a complete workflow for TensorFlow, showing how to train locally, on the SageMaker Notebook, to verify the training completes successfully. Then you train using SageMaker script mode, using on demand training instances. You continue training using SageMaker Managed Spot Training, simulating a spot interruption, and see how model training resumes from the latest epoch, based on the checkpoints saved in S3. Finally, you deploy the model and invoke the endpoint.
  
- **PyTorch resources:**  

  - [**PyTorch Training and using checkpointing on SageMaker Managed Spot Training**](pytorch_managed_spot_training_checkpointing):  This example shows a complete workflow for PyTorch, showing how to train locally, on the SageMaker Notebook, to verify the training completes successfully. Then you train using SageMaker script mode, using on demand training instances. You continue training using SageMaker Managed Spot Training, simulating a spot interruption, and see how model training resumes from the latest epoch, based on the checkpoints saved in S3. Finally, you deploy the model and produce a confusion matrix.    

- **MXNet resources:**  

  - [**Apache MXNet Training and using checkpointing on SageMaker Managed Spot Training**](mxnet_managed_spot_training_checkpointing):  This example shows a training workflow for Apache MXNet, showing how to train using SageMaker script mode, using SageMaker Managed Spot Training, simulating a spot interruption, and see how model training resumes from the latest epoch, based on the checkpoints saved in S3.    

- **XGBoost resources:**  

  - [**Built-in XGBoost Training and using checkpointing on SageMaker Managed Spot Training**](xgboost_built_in_managed_spot_training_checkpointing):  This example shows a complete workflow for built-in XGBoost, showing how to train using SageMaker XGBoost built-in algorithm, using SageMaker Managed Spot Training, simulating a spot interruption, and see how model training resumes from the latest epoch, based on the checkpoints saved in S3.     
  - [**Script-mode XGBoost Training and using checkpointing on SageMaker Managed Spot Training**](xgboost_script_mode_managed_spot_training_checkpointing):  This example shows a complete workflow for script-mode XGBoost, showing how to train using SageMaker XGBoost algorithm in script mode, using SageMaker Managed Spot Training, simulating a spot interruption, and see how model training resumes from the latest epoch, based on the checkpoints saved in S3.

```bash
.
├── README.MD                                                           <-- This instructions file
├── cfn                                                                 <-- AWS CloudFormation Templates
│   └── create-sagemaker-notebook-cfn.yml                               <-- CloudFormation template to creates VPC, subnets, and SageMaker Notebook
├── mxnet_managed_spot_training_checkpointing                           <-- Apache MXNet Training and using checkpointing
│   └── mxnet_managed_spot_training_checkpointing.ipynb                 <-- Apache MXNet Training and using checkpointing notebook
│   └── source_dir                                                      <-- Training script 
        ├── mnist.py                                                    <-- MXNet Training script
├── pytorch_managed_spot_training_checkpointing                         <-- PyTorch Training and using checkpointing
│   └── pytorch_managed_spot_training_checkpointing.ipynb               <-- PyTorch Training and using checkpointing notebook
│   └── utils_cifar.py                                                  <-- Generates cifar10 dataset
│   └── source_dir                                                      <-- Training script 
        ├── cifar10.py                                                  <-- Pytorch Training script
├── tensorflow_managed_spot_training_checkpointing                      <-- TensorFlow Training and using checkpointing
│   └── tensorflow_managed_spot_training_checkpointing.ipynb            <-- TensorFlow Training and using checkpointing notebook
│   └── generate_cifar10_tfrecords.py                                   <-- Generates cifar10 tfrecords
│   └── source_dir                                                      <-- Training script 
        ├── cifar10_keras_main.py                                       <-- TensorFlow Training script
├── xgboost_built_in_managed_spot_training_checkpointing                <-- Built-in XGBoost Training and using checkpointing
│   └── xgboost_built_in_managed_spot_training_checkpointing.ipynb      <-- Built-in XGBoost Training and using checkpointing notebook
├── xgboost_script_mode_managed_spot_training_checkpointing             <-- Script mode XGBoost Training and using checkpointing
│   └── xgboost_script_mode_managed_spot_training_checkpointing.ipynb   <-- Script mode XGBoost Training and using checkpointing notebook
│   └── abalone.py                                                      <-- XGBoost Training script
```

### Installation Instructions

1. [Create an AWS account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one and login.

1. Clone the repo onto your local development machine using `git clone`.

### Setup

To create a VPC, subnets, and a SageMaker Notebook with this GitHub repository cloned, use the `create-sagemaker-notebook-cfn.yml` in the `cfn` directory. 

To deploy this template, run in a terminal:

```
aws cloudformation create-stack --region us-east-1 \
                                --stack-name create-sagemaker-notebook \
                                --template-body file://./create-sagemaker-notebook-cfn.yml\
                                --capabilities CAPABILITY_IAM
```
Note that the `template-body` parameter must include the `file://` prefix when run locally.

## Questions?

Please contact [@e_sela](https://twitter.com/e_sela) or raise an issue on this repo.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

