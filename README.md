## Amazon SageMaker Managed Spot Training Examples

This repository contains examples and related resources regarding Amazon SageMaker Managed Spot Training. 

Amazon SageMaker makes it easy to train machine learning models using managed Amazon EC2 Spot instances. Managed spot training can optimize the cost of training models up to 90% over on-demand instances. SageMaker manages the Spot interruptions on your behalf.

Managed Spot Training uses Amazon EC2 Spot instance to run training jobs instead of on-demand instances. You can specify which training jobs use spot instances and a stopping condition that specifies how long SageMaker waits for a job to run using Amazon EC2 Spot instances. Metrics and logs generated during training runs are available in CloudWatch.

Spot instances can be interrupted, causing jobs to take longer to start or finish. You can configure your managed spot training job to use checkpoints. SageMaker copies checkpoint data from a local path to Amazon S3. When the job is restarted, SageMaker copies the data from Amazon S3 back into the local path. The training can then resume from the last checkpoint instead of restarting. For more information about checkpointing, see Use Checkpoints in Amazon SageMaker.
  

Currently this repository has the following resources:

- **TensorFlow resources:**  

  - [**Training and using checkpointing on SageMaker Managed Spot Training with TensorFlow**](tensorflow_managed_spot_training_checkpointing):  This example shows a complete workflow for TensorFlow, showing how to train locally, on the SageMaker Notebook, to verify the training completes successfully. Then you train using SageMaker script mode, using on demand training instances. You continue training using SageMaker Managed Spot Training, simulating a spot interruption, and see how model training resumes from the latest epoch based on the checkpoints saved in S3. Finally, you deploy the model and produce a confusion matrix.    

## Questions?

Please contact [@e_sela](https://twitter.com/e_sela) or raise an issue on this repo.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

