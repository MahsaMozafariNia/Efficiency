The paper related to this code is accepted by CVPR workshop "Synthetic Data for Computer Vision - CVPR 2025", however, since the camera ready version is not sent yet, we could not publish the code and the name of the paper yet.

ReadMe for EpochLoss Data Minimization Code

The accompanying code is split into 3 main parts:
1. Running scripts: These are scripts indicating roughly how jobs may be run, to give an idea of how the hyperparameters are passed into the necessary scripts
2. Running scripts: The entrypoints for running the code. Namely these are:
	1. Main-steps.py: The script for training either the baseline on all training data or when removing a portion of the training data
	2. Eval-steps.py: A script for evaluating a given checkpoint on any task, for any attack/corruption setting, with or without substitution of synthetic images.
	3. CorruptDatasets.py: Generates corruptions and saves a given dataset. Used for generating the corrupted test sets, which remain the same between trials/experiments for consistency of comparison.

3. Auxiliary Scripts: Removal method code and any accompanying code needed to run the main scripts, such as dataset and network classes and functions



Note: Since we removed unused parts of the code for simplicity in sharing, there is a potential that some unintended reference to removed imports or functions are missing.
		As such this code is intended to give a reference to what was done, but a more thoroughly polished and coded ready-to-run version will be provided after submission.

