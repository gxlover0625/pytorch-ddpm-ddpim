# Background
The origin code is forked from https://github.com/w86763777/pytorch-ddpm.
And we extend the code with the implementation of DDIM.

# Dataset
Cifar: 32 * 32, 10 classes, use `torchvision` to download the datasets.

# Files
constant.py -> This is the file of the parameters for the model.
model.py -> This is the code of the model that capture the features of images.
diffusion.py -> This is the code of the implementation of DDIM and DDPM.
main.py -> The is the code of training or evaluation. 

# Usage
you can alternate the working mode of model in the main.py. The function train()
means training and saving model. And the function eval() means using the trained
model to inference.
```shell
python main.py
```

# Results
## DDPM
The results are stored in the `DDPM Inference` directory.
## DDIM
The results are stored in the `DDIM Inference` directory.