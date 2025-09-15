###########################################
# VGG16 baselines (Laplace Approximation) #
###########################################

# CIFAR-10 (ID) with OOD = SVHN
python evaluate_laplace.py      --seed 1        --dataset cifar10        --ood_dataset svhn        --load-path model_vgg_10        --model vgg16       --runs 25

# CIFAR-10 (ID) with OOD = CIFAR-100
python evaluate_laplace.py      --seed 1        --dataset cifar10        --ood_dataset cifar100        --load-path model_vgg_10        --model vgg16       --runs 25

# CIFAR-10 (ID) with OOD = TinyImageNet
python evaluate_laplace.py      --seed 1        --dataset cifar10        --ood_dataset tinyimagenet        --load-path model_vgg_10        --model vgg16       --runs 25

# CIFAR-100 (ID) with OOD = SVHN
python evaluate_laplace.py      --seed 1        --dataset cifar100        --ood_dataset svhn        --load-path model_vgg_100        --model vgg16       --runs 25

# CIFAR-100 (ID) with OOD = TinyImageNet
python evaluate_laplace.py      --seed 1        --dataset cifar100        --ood_dataset tinyimagenet        --load-path model_vgg_100        --model vgg16       --runs 25


#################################################
# Wide-ResNet baselines (Laplace Approximation) #
#################################################

# CIFAR-10 (ID) with OOD = SVHN (spectral norm + modified arch, coeff=3.0)
python evaluate_laplace.py      --seed 1        --dataset cifar10        --ood_dataset svhn        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0

# CIFAR-10 (ID) with OOD = CIFAR-100
python evaluate_laplace.py      --seed 1        --dataset cifar10        --ood_dataset cifar100        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0

# CIFAR-10 (ID) with OOD = TinyImageNet
python evaluate_laplace.py      --seed 1        --dataset cifar10        --ood_dataset tinyimagenet        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0

# CIFAR-100 (ID) with OOD = SVHN
python evaluate_laplace.py      --seed 1        --dataset cifar100        --ood_dataset svhn        --load-path model_wide_100        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0

# CIFAR-100 (ID) with OOD = TinyImageNet
python evaluate_laplace.py      --seed 1        --dataset cifar100        --ood_dataset tinyimagenet        --load-path model_wide_100        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0


###########################################
# Union OOD evaluation (VGG16 + WideResNet)
###########################################

# VGG16 on CIFAR-10 with union OOD dataset
python evaluate_laplace.py      --seed 1        --dataset cifar10        --ood_dataset ood_union        --load-path model_vgg_10        --model vgg16       --runs 25

# VGG16 on CIFAR-100 with union OOD dataset
python evaluate_laplace.py      --seed 1        --dataset cifar100        --ood_dataset ood_union        --load-path model_vgg_100        --model vgg16       --runs 25

# Wide-ResNet on CIFAR-10 with union OOD dataset (sn + mod, coeff=3.0)
python evaluate_laplace.py      --seed 1        --dataset cifar10        --ood_dataset ood_union        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0

# Wide-ResNet on CIFAR-100 with union OOD dataset
python evaluate_laplace.py      --seed 1        --dataset cifar100        --ood_dataset ood_union      --load-path model_wide_100        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0
