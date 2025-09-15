############################
# VGG16 baselines (SPC-UQ) #
############################

# CIFAR-10 (ID) with OOD = SVHN
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset svhn        --load-path model_vgg_10        --model vgg16       --runs 25    --model-type spc

# CIFAR-10 (ID) with OOD = CIFAR-100
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset cifar100        --load-path model_vgg_10        --model vgg16       --runs 25    --model-type spc

# CIFAR-10 (ID) with OOD = TinyImageNet
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset tinyimagenet        --load-path model_vgg_10        --model vgg16       --runs 25    --model-type spc

# CIFAR-100 (ID) with OOD = SVHN
python evaluate.py      --seed 1        --dataset cifar100        --ood_dataset svhn        --load-path model_vgg_100        --model vgg16       --runs 25    --model-type spc

# CIFAR-100 (ID) with OOD = TinyImageNet
python evaluate.py      --seed 1        --dataset cifar100        --ood_dataset tinyimagenet        --load-path model_vgg_100        --model vgg16       --runs 25    --model-type spc


##################################
# Wide-ResNet baselines (SPC-UQ) #
##################################

# CIFAR-10 (ID) with OOD = SVHN (spectral norm + modified arch, coeff=3.0)
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset svhn        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc

# CIFAR-10 (ID) with OOD = CIFAR-100
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset cifar100        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc

# CIFAR-10 (ID) with OOD = TinyImageNet
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset tinyimagenet        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc

# CIFAR-100 (ID) with OOD = SVHN
python evaluate.py      --seed 1        --dataset cifar100        --ood_dataset svhn        --load-path model_wide_100        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc

# CIFAR-100 (ID) with OOD = TinyImageNet
python evaluate.py      --seed 1        --dataset cifar100        --ood_dataset tinyimagenet        --load-path model_wide_100        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc


###################################################################
# Union OOD evaluation (VGG16 + WideResNet)
# Training on training set for epistemic uncertainty quantification
###################################################################

# VGG16 on CIFAR-10 with union OOD dataset
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset ood_union        --load-path model_vgg_10        --model vgg16       --runs 25    --model-type spc

# VGG16 on CIFAR-100 with union OOD dataset
python evaluate.py      --seed 1        --dataset cifar100        --ood_dataset ood_union        --load-path model_vgg_100        --model vgg16       --runs 25    --model-type spc

# Wide-ResNet on CIFAR-10 with union OOD dataset (sn + mod, coeff=3.0)
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset ood_union        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc

# Wide-ResNet on CIFAR-100 with union OOD dataset
python evaluate.py      --seed 1        --dataset cifar100        --ood_dataset ood_union        --load-path model_wide_100        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc


###################################################################
# Union OOD evaluation (VGG16 + WideResNet)
# Training on calibration set for aleatoric uncertainty calibration
###################################################################

# VGG16 on CIFAR-10
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset ood_union        --load-path model_vgg_10        --model vgg16       --runs 25    --model-type spc   --val_size 0.1  -crossval

# VGG16 on CIFAR-100
python evaluate.py      --seed 1        --dataset cifar100        --ood_dataset ood_union        --load-path model_vgg_100        --model vgg16       --runs 25    --model-type spc   --val_size 0.1  -crossval

# Wide-ResNet on CIFAR-10 (sn + mod, coeff=3.0)
python evaluate.py      --seed 1        --dataset cifar10        --ood_dataset ood_union        --load-path model_wide_10        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc   --val_size 0.1  -crossval

# Wide-ResNet on CIFAR-100
python evaluate.py      --seed 1        --dataset cifar100        --ood_dataset ood_union        --load-path model_wide_100        --model wide_resnet       --runs 25    -sn -mod     --coeff 3.0    --model-type spc   --val_size 0.1  -crossval

