##################################
# ImageNet → ImageNet-O (OOD eval)
# Comparing different UQ methods
##################################

##########
# Softmax
##########

# VGG16 baseline (softmax confidence)
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_vgg_1000 --model imagenet_vgg16 --runs 3 --model-type softmax

# Wide-ResNet baseline (softmax confidence)
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_wide_1000 --model imagenet_wide --runs 3 --model-type softmax


#######
# SPC
#######

# VGG16 with SPC-UQ (split-point self-consistency)
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_vgg_1000 --model imagenet_vgg16 --runs 3 --model-type spc

# Wide-ResNet with SPC-UQ
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_wide_1000 --model imagenet_wide --runs 3 --model-type spc


#######
# GMM
#######

# VGG16 with GMM-based UQ
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_vgg_1000 --model imagenet_vgg16 --runs 3 --model-type gmm

# Wide-ResNet with GMM-based UQ
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_wide_1000 --model imagenet_wide --runs 3 --model-type gmm


######
# OC
######

# VGG16 with Orthogonal Certificates (OC)
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_vgg_1000 --model imagenet_vgg16 --runs 3 --model-type oc

# Wide-ResNet with OC
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_wide_1000 --model imagenet_wide --runs 3 --model-type oc


#############
# Ensemble
#############

# VGG16 with Deep Ensemble
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_vgg_1000 --model imagenet_vgg16 --runs 3 --model-type ensemble

# Wide-ResNet with Deep Ensemble
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_wide_1000 --model imagenet_wide --runs 3 --model-type ensemble


########
# EDL
########

# VGG16 with Evidential Deep Learning (EDL)
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_vgg_1000 --model imagenet_vgg16 --runs 3 --model-type edl

# Wide-ResNet with EDL
python evaluate.py --seed 1 --dataset imagenet --ood_dataset imagenet_o --load-path model_wide_1000 --model imagenet_wide --runs 3 --model-type edl
