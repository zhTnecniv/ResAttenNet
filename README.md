# Residual Attention Network

In this project, I implement and train four residual attention networks (Attention-56, Attention-92, Attention-128, AttentionNeXt-56) as proposed by F. Wang et al. in the paper Residual Attention Network for Image Classification (https://arxiv.org/pdf/1704.06904.pdf). After experimenting with different parameters and slight modification of the architectures, I am able to achieve error rates of about 10% for each these three models on CIFAR-10 dataset, which are around 5% higher than the error rates attained by the authors as shown in the paper. I also implement and train three baseline residual networks (ResNet50V2, ResNet101V2, and ResNet152V2) in order to investigate whether residual attention networks are better subject to the same training parameters. As the three baseline models only attain error rates of about 20%, residual attention networks successfully demonstrate a noticeable advantage in image classification task.

Cheers!
