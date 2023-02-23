# MDARNet
# Multi scale dual attention mechanism residual rain removal network based on monogenic wavelet
	Abstract—Image rain removal is an essential problem of common concern in the fields of image processing and computer vision. 
	Existing methods have resorted to deep learning techniques to separate rain streaks from the background by leveraging some 
	prior knowledge. However, the mismatch between the size of the rain streaks during the training and testing phases, especially 
	when large rain streaks are present, frequently leads to unsatisfactory deraining results. To address this issue, we propose a 
	multi-scale self-calibrated dual attention lightweight residual dense deraining network (MDARNet) for better deraining 
	performance. Specifically, the network consists of monogenic wavelet transform-like hierarchy and self-calibrated dual 
	attention mechanism. With the help of scale-space properties of the monogenic wavelet transform, key features at different 
	scales can be extracted at the same location, which makes it easier to match structural features across scales. The self-calibrated
	double attention mechanism was used as a basic model for enhancing the channel dependence and spatial correlation between each layer 
	component of the monogenic wavelet transform. Thus, the network can establish long-range dependencies and take advantage of rich 
	contextual information and multi-scale redundancy to accommodate rain streaks of different shapes and sizes. Experiments on synthetic
	and real image datasets show that the method outperforms many of the latest single-image deraining methods in terms of visual and 
	quantitative metrics. The source code can be obtained from https://github.com/smart-hzw/MDARNet. 
	
	Index Terms—Single image deraining, Monogenic wavelet transform, Self-calibrated dual attention mechanism, Lightweight residual dense network
![contents](https://github.com/smart-hzw/MDARNet/blob/main/fig1.jpg)

## Prepare
	install toch 1.10
	install hdf5 2.10.0
	install opencv-python 4.5.4.60
	install lpips 0.1.3
	install tqdm 4.59.0
	install tensorboardX 2.2
	install torchvision 0.11.1
	install scikit-image 0.16.2
	install numpy 1.20.3

Download the dataset from (Link：https://pan.baidu.com/s/1sNxF4GuZWte-Wjdgd40PDQ Password：vxqy)  put the dataset folder into the "IDCGAN/data" folder

## Training
* rain100H train
	
		python MDABNet_train.py --train_data ./data/train --val_data ./data/test/Rain100H/ --batchsize 8 --save_weights ./models/

* The deraining results will be in "./result". We only provide the baseline for comparison. There exists the gap (0.1-0.2db) between the provided model and the reported values in the paper, which originates in the subsequent fine-tuning of hyperparameters, training processes and constraints.
	
## Testing
        We will publish test models later

## Citation
    @article{hao2022multi,
     title={Multi-scale Self-calibrated Dual-attention Lightweight Residual Dense Deraining Network Based On Monogenic Wavelets},
    author={Hao, Zhiwei and Gai, Shan and Li, Pengcheng},
    journal={IEEE Transactions on Circuits and Systems for Video Technology},
    year={2022},
    publisher={IEEE}
    }
