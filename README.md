
# MDARNet
Multi-scale Self-calibrated Dual-attention Lightweight Residual Dense Deraining Network Based On Monogenic Wavelets
    Image rain removal is an essential problem of common concern in the fields of image processing and computer vision. Existing methods have resorted to deep learning techniques to separate rain streaks from the background by leveraging some prior knowledge. However, due to complex distribution of rainfall in the real environment, it is difficult for synthetic datasets to simulate these conditions, which results in inaccurate deraining. To address this issue, we propose a multi-scale self-calibrated dual attention lightweight residual dense deraining network (MDARNet) for better deraining performance. Specifically, the network consists of monogenic wavelet transform-like hierarchy and self-calibrated dual attention mechanism. With the help of scale-space properties of the monogenic wavelet transform, key features at different scales can be extracted at the same location, which makes it easier to match structural features across scales. The self-calibrated double attention mechanism was used as a basic model for enhancing the channel dependence and spatial correlation between each layer component of the monogenic wavelet transform. Thus, the network can establish long-range dependencies and take advantage of rich contextual information and multi-scale redundancy to accommodate rain streaks of different shapes and sizes. Experiments on synthetic and real image datasets show that the method outperforms many of the latest single-image denoising methods in terms of visual and quantitative metrics. The source code can be obtained from https://github.com/smart-hzw/MDARNet.

! [image1.jpg](./show/fig1.jpg)

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



## Training

Download the dataset from  [train data](https://pan.baidu.com/s/18HO_Vdb6D550Oo21FVImKQ )  （password：3eht） put the dataset folder into the "./data/train" folder

Run the following commands:

```python
python train.py --train_data ./data/train/ --val_data ./data/test/Rain100H/ --save_weights ./models/mixTrain/
```



## Testing

Download the commonly used testing rain dataset (R100H, R100L, TEST100, TEST1200, TEST2800) [testdata](https://pan.baidu.com/s/1RAltot4YErlmRw69cdHwJA ) (password:fd4r),In addition, the test results of other competing models can be downloaded from here [Rain100H,Rain100L](https://pan.baidu.com/s/1vnflrrUw2HKxK_Xd3uvutg) (password: jwdt) [Test100,Test1200](https://pan.baidu.com/s/1lBjcqXv4e1gfv6KqF9D6pg )(password: johk)

Run the following commands:

```python
python test.py --input_dir ./data/test/ --result_dir  ./data/test/ --weights ./models/mixTrain/net_epoch101.pth
```



* The deraining results will be in "./result". We only provide the baseline for comparison. There exists the gap (0.1-0.2db) between the provided model and the reported values in the paper, which originates in the subsequent fine-tuning of hyperparameters, training processes and constraints.