## 使用MLP 做openpose分类

这是一个简单的案例

我用了5层MLP将openpose得到的16个关键点坐标进行分类

openpose数据集来自kaggle
https://www.kaggle.com/datasets/jorgemora/classification-of-human-poses-keypoints

由于数据集中的openpose是归一化后的，我将输入放大了10倍

训练了200代，得到了92%的准确率，可能有一点overfitting

但是可以用。


----------------------------------------------------------------------------------------

最近我给全连接层加入了BN层
但是没有什么作用