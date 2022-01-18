## StyleGAN-with-jittor
### 一、项目内容
* * *
这个项目是用计图（jittor）实现的StyleGAN。主要目的是参考论文《A Style-Based Generator Architecture for Generative Adversarial Networks》，通过实现StyleGAN论文中的模型，能够生成64 × 64的彩色符号图像，并且展示隐空间插值效果。  
### 二、模型实现
* * *
![图 1](https://user-images.githubusercontent.com/77678715/149555899-6b44d591-b8a6-4bb7-a668-136a754c17cb.png)
##### 图1
#### （一）网络结构：
如图1所示为StyleGAN的完整结构，包括了生成器和判别器两部分。判别器与ProGAN中结构一致，论文主要对生成模型进行了改进，生成模型的输入为学习到常量，输入隐向量经过一个8层的全连接多层感知机映射为风格权重向量，并通过全连接模块A转换成每个通道的比例和偏差，通过AdaIN实现对图像特征的控制。此外，通过引入噪声，实现对图像细节特征的影响。  
#### （二）程序实现：
如图1所示，StyleGAN包括生成网络和判别网络，其中的训练过程采用了ProGAN的方式。本实验参考StyleGAN的pytorch版本进行jittor版本的实现，模型和训练过程只有少量jittor和pytorch框架差异需要做简单调整。在训练过程中增加了对中间过程模型的保存，以期在程序出错后可以使用最近保存的中间模型进行参数加载和继续训练。本次实验不同于pytorch只对生成模型和判别网络的参数进行了保存。
对于隐向量插值的实现，本实验随机生成四个隐编码分别对应隐向量插值的四个角的隐编码。执行时使用线性插值生成第一列和最后一列的隐编码，对于同一行同样使用线性插值生成中间的隐编码，并将这些隐编码输入合成网络（生成器）生成彩色符号。对应的实验结果可以参看实验结果部分图3。噪声影响部分，需要根据待生成图像的分辨率生成多个不同长度正态分布的噪声，噪声对生成图像的影响请参看实验结果部分图5。
### 三、实验结果
* * *
如图2所示是通过训练过程中得到的部分字符的结果。模型训练从8 * 8的分辨率以尺寸增大以备的方式逐步增长至64 * 64。从下图可以看到随着训练步数的增加，生成的彩色字符也越来越清晰了，而且风格与训练集更接近。  
8 * 8： ![image](https://user-images.githubusercontent.com/77678715/149668920-16a15db2-f7a1-4939-b853-3568be93dd4e.png)    
16 * 16：![image](https://user-images.githubusercontent.com/77678715/149668924-5839b042-c32a-43e8-9f74-717faa0ec448.png)  
32 * 32： ![image](https://user-images.githubusercontent.com/77678715/149668928-2d6af3bd-689a-4474-bdb8-cf9d8167a462.png)  
64 * 64： ![image](https://user-images.githubusercontent.com/77678715/149668933-6e138a6e-b0c9-438a-985b-d369184a99e7.png) 
##### 图2 模型训练过程输出的生成字符  
 ![image](https://user-images.githubusercontent.com/77678715/149668984-252fddf5-359e-4a89-9acd-90cd80a0ee98.png) 
##### 图3 隐向量差值的结果  
图3为隐向量差值的结果（分辨率为32 * 32）。从图中可以看出除右下角外，其他行的渐变效果较好。
![image](https://user-images.githubusercontent.com/77678715/149668992-dd6235c4-1579-4b07-93a4-d096b6444201.png)  
##### 图4 Mixing的结果  
图4为Mixing结果（分辨率为32 * 32）。原论文中的Mixing时，不同latent code对生成人脸的肤色、性别等不同属性发挥作用。从图4中也可以观察到相似的结论，生成符号的整体形状明显受到上方的符号影响，而颜色与左侧符号更接近。  
![image](https://user-images.githubusercontent.com/77678715/149669000-31a56492-1a8a-4191-8b02-f88284f6072f.png) 
##### 图5 注入不同噪声时生成的符号结果  
图5为噪声对生成符号的影响结果，图中每一行为输入隐编码相同情况下注入不同噪声时生成的彩色符号。根据原文结论，噪声仅对一些细节特征产生影响，因彩色符号本身较简单，从图5可看到噪声对大部分生成符号影响较小。
用户在使用中，可对超参数进行调整，以获得更好的训练结果。
### 四、项目文件主要结构和基本用法
* * *
源程序中主要包括了以下几部分，在使用中可以按照如下顺序进行执行：
* prepare_data.py是准备lmdb数据集
    > 在使用时可以通过：python prepare_data.py --out LMDB_PATH --n_worker N_WORKER DATASET_PATH
    
* train.py 是对模型进行训练的脚本
    >在使用时可以用过：python train.py lmdb

* generate.py 是利用训练好的模型生成隐向量差值的结果，Mixing的结果和注入不同噪声时生成的符号结果。
* model.py 定义了 StyleGAN 中生成器（Generator）与判别器的（Discriminator）的结构。
* 另外checkpoint 文件夹用于保存训练好的模型文件；sample 文件夹用于保存模型在训练过程中采样的中间图像结果；results 文件夹用于保存 generate.py 生成的隐空间插值等结果。
* 实验中使用的字符数据集一共有近7k张。1080Ti的显卡上训练一天也基本上会获得较好的训练结果。
### 五、主要参考资料及致谢
* * *
本项目主要基于jittor开发。关于计图（jittor）的主要内容，请参考：
* [Jittor\(计图\): 即时编译深度学习框架](https://cg.cs.tsinghua.edu.cn/jittor/)。

另外，项目主要参考的pytorch版本的StyleGAN实现，主要参考内容如下：
* [pytorch版本的StyleGAN](https://github.com/rosinality/style-based-gan-pytorch)。

同时非常感谢助教对项目的大力支持，因为根据助教提供的参考资料，对本项目实现有很大帮助。
除此之外，还需要感谢以下参考资料：
>[1] Karras T, Laine S, Aila T. A style-based generator architecture for generative adversarial networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 4401-4410.  
>[2] <u>http://www.seeprettyface.com/research_notes.html#step3</u>  
>[3] <u>http://www.gwylab.com/pdf/Note_StyleGAN.pdf</u>  
>[4] <u>https://github.com/xUhEngwAng/StyleGAN-jittor</u>
### 六、许可
* * *
本项目中由贡献者编写的文件在不做特殊说明的情况下使用 MIT LICENSE 开源。这意味着您可以任意使用、拷贝、修改、出版以及将本项目用于商业用途，但是在所有基于本项目的拷贝及其衍生品中都必须包含此开源许可证。

其余部分的版权归属各自的作者。
