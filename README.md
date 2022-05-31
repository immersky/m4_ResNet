学习和复现ResNet

原始论文:https://arxiv.org/pdf/1512.03385.pdf

改进论文:https://arxiv.org/pdf/1603.05027.pdf

主要参考了https://blog.csdn.net/qq_45649076/article/details/120494328,https://zhuanlan.zhihu.com/p/72679537,https://zhuanlan.zhihu.com/p/31852747/

代码实现部分参考https://blog.csdn.net/frighting_ing/article/details/121324000

# ResNet特点

## 引入残差构成块:

一个残差构成块有两条路径 F ( x )和 x，F ( x ) 路径被称为residual mapping，x 路径被称为identity mapping或者 shortcut，⨁  表示相加，要求 F ( x ) 与 x  的尺寸相同



![shortcutconnections](https://user-images.githubusercontent.com/74494790/171155241-c9da3ab6-37e7-45c4-8780-f242f7ca3187.png)


对于一个神经网络的结构块，假设想要模拟的函数是H(x)，理想情况下希望输入x，输出H(x)，引入F(x)=H(x)-x,即F(x)+x=H(x),而上图中可见，在网络块中引入了一条支路直接把输入网络块前的x块模拟后的结果相加，正旨在让网络通过模拟F(x)+x来拟合H(x) 而不是直接模拟H(x),在极端情况下，网络块模拟的F(x)为0，至少也是个恒等映射，网络性能不会变差,网络深度得以继续变深。论文还提到，除了保证不变差的情况下，这种结构能够更好的拟合最终函数，举个例子:$H(5)=5.1=F(5)+5$,则$F(5)=0.1$,假设改变对5的映射使输出变化为5.2,则F 需要将映射的输出增加100%，这需要对权重更大幅度的改变（相对于使用传统结构直接拟合H(5)的1-5.2/5.1*100%），可见新的结构对权重调整较大。

用数学公式描述残差块

假设残差块输入x，输出y，有

$y=F(x,\{W_i\})+x$


其中$F(x,\{W_i\})$表示残差块想要拟合的函数(residual mapping to be learned)，比如上文中Figure2里面残差块有两层，则$F=W_2\sigma(W_1x)$，其中$\sigma$表示ReLu，接下来为了确保维数相同，可以给让x通过1x1卷积层，这时公式如下

$y=F(x,\{W_i\})+W_sx$

注意:如果残留块部分(residual mapping)只有一层，公式退化为

$y=W_1x+x$，整个结构退化成了一个普通的神经网络中间层，论文提到,这样没有就没有任何多余好处了。

反向传播求梯度时可以发现(下图)，对恒等映射求x偏导直接为1，而对另一函数求则结果不可能为-1,这样避免了梯度消失

![fanxaing](https://user-images.githubusercontent.com/74494790/171155296-6ff9c92a-6e08-492f-9fd5-4fe6b2a2e3b5.png)


另外，初始论文中提到了残差构建块的两种结构,有bottleneck和无bottleneck,bottleneck结构为右图使用1x1卷积核降维再升维，如同张量流入“瓶颈”

![shortcutkinds](https://user-images.githubusercontent.com/74494790/171155324-e1eb44a4-bd00-458c-9bd7-993e8783696d.png)





在后续一篇论文中，对于identity mapping有更深入的比较和研究

https://arxiv.org/pdf/1603.05027.pdf

主要是将shortcut的x从原分不动加入residual mapping后的结果，改为运用函数映射并分为多种（如下图）并讨论，实验,比如让h(x)的映射不再是恒等,比如成为$x_{l+1}=\lambda_lx_l+F(x_l,W_l)$,


![new](https://user-images.githubusercontent.com/74494790/171155339-a61db23d-be5d-43be-8770-8ecaf5e9a670.png)



另外，可以用下图直观的感受以下short cut对网络梯度下降的作用，网络越深，若容易落入局部最优，但short cut让error surface更加平滑，从而更容易到达全局最优

error surface图

![error surface](https://user-images.githubusercontent.com/74494790/171155360-950bc762-e623-4477-86de-63a2cf6dbc41.png)


## 使用BN(batch normalize)

且不使用dropout

论文提到，在每次卷积后，激活前都是用BN。

BN让数据满足均值为0，方差为1的分布

这个应该处理后使数据向正态分布靠齐,看其算法
![BN](https://user-images.githubusercontent.com/74494790/171155377-7fea677d-86c6-4735-8b3a-eded6e01ad45.png)





# ResNet结构

论文提到，在vgg的结构中加入shortcut并加深网络深度，结构和详细结构如下

![consturction](https://user-images.githubusercontent.com/74494790/171155413-577c0539-06ce-4857-9f6c-99e5a2c13d2b.png)



下图中,  x3意味着这层残差块有三个，其它同理
![structure](https://user-images.githubusercontent.com/74494790/171155433-b071dee1-09a3-4bd2-a3a3-220070081498.png)


如果取上图中18layer参数，则网络结构如下

![18](https://user-images.githubusercontent.com/74494790/171155453-df9a6f2b-9b85-4c6f-ab94-f25680a43d5c.png)


可见resnet有很多重复结构，如果按照之前AlexNet,VGG的复现方式，可以一行一行慢慢构建，比较简单，但是我参考了网上流传的博客，都是运用循环结构构建网络，使用一个list，比如layers=[]，不断往里放入nn.Module类对象，然后直接转换为nn.Sequential:`nn.Sequential(*layers)`,就可以按照nn.Sequential使用了

