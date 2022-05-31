import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms


#DEVICE=torch.device('cuda'if torch.cuda.is_available() else 'cpu')          #转gpu
DEVICE=torch.device( 'cpu')          #转cpu
print(DEVICE)
EPOCH=2
BATCH_SIZE=256

#实现resnet，输入为224x224x224



# 3x3卷积模块
def conv3x3(in_planes, out_planes, stride=1):                           #输入通道数，输出通道数，步长未定
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# basicblock，对应readMe中Table1中间部分每个 {  }xn的部分，指这个块有几个，{ }内即为basicblock具体结构
#注意表中{ }xn的{ }内结构有些是有bottleneck，即先用1x1卷积核降维再生升维的结构，这个就不归到下面实现的basicblock
#而是另外创一个bottleneck 类
#注意到Table1中，论文指出的renet 18,32层用的是下面时实现的basicblock,而50,101,153层则每个基本块都用的是bottleneck块
#注意，basicblock和bottleneck块只是自己实现时对类的称呼,引用了论文内相关结构的名称
class BasicBlock(nn.Module):
    expansion = 1                                       #expansion指通过这个块后，维数的变化倍率
    def __init__(self, inplanes, planes, stride=1, changeConv=None): 
    # inplanes代表输入通道数，planes代表输出通道数。，stride只调整第一层卷积步长
        super(BasicBlock, self).__init__()
        # Conv1
        self.conv1 = conv3x3(inplanes, planes, stride)  #3x3卷积
        self.bn1 = nn.BatchNorm2d(planes)               #论文提到，每次经过卷积，在经过激活函数前进行batch normalize
        self.relu = nn.ReLU(inplace=True)               #经过ReLu
        # Conv2
        self.conv2 = conv3x3(planes, planes)            
        self.bn2 = nn.BatchNorm2d(planes)               #接下来，在通过relu前，要将原始的x调整维数和大小后直接加到结果
        # 如果按顺序经过上边的变换后.输入x和结果维数，大小不一样，就需要用1x1卷积变换维度，大小
        self.changeConv = changeConv                   #传入降维,改大小用的卷积层

    def forward(self, x):
        residual = x                                    #保存下输入网络前的张量，后续直接加到输出结果

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.changeConv is not None:
            residual = self.changeConv(x)
		# F(x)+x
        out += residual                                 #这里直接和residual相加
        out = self.relu(out)                            #论文提到，先相加，再通过激活层

        return out

#下面这个类则是加入Bottleneck结构，先降维再升维
class Bottleneck(nn.Module):
    expansion = 4                                       # 降维后，升维的倍乘,table1给出的所有bottleneck结构，都是升到4倍

    def __init__(self, inplanes, planes, stride=1, changeConv=None):  
        #inplanes,planes分别为输入通道数，降维后的通道数，最终输出的维度是降维后维度*expansion，步长为3x3卷积的步长
        super(Bottleneck, self).__init__()
        # conv1   1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)         #先使用1x1卷积核降维,这样网络参数就变少了
        self.bn1 = nn.BatchNorm2d(planes)                                           #每次卷积后要经过BN
        # conv2   3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,        #再进行卷积
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)                           
        # conv3   1x1  
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)       #再用1x1卷积核升维为原来的四倍
        self.bn3 = nn.BatchNorm2d(planes * 4)                                       
        self.relu = nn.ReLU(inplace=True)
        self.changeConv = changeConv
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.changeConv is not None:                                             #如果需要降维或改大小，传入对应1x1卷积层
            residual = self.changeConv(x)                                           #于此使用

        out += residual                                                             #先加再经过relu
        out = self.relu(out)

        return out


#接下来实现ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  
    # layers：4个元素的list,每一个为数字，分别表示resnet内部四层残留块层每层的残留块数量
    #  block即选择上面两个类其中一个作为残差块构建整个网络
        self.inplanes = 64 
        super(ResNet, self).__init__()
        # 1.conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)    #参照table1,一步一步来，先做卷积
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 2.conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                 #卷积后特征图尺寸从112变为56
        self.layer1 = self._make_layer(block, 64, layers[0])                            #残留块层的构造，见对应函数
        # 3.conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)                 #后面三个残留块层内的卷积步长为2，见表格
        # 4.conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 5.conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)                                                  #见table1,使用平均池化尺寸降为1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)                         #然后全连接

		## 显式初始化权重
  #      for m in self.modules():
  #          if isinstance(m, nn.Conv2d):
  #              n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
  #              m.weight.data.normal_(0, math.sqrt(2. / n))
  #          elif isinstance(m, nn.BatchNorm2d):
  #              m.weight.data.fill_(1)
  #              m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        #构造残留块层block:采用的是上面实现的basic类还是bottleneck类
        changeConv = None
        if stride != 1 or self.inplanes != planes * block.expansion:            
            #如果这层的步长不为1，或者是bottleneck结构_，则第一个残留块内部主路径的输出特征图较输入前size或dimension会变化，
            #则创建一个卷积层让输入张量在shotcut路径变换后再在第一个残留快中加入主路经
            changeConv = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []                                                     #用于存储网络对象的列表
        layers.append(block(self.inplanes, planes, stride, changeConv))
        # 放入第一个残留块
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))   
        # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)                                   #列表里的顺序nn.Module对象转为nn.Sequential对象

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # 将输出结果展成一行
        x = self.fc(x)

        return x


normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
#训练集
path_1=r'D:\imagedb\imageC&D\train_0'
trans_1=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize,
])
 
#数据集
train_set=ImageFolder(root=path_1,transform=trans_1)
#数据加载器
train_loader=torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,
                                         shuffle=True,num_workers=0)
print(train_set.classes)
 
#测试集
path_2=r'D:\imagedb\imageC&D\train_0'
trans_2=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize,
])
test_data=ImageFolder(root=path_2,transform=trans_2)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,
                                        shuffle=True,num_workers=0)
 
#验证集
path_3=r'.D:\imagedb\imageC&D\train_0'
valid_data=ImageFolder(root=path_2,transform=trans_2)
valid_loader=torch.utils.data.DataLoader(valid_data,batch_size=BATCH_SIZE,
                                         shuffle=True,num_workers=0)
 
#定义模型
model=ResNet(BasicBlock,[2,2,2,2],2).to(DEVICE)
#优化器的选择
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)
 
 
#训练过程
def train_model(model,device,train_loader,optimizer,epoch):
    train_loss=0
    model.train()
    for batch_index,(data,label) in enumerate(train_loader):
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,label)
        loss = loss 
        loss.backward()
        optimizer.step()
        if batch_index%300==0:
            train_loss=loss.item()
            print('Train Epoch:{}\ttrain loss:{:.6f}'.format(epoch,loss.item()))
 
    return  train_loss
 
 
#测试部分的函数
def test_model(model,device,test_loader):
    model.eval()
    correct=0.0
    test_loss=0.0
 
    #不需要梯度的记录
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            output=model(data)
            test_loss+=F.cross_entropy(output,label).item()
            output = torch.softmax(output, dim=1)                                           #softmax
            pred=output.argmax(dim=1)
            correct+=pred.eq(label.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print('Test_average_loss:{:.4f},Accuracy:{:3f}\n'.format(
            test_loss,100*correct/len(test_loader.dataset)
        ))
        acc=100*correct/len(test_loader.dataset)
 
        return test_loss,acc
 
 
#训练开始
list=[]
Train_Loss_list=[]
Valid_Loss_list=[]
Valid_Accuracy_list=[]
 
#Epoc的调用
for epoch in range(1,EPOCH+1):
    #训练集训练
    train_loss=train_model(model,DEVICE,train_loader,optimizer,epoch)
    Train_Loss_list.append(train_loss)
    torch.save(model,r'.\model%s.pth'%epoch)
 
    #验证集进行验证
    test_loss,acc=test_model(model,DEVICE,valid_loader)
    Valid_Loss_list.append(test_loss)
    Valid_Accuracy_list.append(acc)
    list.append(test_loss)
 
#验证集的test_loss
 
min_num=min(list)
min_index=list.index(min_num)
 
print('model%s'%(min_index+1))
print('验证集最高准确率： ')
print('{}'.format(Valid_Accuracy_list[min_index]))
 
#取最好的进入测试集进行测试
model=torch.load(r'.\model%s.pth'%(min_index+1))
model.eval()
 
accuracy=test_model(model,DEVICE,test_loader)
print('测试集准确率')
print('{}%'.format(accuracy))
 
 
#绘图
#字体设置，字符显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
 
#坐标轴变量含义
x1=range(0,EPOCH)
y1=Train_Loss_list
y2=Valid_Loss_list
y3=Valid_Accuracy_list
 
#图表位置
plt.subplot(221)
#线条
plt.plot(x1,y1,'-o')
#坐标轴批注
plt.ylabel('训练集损失')
plt.xlabel('轮数')
 
plt.subplot(222)
plt.plot(x1,y2,'-o')
plt.ylabel('验证集损失')
plt.xlabel('轮数')
 
plt.subplot(212)
plt.plot(x1,y3,'-o')
plt.ylabel('验证集准确率')
plt.xlabel('轮数')
 
#显示
plt.show()