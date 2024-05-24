# xcosx_webshell_detect

xcosx_webshell_detect 是一个基于机器学习的webshell检测工具，使用机器学习算法进行训练，可以检测各种类型的webshell。


# 使用

先进行样本处理，将样本处理成csv格式，然后使用机器学习算法进行训练生成模型，训练完成后，即可使用模型进行检测。

## 特征提取

将webshell放到sample/black下，正常脚本语言在white下，运行sample/main.go进行特征提取，生成train.csv

## 超参数调优

超参数调优，使用trainer/tarin.py进行调优，采用网格搜索（枚举法）找到范围内的超参数最优值。

## 训练

根据最优值进行训练模型，使用trainer/main.go进行训练，生成model.json

放入xcosx中即可调用进行webshell检测

---

> 特征提取和训练过程采用的下面大佬的代码 
> https://github.com/LinanV/ai-webshell-scanner