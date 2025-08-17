# 映射Layer和函数说明

抽象的函数和Layer接口兼容torch对应的操作，有三类操作，第一类是有权重的layer，第二类没有权重的Layer, 第三类函数。

使用方式

```python
# Layers
import trtlite.nn as nn
# functions
import trtlite.nn.functional as F
```

## 1. 神经网络层

### 卷积层

- **`Conv2d`** - 二维卷积层
- **`Conv1d`** - 一维卷积层
- **`ConvTranspose2d`** - 转置卷积层（未完成）

### 全连接层
- **`Linear`** - 线性层

### 归一化层
- **`BatchNorm2d`** - 二维批量归一化
- **`BatchNorm1d`** - 一维批量归一化

### 量化层
- **`TensorQuantizer`** - 张量量化器

## 2. 无参数Layer

### 池化层
- **`MaxPool2d`** - 二维最大池化
- **`AvgPool2d`** - 二维平均池化
- **`AdaptiveAvgPool2d`** - 自适应平均池化

### 填充层
- **`ZeroPad2d`** - 零填充

### 激活层
- **`ReLU`** - ReLU激活层
- **`LeakyReLU`** - LeakyReLU激活层

### 上采样层
- **`Upsample`** - 上采样层


## 3. 函数

### 激活函数
- **`activation`** - 通用激活函数
- **`softmax`** - Softmax激活函数
- **`leaky_relu_dla`** - DLA兼容的LeakyReLU
- **`relu`** - ReLU激活函数
- **`sigmoid`** - Sigmoid激活函数
- **`tanh`** - Tanh激活函数
- **`elu`** - ELU激活函数
- **`selu`** - SELU激活函数
- **`softsign`** - Softsign激活函数
- **`softplus`** - Softplus激活函数
- **`clip`** - Clip激活函数
- **`hard_sigmoid`** - HardSigmoid激活函数
- **`scaled_tanh`** - ScaledTanh激活函数
- **`thresholded_relu`** - ThresholdedReLU激活函数
- **`leaky_relu`** - LeakyReLU激活函数

### 一元运算函数

- **`unary(x: ITensor, op: str)`** - 通用一元运算
- **`exp`** - 指数函数
- **`log`** - 对数函数
- **`sqrt`** - 平方根函数
- **`recip`** - 倒数函数
- **`abs_`** - 绝对值函数
- **`neg`** - 取负函数
- **`sin`** - 正弦函数
- **`cos`** - 余弦函数
- **`tan`** - 正切函数
- **`sinh`** - 双曲正弦函数
- **`cosh`** - 双曲余弦函数
- **`asin`** - 反正弦函数
- **`acos`** - 反余弦函数
- **`atan`** - 反正切函数
- **`asinh`** - 反双曲正弦函数
- **`acosh`** - 反双曲余弦函数
- **`atanh`** - 反双曲正切函数
- **`ceil`** - 向上取整函数
- **`floor`** - 向下取整函数
- **`ref`** - 误差函数
- **`not_`** - 逻辑非函数

### 二元运算函数

input要求都是tensor

- **`elementwise_op`** - 逐元素运算
- **`sum_`** - 逐元素加法
- **`prod_`** - 逐元素乘法
- **`max_`** - 逐元素最大值
- **`min_`** - 逐元素最小值
- **`sub_`** - 逐元素减法
- **`div_`** - 逐元素除法
- **`pow_`** - 逐元素幂运算
- **`floor_div_`** - 逐元素整除
- **`and_`** - 逐元素逻辑与
- **`or_`** - 逐元素逻辑或
- **`xor_`** - 逐元素逻辑异或
- **`equal_`** - 逐元素相等比较
- **`greater_`** - 逐元素大于比较
- **`less_`** - 逐元素小于比较


### 数学运算函数

- **`add(input, other, *, out=None)`** - 张量加法, other支持tensor or 常量
- **`mul(input, other, *, out=None)`** - 张量乘法, other支持tensor or 常量
- **`max(input, dim=None, keepdim=False, *, out=None)`** - 最大值
- **`gather(input, dim, index, *, sparse_grad=False, out=None)`** - 索引选择
- **`slice(input, shape, start=None, stride=None, dim=None)`** - 张量切片
- **`arange(length, dtype=np.int32)`** - 生成数值序列

### 池化函数

- **`avg_pool`** 平均池化

### 张量操作函数

- **`padding_nd`** - N维填充
- **`padding`** - 2D填充
- **`pad`** - 通用填充
- **`cat`** - 张量拼接
- **`interpolate`** - 插值
- **`squeeze(input, dim)`** - 压缩维度
- **`unsqueeze(input, dim)`** - 扩展维度
- **`split`** - 张量分割
- **`view`** - 重塑张量
- **`reshape`** - 重塑张量（view的别名）
- **`permute(input, dims)`** - 张量转置
- **`flatten(input, start_dim=0, end_dim=-1)`** - 张量展平
- **`transpose(input, dims0, dims1=None)`** - 张量转置

### 类型转换函数

- **`identity(input, dtype=None)`** - 恒等变换

### 量化相关函数

- **`add_ops_quantize`** - 量化操作
- **`add_ops_dequantize`** - 反量化操作

### 常量和实用函数

- **`add_constant`** - 添加常量
- **`add_tile`** - 张量平铺
