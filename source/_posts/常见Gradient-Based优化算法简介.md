---
title: 常见Gradient-Based优化算法简介
date: 2017-05-17 04:32:09
mathjax: true
tags:

- Machine Learning
- Deep Learning
- 笔记
- Gradient Descent
- Optimization

---

## 引言

自Hinton提出预训练概念以来，神经网络得到了长足的发展，从AlexNet在开始各项比赛中疯狂屠榜。当然，像多层神经网络这种参数量巨大的深度学习模型真正变得实用，还是要感谢近些年GPU运算的发展所导致的计算力提高（感谢老黄）。模型巨大化的一个结果是，基于Gradient的优化算法再一次成为主流。二阶算法虽然有不需要指定learning rate，迭代总次数少的优势，但是在参数特别多的情况下，其单步时间复杂度太高，占用内存太多。这制约了其在神经网络中的应用。

<!--more-->

导致参数量巨大化的原因除了计算力提高之外，还有数据量的巨大化（大数据？），这使得full batch learning，也就是迭代每一步都是用全部数据变得不是那么现实，而stochastic的方法，也就是每一步随机使用一部分数据变得更实际一些。这从某种意义上也制约了二阶方法例如L-BFGS的应用，因为L-BFGS在随机抽mini-batch的方式下表现不是很好。

基于Gradient的优化算法称为一阶方法，其本质上都是在Gradient Descent上做改进来避免GD上的一些问题，比如记录前步的位移来保持一定惯性，调低跑的太多/快的方向上的learning rate等等。常用的算法大概有Momentum，Nesterov Momentum，AdaGrad，RMSProp，Adam和AdaDelta。某种意义上这些算法都是启发式搜索方法，很难说谁好谁坏。Adam是其中最General的方法，也是目前比较常用的方法。

## Gradient Descent的由来

Gradient Descent的『本质』名字是Steepest Descent。顾名思义，它的核心思想就是沿最陡方向下降。

首先，用迭代方法来解优化问题的思想是：假设$w$是标量，我们想最小化$f(w)$，则存在一个足够小的$\epsilon$，满足$f(w - \epsilon f'(w)) < f(w)$，也就是我们沿着$f'(w)$迈出足够小的一步就能让目标函数值$f(w)$减小。

一元情况下只有两个方向很直接，而多元情况下，我们应该沿哪个方向才能下降？在可以下降的这些方向中，哪个方向下降的最快（Steepest Descent）呢？

函数$f(\vec{w})$在点$\vec{w}$沿某个单位方向$\vec{u}$（$||u||_2 = 1$）下降的速度，也就是说，当我们沿着$\vec{u}$走一小步$\epsilon\vec{u}$时，目标函数的变化值，可以表达为

$$
\begin{aligned}
\lim_{\epsilon \rightarrow 0} \frac{f(\vec{w} + \epsilon \vec{u}) -  f(\vec{w})}{\epsilon} = \lim_{\epsilon \rightarrow 0} \frac{f(\vec{w} + \epsilon \vec{u}) -  f(\vec{w} + 0 \vec{u})}{\epsilon}
\end{aligned}
$$

上式其实就是$f(\vec{w} + \alpha \vec{u})$关于$\alpha$在$\alpha=0$处的导数，即

$$
\begin{aligned}
\frac{\partial f(\vec{w} + \alpha \vec{u})}{\partial \alpha}|_{\alpha = 0} = \vec{u}^T \triangledown_{\vec{w}} f(\vec{w})
\end{aligned}
$$

其中偏导数$\frac{\partial f(\vec{w})}{\vec{w}} = \triangledown_{\vec{w}} f(\vec{w})$，是一个向量。函数$f(\vec{w})$在一个方向$\vec{u}$的坡度被称为方向导数，即directional derivative。

我们希望找到一个$\vec{u}$使得上式值最小，这样当我们延这个方向位移一小步，函数增量最小，也就是

$$
\begin{aligned}
argmin_{\vec{u}} \vec{u}^T \triangledown_{\vec{w}}f(\vec{w}) = argmin_{\vec{u}} ||\vec{u}||_2 ||\triangledown_{\vec{w}}f(\vec{w})||_2 cos(\theta) \\
= argmin_{\vec{u}} cos(\theta)
\end{aligned}
$$

其中$$||\vec{u}||_2=1$$，而$$||\triangledown_{\vec{w}}f(\vec{w})||_2$$与$\vec{u}$无关。

当夹角为180度的时候，该值最小，所以$\vec{u}$应该是$\triangledown_{\vec{w}}f(\vec{w})$的反方向。**所以有沿偏导数方向反方向前进，下降速度最快**的结论。这就是梯度下降的由来。所以每一步我们有更新规则

$$
\begin{aligned}
\vec{w} \leftarrow \vec{w} - \epsilon \triangledown_{\vec{w}}f(\vec{w})
\end{aligned}
$$

>注：其实我们可以将$w$看做位置，$\epsilon$看做时间，而$\triangledown f$就是速度。然而由于单步迭代的时间长短，也就是$\epsilon$是我们可以任意决定的，所以沿着当前点最快降速的方向前进只是个启发式的想法，并不一定是让函数值下降很多的好方法。
>本质上来说，Gradient Descent是一阶泰勒展开的应用，也就是说它默认了函数局部可以用一阶函数良好表达。

### 二阶方法

当我们在$\vec{w}$点处沿一个方向$\vec{u}$前进可以使函数值下降时，走多远就是个比较棘手的问题了。一阶方法中，这个步长由$\epsilon$控制。一个简单的想法是，如果沿着该方向curve down，也就是说下坡越来越陡，那么我们就多走点；如果curve up，也就是说坡度逐渐变缓，可能过一会儿就上升了也说不定，这样我们就走得不要太远比较保险。表达某点的curvature程度的的量就是该点处的二阶导数（反映一阶导数的变化趋势）。这也就是为什么二阶信息可能是有用的一个直观解释。

仍以Steepest Descent为例，如果我们将沿梯度反方向下降一小步的函数值进行二阶泰勒展开

$$
\begin{aligned}
f(\vec{w}_0 - \epsilon \triangledown f_{\vec{w}_0}) - f(\vec{w}_0) \approx - \epsilon \triangledown f^T  \triangledown f + \frac{1}{2} \epsilon^2 \triangledown f^T \mathbf{H} \triangledown f
\end{aligned}
$$

其中第一项一定为非正项，是梯度下降思想的根源。而第二项则是没有考虑到的。如果第二项过大的话，那么梯度下降就不会进行的很顺利。

下面考虑一下究竟沿着$-\triangledown f_{\vec{w}_0}$走多远为好，也就是$\epsilon$多大的问题。这就是个简单的关于$\epsilon$二次优化求极小值问题了。

注意：当

- $\triangledown f^T\mathbf{H} \triangledown f \leq 0$，函数是没有极小值的。当然，这并不意味着不存在一个最优的$\epsilon$，而是泰勒展开在距离$\vec{w}_0$太远的地方就不准确了
- 对于$\triangledown f^T \mathbf{H} \triangledown f >0$的情况，我们令导数为0：$-\triangledown f^T  \triangledown f + \epsilon \triangledown f^T \mathbf{H} \triangledown f = 0$，则有

$$
\begin{aligned}
\epsilon^* = \frac{\triangledown f^T  \triangledown f }{\triangledown f^T \mathbf{H} \triangledown f}
\end{aligned}
$$

### Newton Method

抛开一阶方法，我们可以直接从二阶泰勒展开出发，求二阶近似下，最优$\Delta \vec{w}$，使得$f(\vec{w}_0 + \Delta \vec{w}) - f(\vec{w}_0)$最小。

$$
\begin{aligned}
f(w + \Delta \vec{w}) - f(w) \approx
\triangledown f_{\vec{w}_0}^T \Delta \vec{w} + \frac{1}{2} \Delta \vec{w}^T \mathbf{H} \Delta \vec{w}
\end{aligned}
$$

求关于$\Delta \vec{w}$的导数，令其为0
$$
\begin{aligned}
\triangledown f_{\vec{w}_0} + \mathbf{H}_{\vec{w}_0} \Delta \vec{w} = 0
\end{aligned}
$$

我们有

$$
\begin{aligned}
\Delta w = - \mathbf{H}_{\vec{w}_0} ^ {-1} \triangledown f_{\vec{w}_0}
\end{aligned}
$$

因此我们得到更新规则为：

$$
\begin{aligned}
\vec{w}_t  \leftarrow \vec{w}_{t-1} - \Delta \vec{w} = \vec{w}_{t-1} - \mathbf{H}_{\vec{w}_{t-1}} ^ {-1} \triangledown f_{\vec{w}_{t-1}}
\end{aligned}
$$

这就是牛顿法。一个明显的好处是，我们不再需要选择learning rate $\epsilon$。Newton法相比一阶方法来说，收敛需要的步数明显少很多，尤其是当函数局部可以被二次函数良好近似的情况下。但是Newton法要计算Hessian matrix的逆矩阵，这个运算量和内存的要求还是比较鬼畜的，尤其是参数数目巨大的情况下。而且Newton法遇到Saddle Point比较棘手。

当然，Newton Method好使的前提也是局部能被二阶函数良好的逼近。

## More Gradient Descent

### Momentum

Momentum记录以前各个方向上累计起来的单步“位移”。这样如果在某个方向上，一直有比较恒定的小位移，这个位移就可以累计起来。而如果一个方向上符号总是发生变化，其累计位移会因为方向相反而互相抵消，变得比较小。为了不使单步步长过大，$v_{t-1}$前有一个衰减项$0 < \alpha < 1$，当$\alpha$，以“遗忘”以前累积的位移。当$\alpha = 0$时，Momentum就又退化为普通Gradient Descent了。

$$
\begin{aligned}
&\vec{s}_{t+1} \leftarrow \alpha \vec{s}_{t} + \epsilon \triangledown_{\vec{w}_{t}}\\
&\vec{w}_{t+1} \leftarrow \vec{w}_{t} - \vec{s}_{t+1}
\end{aligned}
$$

有些材料写到我们累计各个方向上的速度，这个描述是不准确的，因为实际上速度是位移关于时间的导数，所以其实$\triangledown_{\vec{w}_t}$才是物理意义上的速度。learning rate则是我们预设的单步时间。

### Nesterov Momentum

$$
\begin{aligned}
&\vec{s}_{t+1} \leftarrow \alpha \vec{s}_{t} + \epsilon \triangledown_{\vec{w}_t - \alpha \vec{s}_{t}}\\
&\vec{w}_{t+1} \leftarrow \vec{w}_{t} - \vec{s}_{t+1}
\end{aligned}
$$

观察momentum我们不难发现$$\vec{w}_{t+1} \leftarrow \vec{w}_{t} - \alpha \vec{s}_{t} - \epsilon \triangledown_{\vec{w}_{t}}$$。也就是我们是先从位置$$\vec{w}_{t}$$移动到$$\vec{w}_{t} - \alpha \vec{s}_{t}$$，然后再移动了$$- \epsilon \triangledown_{\vec{w}_{t}}$$。那么我们既然第一步已经移动到$$\vec{w}_{t} - \alpha \vec{s}_{t}$$这个位置，那么为什么下一步移动不用当前位置的梯度呢，即$$- \epsilon \triangledown_{\vec{w}_{t} - \alpha \vec{s}_{t}}$$。

这就是Nesterov Momentum的思想了。

### AdaGrad

$$
\begin{aligned}
&\vec{u}_{t+1} \leftarrow \vec{u}_{t} + \triangledown^2_{\vec{w}_t}\\
&\vec{w}_{t+1} \leftarrow \vec{w}_{t} + \frac{\vec{\epsilon}}{\sqrt{\vec{u}_{t+1}} + 1e^{-7}} \triangledown_{\vec{w}_t}
\end{aligned}
$$

AdaGrad的基本思路是对于跑的过『快』的方向，我们减小其learning rate。这里我们的learning rate是一个向量$\vec{\epsilon}$。$w$中每一个方向都有自己独立的learning rate。这里对$\vec{\epsilon}$的除法是element-wise除法。

### RMSProp

$$
\begin{aligned}
&\vec{u}_{t+1} \leftarrow \beta \vec{u}_{t} + (1-\beta) \triangledown^2_{\vec{w}_t}\\
&\vec{w}_{t+1} \leftarrow \vec{w}_{t} + \frac{\vec{\epsilon}}{\sqrt{\vec{u}_{t+1}} + 1e^{-7}} \triangledown_{\vec{w}_t}
\end{aligned}
$$

AdaGrad的问题是，由于位移的平方累计量越来越大，各个方向上对的learning rate会一直衰减，这可能会导致最后还没收敛learning rate就已经太小了。

所以RMSProp像Momentum一样，对历史位移和当前位移做了一个加权平均。这里和Momentum不一样的地方时，Momentum中两个权值的和不为1，所以可以理解为在累积过去的位移，带有一定程度的遗忘。而RMSProp中，两个权值的和位移，这从某种意义上，可以看做一种学习过程，我们想通过历史单步位移长度平方来加权当前单步位移长度得到一个可能比较好的单步位移长度平方值。所以其实是相当于用历史值对当前值做一个smoothing。

其实，如果用Momentum那种累积的模式估计也Work。

### Adam

$$
\begin{aligned}
&\vec{s}_{t+1} \leftarrow \alpha \vec{s}_{t} - (1-\alpha) \triangledown_{\vec{w}_t}\\
&\vec{u}_{t+1} \leftarrow \beta \vec{u}_{t} + (1-\beta) \triangledown^2_{\vec{w}_t}\\
&\vec{w}_{t+1} \leftarrow \vec{w}_{t} + \frac{\vec{\epsilon}}{\sqrt{\vec{u}_{t+1}} + 1e^{-7}} \vec{s}_{t+1}
\end{aligned}
$$

Adam基本可以看做Momentum和RMSProp的结合。但是其中momentum部分和上文提到的Momentum有些许不同，这里$\vec{s}_t$实际上计算的是单步期望位移，而Momentum中计算的带遗忘的累积位移。

不过，当考虑$t=0$时的第一次参数更新，有

$$
\begin{aligned}
&\vec{s}_{1} \leftarrow (1-\alpha) \triangledown_{\vec{w}_0}\\
&\vec{u}_{1} \leftarrow (1-\beta) \triangledown^2_{\vec{w}_0}\\
&\vec{w}_{1} \leftarrow \vec{w}_{0} + \frac{\vec{\epsilon}}{\sqrt{\vec{u}_1} + 1e^{-7}} \vec{s}_1
\end{aligned}
$$

由于$\vec{s}_0, \vec{u}_0$初始化为0，所以最初几步（$t$很小的时候），导数和导数平方都会被“衰减”，所以在最初几步可以做一个Bias Correction：

$$
\begin{aligned}
&\vec{sc}_{t+1} = \frac{\vec{s}_t}{1 - \alpha^t} \\
&\vec{uc}_{t+1} = \frac{\vec{u}_t}{1 - \beta^t} \\
&\vec{w}_{t+1} \leftarrow \vec{w}_{t} + \frac{\vec{\epsilon}}{\sqrt{\vec{uc}_t} + 1e^{-7}} \vec{sc}_t
\end{aligned}
$$

当$t$增大后，修正项就不起什么作用了。

## 参考资料

<https://en.wikipedia.org/wiki/Newton%27\vec{s}_method_in_optimization>
<http://www.deeplearningbook.org/>
<http://cs231n.github.io/optimization-1/>
<http://cs231n.github.io/neural-networks-3/#ada>
