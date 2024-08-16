---
title: Support Vector Machine笔记
mathjax: true
tags:
- Machine Learning
- 笔记
- SVM
---

## Maximum Margin Classifier

考虑一个简单的二类分类问题，如果数据点集合$x_i, i = 1...N, x_i \in R^n$ 能被$R^n$空间中一个超平面$w^Tx+b$分开，即一侧是正样本，一侧是负样本，则我们称这些数据点是线性可分的。但是能分开这些数据点的超平面很可能有无数个，我们需要从中挑出我们认为将这些点分开的最好最合理的那个平面。

Support Vector Machine（SVM）就是提供了一种关于从能分开正负样本的超平面中选择最好的超平面的“偏好”或者说标准。它认为离两类数据点“最远”，Margin最大的那个平面是最好的，所以SVM又被称为Maximum Margin Classifier。这种偏好并非像其他模型参数一样是从数据中学习到的，而是我们用自己分析出的结果，算是一种先验知识。SVM的最大Margin的思路是，数据中的随机噪声和扰动甚至错误都是很常见甚至难以避免的，如果一个超平面离两类数据点都比较远，那么它受到这种扰动干扰的几率也就越小，进而在新数据点上预测的泛化效果应该也就越好。

<!--more-->

我们不难将SVM的这个Maximum Margin的偏好公式化为：

{% mathjax %}
\begin{aligned}
w^*, b^* = argmax_{w,b} d_{min} \\
s.t. d_i \geq d_{min}
\end{aligned}
{% endmathjax %}

其中$d_i$是数据点$x_i$距离超平面$wx+b=0$的距离。

为了推导方便，我们不失一般性的规定，正样本的标记$y_i=+1$，其在分割平面的$w^Tx+b>0$一侧，而负样本的标记为$y_i=-1$，在$w^Tx+b<0$一侧。

下面我们要搞清楚的是$d_i$如何表达成$w, b$的形式。假设$x_i$在超平面上的投影的点是$x’_i$，那么因为$w$是垂直于超平面的，所以有

$$x_i = x’_i + d_i y_i \frac{w}{||w||}$$

 其中 $y_i \frac{w}{||w||}$是相对超平面而言指向$x_i$一侧，垂直于超平面的单位向量。
 
之后我们有

{% mathjax %}
\begin{aligned}
& d_i y_i w = (x_i - x’_i) ||w||\\
& d_i y_i w^T w = w^T(x_i - x’_i) ||w||\\
& d_i y_i ||w||^2 = (w^Tx_i - w^Tx’_i) ||w||
\end{aligned}
{% endmathjax %}

因为$x’_i$在平面上，所以有$w^Tx’_i = -b$，因而

$$d_i = \frac{(w^T x_i +b) y_i}{||w||}$$

我们将优化问题中$d_i$替换成$w, b$的形式有

{% mathjax %}
\begin{aligned}
& w^*, b^* = argmax_{w,b} d_{min}\\
& s.t. (w^T x_i +b) y_i \geq d_{min} ||w||
\end{aligned}
{% endmathjax %}

现在我们虽然可以根据这个标准求得最优超平面，但是因为$\{a w, a b| \forall a \neq 0\}$定义的是**同一个**超平面，我们需要加条件来固定$w$和$b$，以从无数$w,b$的组合中挑一个。这里我们可以选取$||w|| = c$或者{% mathjax %}d_{min}||w|| =c{% endmathjax %}。取第二种的时候我们可以将$d_{min}$这个量从优化问题中消灭掉。当取$c=1$时，我们有

{% mathjax %}
\begin{aligned}
& w^*, b^* = argmax_{w,b}\frac{1}{||w||}\\
& s.t. (w^T x_i +b) y_i \geq 1
\end{aligned}
{% endmathjax %}

当我们得到这个超平面之后，有一部分数据点满足$(w^T x +b) y =1$，他们是离超平面最近的点的集合，也就是在Margin平面上，它们最终决定了我们得到的超平面，也可以说是支撑着这个超平面，所以它们被称为**支持向量**，这就是支持向量机这个名字的来由。另一部分数据点则仅仅满足$(w^T x_i +b) y_i >1$，实际上它们$(w^T x_i +b) y_i$具体等于8还是200还是100万都对决定我们的超平面无关紧要，甚至可以被抛弃掉。这一点后面还会讲到。

回到上面的优化问题，我们可以进一步将其整理为常见的、好优化的等价二次形式

{% mathjax %}
\begin{aligned}
&w^*, b^* = argmin_{w,b} ||w||^2\\
&s.t. (w^T x_i +b) y_i - 1 \geq 0, \forall i = 1...n
\end{aligned}
{% endmathjax %}

这个问题可以用Quadratic Programming来解决。但是这个分类器不能处理数据点线性不可分的情况，并不实用。

针对这种情况我们有两个方案
1. Basis Expansion: $x \rightarrow \phi(x)$，例如，假设数据点有2个features，那么我们可以将其转为5维$(x_1, x_2) \rightarrow (x_1, x_2, x_1^2, x_2^2, x_1x_2)$。但是二阶的时候变量数目就从2上升到5，如果feature数目多的话，升维之后维度数目“不堪设想”。这会使训练和预测速度也会受到影响，且其实也不能保证高维空间里就线性可分了；
2. 第二个方法是我们放宽优化问题中的要求，上面的SVM优化形式又被称为Hard Margin SVM，这是因为其要求$s.t. (w^T x_i +b) y_i \geq 1, \forall i = 1...n$，如果我们给其加一个松弛变量$\xi_i \geq 0$放松一下要求，$s.t. (w^T x_i +b) y_i \geq 1 - \xi_i, \forall i = 1...n$，那么我们就得到了Soft Margin SVM。

## Soft Margin SVM
$\xi_i$是我们新引入的松弛变量，可以和$w, b$一起作为模型参数从数据中学习处。当然如果我们仅仅是限制其$\xi_i \geq 0$，不改变优化问题的话，显然$\xi = +\inf$可以使所有条件都满足/失效，这种情况下优化问题就退化成了无约束的形式，如下

{% mathjax %}
\begin{aligned}
&w^*, b^* = argmin_{w,b} ||w||^2\\
&\rightarrow w^* = 0
\end{aligned}
{% endmathjax %}


这显然不靠谱。直觉来说，我们可以对某些点的要求作出宽限，允许其进入Margin范围内，但是这种越界行为应该越少越好，能不出现就不出现。根据这一想法，我们将优化目标加一个关于松弛变量惩罚项，来保证我们的模型“能不放松就不放松”：

{% mathjax %}
\begin{aligned}
&w^*, b^* = argmin_{w,b} ||w||^2 + C \sum_i \xi_i\\
&s.t. (w^T x_i +b) y_i  \geq 1 - \xi_i, \forall i = 1...n\\
&\xi_i \geq 0, \forall i = 1...n
\end{aligned}
{% endmathjax %}

其中$C$是人为指定的一个超参数，可以用交叉检验来确定其最优值。这就是Soft Margin SVM的最终形式。

### Linear Classifier with Hinge Loss
人们常说SVM就是线性分类器+Hinge Loss，Logistics Regression就是线性分类器+Cross Entropy。实际上这里线性分类器+Hinge Loss = SVM指的是Soft Margin SVM。

整理上面的优化问题的第一个约束条件约束条件，我们有

$$s.t. \xi_i   \geq 1 - f(x_i) y_i, \forall i = 1...n$$

又因为我们希望$\xi_i$越小越好，且第二个约束条件是$\xi_i \geq 0, \forall i = 1...n$，我们可以将两个约束条件结合为

$$s.t. \xi_i  = max(1 - f(x_i) y_i, 0), \forall i = 1...n$$

带入原优化问题，我们得到一个无约束优化问题

{% mathjax %}
\begin{aligned}
w^*, b^* = argmin_{w,b} ||w||^2 + C \sum_i max(1 - f(x_i) y_i, 0)
\end{aligned}
{% endmathjax %}

其中第一项就是我们熟知的Regularization项，第二项则是每个数据点的Hinge Loss。多说一句，关于Hinge Loss和Cross Entropy Loss，它们为什么make sense呢？其实是因为它们都是最Naive的0-1 Loss $I_{y_i =sign(f(x_i))}$的近似，0-1 Loss太难优化了。


### 问题依旧

然而即使是通过松弛变量宽限成Soft Margin SVM，如果正负样本在当前空间内就是无法很好地用超平面分割开，再怎么宽限也帮助不大。这时我们还是要将其尝试映射到高维/其它空间看看能不能分开得好一些。然而上面说过，升维之后会导致计算量增大，这就需要Kernel出马了。在介绍如何将SVM与Kernel之前，我们得先介绍Lagrange Multiplier以及对偶问题。因为只有在利用Lagrange Multiplier转换了优化问题形式之后，再转成其对偶形式后，才能使用Kernel。（除了能使用Kernel外，还有其他的一些好处。）

## 拉格朗日乘数与对偶问题

我们可以用拉格朗日乘数来进一步将这个优化问题进行转化，然后进一步转化成其对偶为题，这样做有很多（潜在的）好处：
1. 有更高效的优化方法
2. 我们将 feature-wise weights $w$转成了sample-wise weights $a$，其非零项数目为支持向量数目，如果支持向量数目远少于feature数目的话，最后的参数数会减少
3. 可以使用Kernel，这在一定程度上解决数据点线性不可分的问题，同时不需要我们提前显式地将features $x$做一些basis expansion转化成$\phi(x)$ （比如 $(x_1, x_2) \rightarrow (x_1, x_2, x_1^2, x_2^2, x_1 x_2)$），这样可以避免升维带来的优化速度降低问题

下面我们仍以Hard Margin SVM为例子，来介绍如何使用Lagrange Multiplier来将约束条件融合到优化目标中，以及如何转换成其对偶问题并求解。Soft Margin SVM的推导大同小异。

首先我们构造拉格朗日函数，即将约束条件“加乘”到优化目标中

$$L(w,b,a) = \frac{1}{2} ||w||^2 - \sum_i a_i ((w^T x_i +b) y_i - 1)$$

$$a_i \geq 0, \forall i = 1...n$$

然后我们构造一个Dual Function：

$$g(a) = min_{w, b} L(w, b, a)$$

对于满足约束条件的$w,b, a$，设原目标函数为$p(w, b) = \frac{1}{2} ||w||^2$，我们有

$$  p(w, b) \geq p(w, b) - \sum_i a_i ((w^T x_i +b) y_i - 1) = g(a)$$

所以$g(a)$总是$p(w, b)$的下界，这就是**Weak Duality**。同时$g(a)$还一定Concave的，这个性质甚至在原目标函数和不等式约束条件非Convex的时候都成立。而我们的问题是Convex的，这（通常）都保证了**Strong Duality**，也就是说在feasible区域中，{% mathjax %}max_{a} g(a) = min_{w, b} p(w, b){% endmathjax %}。之所以说是“通常”是因为还要满足一个Slater Condition，一个非常容易满足的条件。

总而言之，我们得到了一个与Primal问题解相同**对偶问题**

{% mathjax %}
\begin{aligned}
d^* =  max_{a\geq 0} g(a) = max_{a \geq 0} min_{w, b} L(w,b,a) = max_{a \geq 0} min_{w, b} \frac{1}{2} ||w||^2 - \sum_i a_i ((w^T x_i +b) y_i - 1)
\end{aligned}
{% endmathjax %}

在给定$a$的情况下，我们可以解决问题$min_{w, b} \frac{1}{2} ||w||^2 - \sum_i a_i ((w^T x_i +b) y_i - 1)$：

{% mathjax %}
\begin{aligned}
&dL/dw = w - \sum_i a_i y_i x_i = 0\\
&\rightarrow w = \sum_i a_i y_i x_i
\end{aligned}
{% endmathjax %}

{% mathjax %}
\begin{aligned}
&dL/db = - \sum_i a_i y_i = 0\\
&\rightarrow \sum_i a_i y_i = 0
\end{aligned}
{% endmathjax %}

带入对偶优化问题中可以得到 

$$d^* =  max_{a \geq 0}  \frac{1}{2} (\sum_i a_i y_i x_i)^T (\sum_i a_i y_i x_i) - \sum_i  a_i ( y_i \sum_j a_j y_j x_j^T x_i - 1) - b \sum_i a_i y_i$$

$$d^* =  max_{a \geq 0}  \frac{1}{2} \sum_i \sum_j a_i  a_j y_i y_j x_i^T x_j - \sum_i \sum_j a_i a_j y_i y_j x_j^T x_i - \sum_i  a_i$$

最终，我们得到原问题（Primal）对偶优化问题（Dual）

$$d^* =  min_{a \geq 0}  \frac{1}{2} \sum_i \sum_j a_i  a_j y_i y_j x_i^T x_j  + \sum_i  a_i$$

$$s.t. \sum_i a_i y_i = 0$$

### The Intercept
然而，给新数据点分类的过程中{% mathjax %}\hat{y} = w^{*T}x + b^*{% endmathjax %}还是需要求出{% mathjax %}b^*{% endmathjax %}的。其求{% mathjax %}b^*{% endmathjax %}的过程非常直接，因为，假设$x_{sv}$是Support Vector，那么就有


{% mathjax %}
\begin{aligned}
y_{sv}(w^{*T}x_{sv} + b^*) = 1
\end{aligned}
{% endmathjax %}

于是我们有

{% mathjax %}
\begin{aligned}
b^* = y_i - w^{*T}x_{sv} = y_{sv} - \sum_j a_jy_jx_jx_{sv}, s.t. a_i > 0
\end{aligned}
{% endmathjax %}

Support Vector的判定标准是$a_{sv} > 0$，会在后面提到。

[MIT Open Course](https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec12.pdf)和[Andrew Ng的notes](https://www.cs.cmu.edu/~ggordon/SVMs/new-svms-and-kernels.pdf)分别给出了如下两种形式，但是其思路都是一样的。
1. MIT Open Course: {% mathjax %}b^* = 1 - min_{i:y_i=1} w^{*T}x_i{% endmathjax %}
2. Andrew Ng: {% mathjax %}b^* = -(min_{i:y_i=1} w^{*T}x_i + max_{i:y_i=-1} w^{*T}x_i)/2{% endmathjax %}

### Kernel Trick

这时我们不再直接求解feature-wise weights $w$，而是关心sample weights $a$了。不过此时我们还没有解决任何问题。之所以转成对偶问题，一个好处是我们可以用Kernel来避免$x \rightarrow \phi(x)$升维带来的性能问题。那么Kernel是什么又是怎么做到避免这个问题的呢？下面让我们来整理一下我们训练和预测时所要做的所有工作：

- 训练：训练过程中我们要解决如下优化问题：
$$d^* =  min_{a \geq 0}  \frac{1}{2} \sum_i \sum_j a_i  a_j y_i y_j x_i^T x_j  + \sum_i  a_i$$

$$s.t. \sum_i a_i y_i = 0$$
 
 - 预测：对于新数据点$x$
 
 {% mathjax %}
 \begin{aligned}
 \hat{y} = w^T x + b = \sum_i a_i y_i x_i^T x + y_{sv} - \sum_j a_jy_jx_jx_{sv}
 \end{aligned}
 {% endmathjax %}

如果我们做$x \rightarrow \phi(x)$，则有
{% mathjax %}
\begin{aligned}
& d^* =  min_{a \geq 0}  \frac{1}{2} \sum_i \sum_j a_i  a_j y_i y_j \phi(x_i)^T \phi(x_j)  + \sum_i  a_i\\
& s.t. \sum_i a_i y_i = 0\\
& \hat{y} = w^T x + b = \sum_i a_i y_i \phi(x_i)^T \phi(x) + y_{sv} - \sum_j a_jy_j\phi(x_j)\phi(x_{sv})
\end{aligned}
{% endmathjax %}

很容易看出，$x$或者说$\phi(x)$永远都不单独出现，而是出现在和另一个$\phi(x')$的内积运算中，如果我们能避免做这个升维操作而直接用$x$和$x'$计算出这个内积$\phi(x)\phi(x')$的结果，那我们就解决了升维度带来的效率问题，这个能避免升维就能用$x$和$x'$能计算出升维后内积的函数就是Kernel Function $k(x, x') \Longleftrightarrow  \phi(x) \phi(x')$。通过核函数，我们可以将训练和预测问题进一步转化为

$$d^* =  min_{a \geq 0}  \frac{1}{2} \sum_i \sum_j a_i  a_j y_i y_j k(x_i, x_j)  + \sum_i  a_i$$

$$s.t. \sum_i a_i y_i = 0$$

{% mathjax %}
\begin{aligned}
\hat{y} = w^T x + b = \sum_i a_i y_i k(x_i, x) + y_{sv} - \sum_j a_jy_j k(x_j, x_{sv})
\end{aligned}
{% endmathjax %}

几种常见的核函数有：
1. 多项式核：$k(x, x') = (x^T x' + 1)^d$
2. RBF核：$k(x, x') = exp(-\frac{||x - x'||^2}{2 \sigma^2})$
3. 线性核：$k(x, x') = x^T x'$

可以看出几种核函数都避免了升维的过程，其内积操作都是在原维度空间中完成的。举个例子来说$k((x_1, x_2), (x'_1, x'_2)) = ((x_1, x_2)^T (x'_1, x'_2) + 1)^2$是与$(x_1, x_2) \rightarrow (x_1, x_2, x^2_1, x^2_2, x_1 x_2)$基本等价的，然而用核方法就避免了这个升维度的过程。

不过如果任意给定一个$\phi(x)$让我们去找其核函数，能不能做到呢？答案是非常困难。所以其实常用的核函数一般就那么几个。

细心的你可能突然发现，慢着这几个核函数看起来似乎很像两个向量的距离/不相似度函数啊。实际上也是这样，而且当我们再回头看一下预测函数$\hat{y} = w^T x + b = \sum_i a_i y_i k(x_i, x) + b$就不难理解了，对于新数据点$x$我们就是在用训练数据中支持向量们（$a_i \neq 0$）的labels做weighted majority votes来决定，其权重为$a_i k(x_i, x)$。当然，这里我们不能随便拿个距离函数就当做核函数来用，因为其会改变优化问题自身的性质。


### 用拉格朗日函数的形式表达Primal问题

实际上原优化问题也可以表示为拉格朗日函数的形式，其形式为

{% mathjax %}
\begin{aligned}
p^* = min_{w, b} max_{a\geq0} L(w, b, a) = min_{w,b} max_{a\geq0} \frac{1}{2} ||w||^2 - \sum_i a_i(y_i(w^Tx_i + b) - 1)
\end{aligned}
{% endmathjax %}

这是因为：
(1) 对于不满足约束条件的$w, b$，即{% mathjax %}\exists x_i, w^T x_i + b < 0{% endmathjax %}，则$a_i$越大{% mathjax %}-a_i(y_i(w^Tx_i + b) - 1){% endmathjax %}越大，最终{% mathjax %}max_{a\geq0} L(w, b, a) = +\infty{% endmathjax %}，所以{% mathjax %}min_{w, b} max_{a\geq0} L(w, b, a){% endmathjax %}显然不会挑选不满足约束条件的$w, b$
(2) 对于满足约束条件的$w,b$，即{% mathjax %}\forall x_i, w^T x_i + b \geq 0{% endmathjax %}，则{% mathjax %}max_{a\geq0} \frac{1}{2} ||w||^2 - \sum_i a_i(y_i(w^Tx_i + b) - 1){% endmathjax %}中显然第二项为0（又称为Complementary slackness，是KKT条件的一部分，在此不做细说），所以{% mathjax %}max_{a\geq0} L(w, b, a) = \frac{1}{2}||w||^2{% endmathjax %}，即优化问题又变回了{% mathjax %}min_{w, b} \frac{1}{2} ||w||^2{% endmathjax %}

所以上面这个优化问题就是和原优化问题等价的。指的注意的是，它和Dual问题的区别仅仅是min-max的顺序。从这个角度我们也可以通俗的解释为什么{% mathjax %}d^*{% endmathjax %}是{% mathjax %}p^*{% endmathjax %}的下界。{% mathjax %}p^*{% endmathjax %}是从一堆大个儿里挑最小的，而$d^*$是从一堆个小的里面挑个最大的。

另外一个有趣的事实是，(2)中阐述过，对于符合约束条件的$w,b$，$\sum_i a_i(y_i(w^Tx_i + b) - 1) = 0, a_i 
\geq 0$，这也就意味着当$y_i(w^Tx_i + b) - 1 > 0$时，其sample weight $a_i$必须是0。也就是说非支持向量的数据点在预测时完全被抛弃了。这给了我们一个判定支持向量的方法，在上文中计算截距$b$的时候用到了。

### Soft Margin SVM的Dual问题
Soft Margin SVM的对偶问题推导和Hard Margin SVM大同小异，得到的结果也非常相似

$$d^* =  min_{C \geq a \geq 0}  \frac{1}{2} \sum_i \sum_j a_i  a_j y_i y_j x_i^T x_j  + \sum_i  a_i$$

$$s.t. \sum_i a_i y_i = 0$$

仅仅是多了条件$a \leq C$，出人意料的简单。预测过程中，截距$b$的计算过程基本一致，此时因为有松弛变量的存在，只有满足$C > a > 0$的点才是落在Margin上没有松弛的点。


## 参考资料
[1] http://cs229.stanford.edu/notes/cs229-notes3.pdf 
[2] https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec12.pdf
[3] http://blog.pluskid.org/?page_id=683
