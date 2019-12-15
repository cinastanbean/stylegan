# StyleGAN - Generate Fashion Images 

目的：服饰生成、模特生成。

## 0. Verbose

**默认读者已知晓内容：README-raw.md**

# 1. 实验

# 1.1 模特生成

示例图片：

![](md-images/v5-zalando-skirt-hq-b-fakes007740.png)

![](md-images/Xnip2019-05-22_18-55-29.png)


示例视频：

[![Watch the video](md-images/v5-zalando-skirt-hq-b-fakes007740-video-snapshot.png)](https://v.youku.com/v_show/id_XNDIzNzI4MTgxMg==.html?spm=a2h3j.8428770.3416059.1)




[![Watch the video](https://img.youtube.com/vi/T-D1KVIuvjA/maxresdefault.jpg)](https://youtu.be/T-D1KVIuvjA)

#  1.2 服饰生成

示例图片：

![](md-images/v5-zalando-skirt-hq-a-fakes007710.png)


# 3. Results实验分析


关于算法实现：

1. TensorFlow搭建动态网络图非常蛋疼；如果习惯了图流的方式便会觉得非常巧妙；代码主体思路：先搭建起受变量控制的模型，其尺寸和深度递归下依赖控制变量，然后在训练中改变控制变量以改变图。图结构在`training.networks_stylegan.G_style`就描述清楚，通过`lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)`控制了其执行逻辑。
2. 训练耗时非常长；单卡512分辨率要3周时间。
3. 效果很好，窃以为PG训练过程为迄今解决模式崩塌问题最好方式（2019-05）。



