# StyleGAN - Generate Fashion Images

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![TensorFlow 1.10](https://img.shields.io/badge/tensorflow-1.10-green.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

## 0. Verbose

**默认读者已知晓内容：README-raw.md**


>
1. TensorFlow搭建动态网络图非常蛋疼；如果习惯了图流的方式便会觉得非常巧妙；代码主体思路：先搭建起受变量控制的模型，其尺寸和深度递归下依赖控制变量，然后在训练中改变控制变量以改变图。图结构在`training.networks_stylegan.G_style`就描述清楚，通过`lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)`控制了其执行逻辑。
2. 训练耗时非常长；单卡512分辨率要3周时间。
3. 效果很好，窃以为PG训练过程为迄今解决模式崩塌问题最好方式（2019-05）。



## 1. Results

![](md-images/Xnip2019-05-22_18-55-29.png)

## 2. G / D Networks

```
Constructing networks...

G                             Params    OutputShape         WeightShape     
---                           ---       ---                 ---             
latents_in                    -         (?, 512)            -               
labels_in                     -         (?, 0)              -               
lod                           -         ()                  -               
dlatent_avg                   -         (512,)              -               
G_mapping/latents_in          -         (?, 512)            -               
G_mapping/labels_in           -         (?, 0)              -               
G_mapping/PixelNorm           -         (?, 512)            -               
G_mapping/Dense0              262656    (?, 512)            (512, 512)      
G_mapping/Dense1              262656    (?, 512)            (512, 512)      
G_mapping/Dense2              262656    (?, 512)            (512, 512)      
G_mapping/Dense3              262656    (?, 512)            (512, 512)      
G_mapping/Dense4              262656    (?, 512)            (512, 512)      
G_mapping/Dense5              262656    (?, 512)            (512, 512)      
G_mapping/Dense6              262656    (?, 512)            (512, 512)      
G_mapping/Dense7              262656    (?, 512)            (512, 512)      
G_mapping/Broadcast           -         (?, 16, 512)        -               
G_mapping/dlatents_out        -         (?, 16, 512)        -               
Truncation                    -         (?, 16, 512)        -               
G_synthesis/dlatents_in       -         (?, 16, 512)        -               
G_synthesis/4x4/Const         534528    (?, 512, 4, 4)      (512,)          
G_synthesis/4x4/Conv          2885632   (?, 512, 4, 4)      (3, 3, 512, 512)
G_synthesis/ToRGB_lod7        1539      (?, 3, 4, 4)        (1, 1, 512, 3)  
G_synthesis/8x8/Conv0_up      2885632   (?, 512, 8, 8)      (3, 3, 512, 512)
G_synthesis/8x8/Conv1         2885632   (?, 512, 8, 8)      (3, 3, 512, 512)
G_synthesis/ToRGB_lod6        1539      (?, 3, 8, 8)        (1, 1, 512, 3)  
G_synthesis/Upscale2D         -         (?, 3, 8, 8)        -               
G_synthesis/Grow_lod6         -         (?, 3, 8, 8)        -               
G_synthesis/16x16/Conv0_up    2885632   (?, 512, 16, 16)    (3, 3, 512, 512)
G_synthesis/16x16/Conv1       2885632   (?, 512, 16, 16)    (3, 3, 512, 512)
G_synthesis/ToRGB_lod5        1539      (?, 3, 16, 16)      (1, 1, 512, 3)  
G_synthesis/Upscale2D_1       -         (?, 3, 16, 16)      -               
G_synthesis/Grow_lod5         -         (?, 3, 16, 16)      -               
G_synthesis/32x32/Conv0_up    2885632   (?, 512, 32, 32)    (3, 3, 512, 512)
G_synthesis/32x32/Conv1       2885632   (?, 512, 32, 32)    (3, 3, 512, 512)
G_synthesis/ToRGB_lod4        1539      (?, 3, 32, 32)      (1, 1, 512, 3)  
G_synthesis/Upscale2D_2       -         (?, 3, 32, 32)      -               
G_synthesis/Grow_lod4         -         (?, 3, 32, 32)      -               
G_synthesis/64x64/Conv0_up    1442816   (?, 256, 64, 64)    (3, 3, 512, 256)
G_synthesis/64x64/Conv1       852992    (?, 256, 64, 64)    (3, 3, 256, 256)
G_synthesis/ToRGB_lod3        771       (?, 3, 64, 64)      (1, 1, 256, 3)  
G_synthesis/Upscale2D_3       -         (?, 3, 64, 64)      -               
G_synthesis/Grow_lod3         -         (?, 3, 64, 64)      -               
G_synthesis/128x128/Conv0_up  426496    (?, 128, 128, 128)  (3, 3, 256, 128)
G_synthesis/128x128/Conv1     279040    (?, 128, 128, 128)  (3, 3, 128, 128)
G_synthesis/ToRGB_lod2        387       (?, 3, 128, 128)    (1, 1, 128, 3)  
G_synthesis/Upscale2D_4       -         (?, 3, 128, 128)    -               
G_synthesis/Grow_lod2         -         (?, 3, 128, 128)    -               
G_synthesis/256x256/Conv0_up  139520    (?, 64, 256, 256)   (3, 3, 128, 64) 
G_synthesis/256x256/Conv1     102656    (?, 64, 256, 256)   (3, 3, 64, 64)  
G_synthesis/ToRGB_lod1        195       (?, 3, 256, 256)    (1, 1, 64, 3)   
G_synthesis/Upscale2D_5       -         (?, 3, 256, 256)    -               
G_synthesis/Grow_lod1         -         (?, 3, 256, 256)    -               
G_synthesis/512x512/Conv0_up  51328     (?, 32, 512, 512)   (3, 3, 64, 32)  
G_synthesis/512x512/Conv1     42112     (?, 32, 512, 512)   (3, 3, 32, 32)  
G_synthesis/ToRGB_lod0        99        (?, 3, 512, 512)    (1, 1, 32, 3)   
G_synthesis/Upscale2D_6       -         (?, 3, 512, 512)    -               
G_synthesis/Grow_lod0         -         (?, 3, 512, 512)    -               
G_synthesis/images_out        -         (?, 3, 512, 512)    -               
G_synthesis/lod               -         ()                  -               
G_synthesis/noise0            -         (1, 1, 4, 4)        -               
G_synthesis/noise1            -         (1, 1, 4, 4)        -               
G_synthesis/noise2            -         (1, 1, 8, 8)        -               
G_synthesis/noise3            -         (1, 1, 8, 8)        -               
G_synthesis/noise4            -         (1, 1, 16, 16)      -               
G_synthesis/noise5            -         (1, 1, 16, 16)      -               
G_synthesis/noise6            -         (1, 1, 32, 32)      -               
G_synthesis/noise7            -         (1, 1, 32, 32)      -               
G_synthesis/noise8            -         (1, 1, 64, 64)      -               
G_synthesis/noise9            -         (1, 1, 64, 64)      -               
G_synthesis/noise10           -         (1, 1, 128, 128)    -               
G_synthesis/noise11           -         (1, 1, 128, 128)    -               
G_synthesis/noise12           -         (1, 1, 256, 256)    -               
G_synthesis/noise13           -         (1, 1, 256, 256)    -               
G_synthesis/noise14           -         (1, 1, 512, 512)    -               
G_synthesis/noise15           -         (1, 1, 512, 512)    -               
images_out                    -         (?, 3, 512, 512)    -               
---                           ---       ---                 ---             
Total                         26179768                                      
```


```
D                    Params    OutputShape         WeightShape     
---                  ---       ---                 ---             
images_in            -         (?, 3, 512, 512)    -               
labels_in            -         (?, 0)              -               
lod                  -         ()                  -               
FromRGB_lod0         128       (?, 32, 512, 512)   (1, 1, 3, 32)   
512x512/Conv0        9248      (?, 32, 512, 512)   (3, 3, 32, 32)  
512x512/Conv1_down   18496     (?, 64, 256, 256)   (3, 3, 32, 64)  
Downscale2D          -         (?, 3, 256, 256)    -               
FromRGB_lod1         256       (?, 64, 256, 256)   (1, 1, 3, 64)   
Grow_lod0            -         (?, 64, 256, 256)   -               
256x256/Conv0        36928     (?, 64, 256, 256)   (3, 3, 64, 64)  
256x256/Conv1_down   73856     (?, 128, 128, 128)  (3, 3, 64, 128) 
Downscale2D_1        -         (?, 3, 128, 128)    -               
FromRGB_lod2         512       (?, 128, 128, 128)  (1, 1, 3, 128)  
Grow_lod1            -         (?, 128, 128, 128)  -               
128x128/Conv0        147584    (?, 128, 128, 128)  (3, 3, 128, 128)
128x128/Conv1_down   295168    (?, 256, 64, 64)    (3, 3, 128, 256)
Downscale2D_2        -         (?, 3, 64, 64)      -               
FromRGB_lod3         1024      (?, 256, 64, 64)    (1, 1, 3, 256)  
Grow_lod2            -         (?, 256, 64, 64)    -               
64x64/Conv0          590080    (?, 256, 64, 64)    (3, 3, 256, 256)
64x64/Conv1_down     1180160   (?, 512, 32, 32)    (3, 3, 256, 512)
Downscale2D_3        -         (?, 3, 32, 32)      -               
FromRGB_lod4         2048      (?, 512, 32, 32)    (1, 1, 3, 512)  
Grow_lod3            -         (?, 512, 32, 32)    -               
32x32/Conv0          2359808   (?, 512, 32, 32)    (3, 3, 512, 512)
32x32/Conv1_down     2359808   (?, 512, 16, 16)    (3, 3, 512, 512)
Downscale2D_4        -         (?, 3, 16, 16)      -               
FromRGB_lod5         2048      (?, 512, 16, 16)    (1, 1, 3, 512)  
Grow_lod4            -         (?, 512, 16, 16)    -               
16x16/Conv0          2359808   (?, 512, 16, 16)    (3, 3, 512, 512)
16x16/Conv1_down     2359808   (?, 512, 8, 8)      (3, 3, 512, 512)
Downscale2D_5        -         (?, 3, 8, 8)        -               
FromRGB_lod6         2048      (?, 512, 8, 8)      (1, 1, 3, 512)  
Grow_lod5            -         (?, 512, 8, 8)      -               
8x8/Conv0            2359808   (?, 512, 8, 8)      (3, 3, 512, 512)
8x8/Conv1_down       2359808   (?, 512, 4, 4)      (3, 3, 512, 512)
Downscale2D_6        -         (?, 3, 4, 4)        -               
FromRGB_lod7         2048      (?, 512, 4, 4)      (1, 1, 3, 512)  
Grow_lod6            -         (?, 512, 4, 4)      -               
4x4/MinibatchStddev  -         (?, 513, 4, 4)      -               
4x4/Conv             2364416   (?, 512, 4, 4)      (3, 3, 513, 512)
4x4/Dense0           4194816   (?, 512)            (8192, 512)     
4x4/Dense1           513       (?, 1)              (512, 1)        
scores_out           -         (?, 1)              -               
---                  ---       ---                 ---             
Total                23080225                                      

Building TensorFlow graph...
```