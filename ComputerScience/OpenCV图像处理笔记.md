#   OpenCV图像处理笔记

### 1. 图像读入、显示与保存

* ##### 1.1 读入图像

​      **retval = cv2.imread(文件名，[,显示控制参数])**

​	文件名： 完整文件名

​	参数： cv.IMREAD_UNCHANGED

​	            cv.IMREAD_GRAYSCALE

​	            cv.IMREAD_COLOR

​	例：img = cv2.imread("d:\\\\image.jpg")

* ##### 1.2 显示图像

  **None = cv2.imshow(窗口名，图像名)**

​	例：cv2.imshow("demo",imgage)

​	**retval = cv2.waitKey(   [,delay])**

​	delay:   

* delay > 0   等待delay毫秒

* delay < 0   等待键盘单击

* delay = 0   无限等待

  例：cv2.waitKey(0)

​	**cv2.destroyAllWindows()**

​	功能： 删除所有窗口

* ##### 1.3 保存图像

  ##### **retval = cv2.imwrite(文件地址，文件名)**

  例： cv2.imwrite('D:\\\\test.jpg',img)

  



### 2. 图像处理入门基础

* **图像是由像素构成的**
* **图像的分类**

​          二值图像 ：值为 0或者1

​          灰度图像  ：值为0~255 

​                              0：白色

​                           255：黑色

​                0~255之间：灰色

​         彩色图像 - RGB图像 ： （205,89,69）

​                       BGR图像 ： opencv中通常是BGR图像，所以通常需要处理成RGB图像

图像处理通常用的操作：  RGB转灰度

​                                       灰度转二值

### 3. 像素处理

* ##### 3.1 读取像素

​        **返回值 = 图像（位置参数）**

​           灰度图像，返回灰度值

​              例：p = img[88,142]

​                     print(p)

​           BGR图像，返回值为B, G, R的值

​               例：

​                打印单个通道的值

​                      blue = img[78,125,0]

​                      print(blue)

​                打印三个通道的值

​					  p = img[78,125]

​                      print(p)

* ##### 3.2 **修改像素值**

     灰度图像

  例：img[88,99] = 255

     BGR图像

  例：img[88,99,0] = 255

  ​	   img[88,99,1] = 255

  ​	   img[88,99,2] = 255

  或：img[88,99] = [255,255,255]

### 4. 使用numpy访问像素

* ##### 4.1 读取像素

  返回值 = 图像.item(位置参数) 

  例：Blue = img.item(78,125,0)

  ​       Green = img.item(78,125,1)

  ​       Red = img.item(78,125,2)

* ##### 4.2 修改像素

  图像名.itemset(位置,新值)

  Img.itemset((88,99),255)

### 5. 获取图像属性

* **Shape**: 可以获取图像的形状，返回包含行数、列数、通道数的元组
  * 灰度 返回行数，列数
  * 彩色 返回行数，列数，通道数
* **Size**: 可以获取图像的像素数目
  * 灰度 返回:行数*列数*
  * *彩色 返回：行数*列数*通道数

* **dtype**: 返回的是图像的数据类型

### 6. 感兴趣区域ROI

* 从被处理的图像以方框、圆、椭圆、不规则多边形等方式勾勒出需要处理的区域

  可以通过各种算法和函数来求得感兴趣区域ROI，并进行图像的下一步处理。

### 7. 通道的拆分与合并

* **拆分通道** 

  例：

  ```python
  Import v2
  
  Import numpy as np
  
  a = cv.imread(“image\lenacolor.png”)
  
  b,g,r = cv2.split(a)
  
  cv2.imshow(“B”,b)
  
  cv2.imshow(“G”g)
  
  cv2.imshow(“R”,r)
  
  cv2.waitKey(0)
  
  cv2.destroyAllWindows()
  ```

  * **合并通道**

  ​       bgr = cv2.merge([b,g,r])

  ​       rgb = cv2.merge([r,g,b])

  使用不同的顺序合并得到的结果不一样

### 8. 图像加法

* **numpy加法**

  * 取模加法 ：结果 = 图像1 + 图像2

    

  * 结果 = 图像1 + 图像2  = $\begin{cases} 像素值 <= 255 ,  图像1 + 图像2\\ 像素值 > 255,  结果对255取模\end{cases}$

* **OpenCV加法**

  * 饱和运算：结果 = cv2.add(图像1，图像2)

    

  * 结果 = cv2.add(图像1，图像2) $ = \begin{cases}像素值 <= 255,  图像1 + 图像2\\像素值 > 255,  取值255\end{cases}$

  

### 9. 图像融合

* 将两张或两张以上的图像信息融合到一张图像上。

* 融合的图像含有更多的信息、能够更方便人来观察或者计算机处理。

* 图像融合：结果图像 = 图像1*洗漱1 + 图像2 * 系数2 + 亮度调解量

  例:  img = img1 * 0.3 + img2 * 0.7 + 18

  `函数addweighted`

  Dst = cv.addweighted(src1,alpha,src2,alpha,gamma) 参数gamma不能省略

### 10. 类型转换

* 彩色转灰度

* BGR转RGB

* 灰度转BGR

  Opencv提供200多种不同类型之间的转换

  Cv2.COLOR_VGR2GRAY

  Cv2.COLOR_BGR2RGB

  Cv2.COLOR_GRAY2BGR

### 11. 图像缩放

* Dst = cv2.resize(src,dsize[,dst[,fx[,fy[,interpolation]]]])

  Dst = cv2.resize(src,dsize)

  Desize（列，行）

  例：

  B = cv2.resize(a,(round(cols * 0.5),round(rows * 1.2))) 

  *列变为原来的0.5倍，行变为原来的1.2倍*

  B = cv2.resize(a,none,fx = 1.2,fy = 0.5)

### 12. 图像反转

* Dst = cv2.flip(src,flipCode)
  * flipCode = 0 以x轴为对称轴上下翻转
  * flipCode > 0 以y轴为对称轴左右翻转
  * flipcode < 0 以x轴、y轴上下左右同时翻转

### 13. 基础理论

* 1. 二进制阈值化

  函数表示： Dst（x,y） = {maxVal  if src(x,y) >thresh  ; 0  otherwise

  说明：如果像素值比阈值大那么取最大值maxval，否则取最小值0

   

  2. 反二进制阈值化

  （则选定一个特定的灰度值作为阈值）

  函数表示：Dst（x，y） = {0 if src(x, y) > thresh ; maxVal otherwise

  说明： 如果像素值比阈值大那么取最小值0，否则取最大值maxval，与二进制阈值化的结果正好相反

   

  3. 截断阈值化

  函数表示：Dst (x, y) = {threshold if src(x,y) >thresh ; src (x,y) otherwise

  说明： 如果像素值比阈值大那么变成阈值，否则不变

   

  4. 反阈值化为0

  函数表示： Dst(x,y) = { 0 if src(x,y) > thresh ; src(x,y) otherwise 

  说明：大于阈值的像素处理为0，否则保持不变

   

  5. 阈值化为0

  函数表示： Dst（x，y）={if src（x，y）> thresh ; 0 otherwise

  说明：大于阈值不变，小于阈值 的处理为0

### 14. threshold函数

* **函数threshold**

  retval, dst = cv2.threshold( src, thresh, maxval, type)

  * retval,  阈值
  * dst ,  处理结果
  * src, 源图像
  * threshold , 阈值
  * maxval, 最大值
  * type, 类型

* 二进制阈值化(也可以是二值阈值化)

  * 关键字： **cv2.THRESH_BINARY**

    ```python
    import cv2
    a = cv2.imread("image\\lena512.bmp", cv2.IMREAD_UNCHANGED)
    r,b = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
    ```

    > 大于阈值处理为255     （亮的处理为白色）
    >
    > 小于或等于阈值处理为0  （暗的处理为黑色）

  ​        

* 反二进制阈值化

  * 关键字： **cv2.THRESH_BINARY_INV**

    ```python
    import cv2
    a = cv2.imread("image\\lena512.bmp", cv2.IMREAD_UNCHANGED)
    r,b = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY_INV) 
    ```

    > 大于阈值的处理为0       （亮的处理为黑色）
    >
    > 小于或等于阈值的处理为255（暗的处理为白色）

* 截断阈值化

  * 关键字： **cv2.THRESH_TRUNC**

    ```python
    import cv2
    a = cv2.imread("image\\lena512.bmp", cv2.IMREAD_UNCHANGED)
    r,b = cv2.threshold(a, 127, 255, cv2.THRESH_TRUNC) 
    ```

    > 把大于阈值的都处理为阈值
    >
    > 即：把比较亮的像素值都处理为了阈值

* 反阈值化为0

  * 关键字： **cv2.THRESH_TOZERO_INV**

    ```python
    import cv2
    a = cv2.imread("image\\lena512.bmp", cv2.IMREAD_UNCHANGED)
    r,b = cv2.threshold(a, 127, 255, cv2.THRESH_TOZERO_INV) 
    ```

    > 把比较亮的部分处理为0， 小于等于阈值的不变

* 阈值化为0

  * 关键字： **cv2.THRESH_TOZERO**

    ```python
    import cv2
    a = cv2.imread("image\\lena512.bmp", cv2.IMREAD_UNCHANGED)
    r,b = cv2.threshold(a, 127, 255, cv2.THRESH_TOZERO) 
    ```

    > 把比较亮的部分不变，比较暗的部分处理为0



### 15. 均值滤波

* 任意一点的像素值，都是周围N*N像素值的均值
* 函数blur 处理结果 = cv2.blur（原始图像，核大小）
  * 核大小：以（高度、宽度）的形式表示的元组

### 16. 方框滤波

* boxFilter函数

  处理结果 = cv2.boxFilter(原始图像，目标图像深度，核大小，归一化属性（normalize）)

  normalize属性 k=1/α[……]

  $α=\begin{cases} \frac{1}{width  *  neight},     normalize =true \\ 1,            normalize =false\end{cases}$  

  即当normalize =0时，不进行归一化处理。

  当normalize=1时，归一化处理，与均值滤波相同。

### 17. 高斯滤波

* 让临近的像素具有更高的重要度，对周围像素计算加权平均值，较近的像素具有较大的权重值。

  `GaussianBlur函数`

  Dst= cv2.GaussianBlur（src， ksize， sigmax）

  * Sigmax ：x方向差，控制权重  y方向的方差与x保持一致
  *  Sigmax =0.3 \*(( ksize-1) * 0.5)-1)+ 0.8

### 18. 中值滤波

* 让临近的像素按照大小排列，取排列像素几种位于中间位置的值作为中值滤波后的像素值。

  `MedianBlur函数`

  Dst =cv2.medianBlur(src, ksize)

  * Ksize: 核大小，必须是比1大的奇数，如3，5，7等。

### 19. 图像腐蚀

* #### 基础

  **1. 形态学转换主要针对的是二值图像**

  **2. 两个输入对象。**

  ​    对象1： 二值图像

  ​    对象2： 卷积核

* *卷积核的中心逐个像素扫描原始图像*

* *被扫描到的元素图像中的像素点，只有当卷积核对应的元素值均为1时，其值才为1，否则值为0*

  #### 函数erode

  dst = cv2.erode( src, kernel, iterations )

  dst , 处理结果     src， 源图像

  ​                          kernel, 卷积核

  ​                          iterations, 迭代次数

  

### 20. 图像膨胀

* 膨胀是腐蚀操作的逆操作
* *图像被腐蚀后，去除了噪声，但是会压缩图像*
* *对腐蚀过的图像，进行膨胀处理，可以去除噪声，并保持原有形状*
* *卷积核的中心逐个像素扫描原始图像*
* *被扫描到的元素图像中的像素点，只有当卷积核对应的元素值均为1时，其值才为1，否则值为0*

       ####        函数dilate

​         dst = cv2.dilate( src, kernel, iterations )

​         dst , 处理结果     src， 源图像

​                                   kernel, 卷积核

​                                   iterations, 迭代次数

### 21. 开运算

​	**开运算（image） = 膨胀（腐蚀（image））**

* *图像被腐蚀后，去除了噪声，但是会压缩图像*

* *对腐蚀过的图像，进行膨胀处理，可以去除噪声，并保持原有的形状*

​	**函数morphologyEX**

​		opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

​		*opening* ,开运算结果                     	| *img*,源图像

​                                         				        | cv2.MORPH_OPEN,开运算

​									    			   		| kernel,卷积核        例：kernel = np.ones((5,5),np.unit8)



### 22. 闭运算

​	**闭运算（image） = 腐蚀（膨胀（image））**

* *先膨胀，后腐蚀*

* *它有助于关闭前景物体内部的小孔，或物体上的小黑点*

**函数morphologyEX**

closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

*closing* ,闭运算结果                   | *img*,源图像

​                                                 | cv2.MORPH_CLOSE,闭运算

​									     		| kernel,卷积核        例：kernel = np.ones((5,5),np.unit8)



### 23. 梯度运算

​	 **梯度（image) = 膨胀（image) - 腐蚀（image)**

- *得到轮廓图像*

  **函数morphologyEX**

  result = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

  *result* ,梯度结果                     	| *img*,源图像

  ​                                                 | cv2.MORPH_GRADIENT,梯度

  ​									     		| kernel,卷积核        例：kernel = np.ones((5,5),np.unit8)



### 24. 礼帽操作

​	**礼帽（image) = 原始图像（image) - 开运算（image)**

- *得到噪声图像*

  **函数morphologyEX**

​	result = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)

* result* ,礼帽结果                     	| *img*,源图像

  ​                                                   | cv2.MORPH_TOPHAT,礼帽

  ​									      	 	| kernel,卷积核        例：kernel = np.ones((5,5),np.unit8)



### 25. 黑帽图像处理

​	黑帽（image) = 闭运算（image) - 原始（image)**

- *得到图像内部的小孔，或前景色中的小黑点*

  **函数morphologyEX**

  result = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)

  *result* ,黑帽结果                     	| *img*,源图像

  ​                                                 | cv2.MORPH_BLACKHAT,黑帽

  ​									     		| kernel,卷积核        例：kernel = np.ones((5,5),np.unit8)

### 26. sobel理论基础

### 27. sobel算计及其函数使用

​    **dst = cv2.Sobel( src, ddepth, dx, dy, [ ksize ])**

​    dst, 计算结果        src, 原始图像

​								ddepth, 处理结果图像深度   

​								dx, x轴方向

​								dy, y轴方向

​								ksize, 核大小

* ddepth, 处理结果图像深度   

​			*通常情况下，可以将该参数的值设置为**-1**，让处理结果与原始图像保持一致*

​			*实际操作中，计算梯度值可能会出现负数。通常处理的图像是np.unit8类型，如果结果也是该类型，所有负数会自动截断为0，发生信息丢失。所以通常计算时，使用更高的数据类型，cv.CV_64F, 取绝对值后，再转换为np.unit8（cv2.CV_8U)类型。*

​            *dst = cv2.convertScaleAbs( src [, alpha [ , beta]])*

​            *作用： 将原始图像src转换为256色位图*

​            *公式*： 目标图像 = 调整 （ 原始图像 * alpha + beta )

​            直接调整： 目标图像 = cv2.convertScaleAbs( 原始图像 )

* dx , x 轴方向

* dy , y 轴方向

  ​    计算x方向梯度 ： 【dx = 1, dy =0]

  ​    计算y方向梯度 ： 【dx = 0, dy =1]

* 计算sobel结果

   * * 方式1

       ​      dx = 1, dy = 1

       ​      dst = cv2.Sobel( src, ddepth, 1, 1 )

   * * 方式2

       ​      dx = cv2.Sobel( src, ddepth, 1, 0 )

       ​      dy = cv2.Sobel( src, ddepth, 0, 1 )

       ​      dst = dx + dy

       ​      dst = dx * 系数1 + dy * 系数2

       

     **函数**
     *dst = cv2.addWeighted(src1，alpha，src2，beta，gamma）*

     *功能： 计算两幅图像的权重和*

     ​     dst，计算结果         src1， 源图像1
     ​                                    alpha， 源图像1的系数
     ​                                    src2， 源图像2
     ​                                    beta， 源图像2的系数
     ​                                    gamma， 修正值
     关系  dst( I ) = saturate ( src1 ( I ) * alpha + src2( I ) * beta + gamma)
     例： dst = cv2.addWeighted(src1，0.5，src2，0.5，0）

### 28. scharr算子函数及其使用

​    使用3 * 3的sobel算子时，可能不太精准，scharr算子效果更好

​    **dst = Scharr ( src , ddpeth, dx, dy )**

     dst，计算结果                  src1， 原始图像
                                    ddepth， 处理结果图像深度
                                    dx， x轴方向
                                    dy， y轴方向                           
​	ddepth, 处理结果图像深度

​          dst = Scharr ( src, cv2.CV_64F, dx, dy )

​          dst = cv2.convertScaleAbs( dst )

    dx， x轴方向        dy， y轴方向 
    dst = Scharr ( src, ddpeth, dx = 1, dy = 0 )
    dst = Scharr ( src, ddpeth, dx = 0, dy = 1 )
    两个方向的梯度 = dx + dy
    scharrxy = cv2.addWeighted ( scharrx, 0.5, scharry, 0.5, 0 )
​         **满足条件： dx >= 0 && dy >= 0 && dx + dy == 1**

​         dst = Scharr ( src , ddpeth, dx, dy ) 

   等价于

​        dst = cv2.Sobel( src, ddepth, dx, dy, -1)

### 29. sobel算子和scharr算子的比较

​      不同之处：系数不同

​       相同： 使用的卷积核大小一样，运算速度一样

​        scharr更精密一点

### 30. laplacian算子及其使用

​        拉普拉斯算子类似于二阶sobel导数。实际上，在OpenCV中通过调用sobel算子来计算拉普拉斯算子。

​        一阶导数     sobel算子 = |左 - 右| + |下 - 上|     scharr算子 = |左 - 右| + |下 - 上| 

​        二阶导数     Laplacian算子 = |左 - 右| + |下 - 上| + |左 - 右| + |下 - 上| 

* **dst = cv2.Laplacian( src, ddepth )**

  * dst , 结果图像

  * src , 原始图像

  * ddepth , 图像深度

    * 通常情况下，可以将该参数的值设为**-1**，让处理结果与原始图像保持一致。

    * 实际操作中，计算梯度值可能会出现负数。
      通常处理的图像是np.uint8类型，如果结果也是该类型，所有负数会自动截断为0，发生信息丢失。
      所以，通常计算时，使用更高的数据类型**cv2.CV_64F**，取绝对值后，再转换为np.uint8(cv2℃v81-J)类型。

      出现负数时：dst = cv2.convertScaIeAbs(src）
                          将原始图像调整为256色位图。
                          示例：目标图像 = cv2.convertScaleAbs（原始图像）

### 31. canny边缘检测原理

* Canny边缘检测的一般步骤

  1. 去噪

     * 边缘检测容易受到噪声的影响。因此，在进行边缘检测前，通常需
       要先进行去噪。
     * 通常采用高斯滤波器去除噪声。
     * 让临近的像素具有更高的重要度。对周围像素计算加权平均值，较近的像素具有较大的权重值。

  2. 梯度

     * 对平滑后的图像采用sobel算子计算梯度和方向。
     * 梯度的方向一般总是与边界垂直
     * 梯度方向被归为四类：垂直， 水平，和两个对角线

  3. 非极大值抑制

     * 在获得了梯度和方向后，遍历图像，去除所有不是边界的点。
     * 实现方法：逐个遍历像素点，判断当前像素点是否是周围像素点中具有相同
       方向梯度的最大值

  4. 滞后阈值

     ​    ![image-20200331171827060](C:\Users\Ovis\AppData\Roaming\Typora\typora-user-images\image-20200331171827060.png)

### 

### 33. 理论基础

* **图像金字塔**： 同一图像的不同分辨率的子图集合

  * 向下取样：从高分辨率到低分辨率图像，缩小图像

    * 从第 i 层获取第 i + 1层， G<sub>i</sub> -> G<sub>i+1</sub>

          1.   对图像 G<sub>i</sub> 进行高斯核卷积。
  
     2.   删除所有的偶数行和列
  
    * 原始图像 M * N兮处理结果 M/2 *  N/2
      每次处理后，结果图像是原来的1/4．
      上述操作被称为：Octave
    * 重复执行该过程，构造图像金字塔  
  * *注意*：向下会丢失信息
  
* 向上取样：在每个方向上扩大为原来的2倍，新增的行和列以0填充。放大图像
  
    * 使用与“向下采用”同样的卷积核乘以4，获取“新增像素”的新值。
  * 注意：放大后的图像比原始图像要模糊。
  
  * 向上取样、向下取样不是互逆操作，经过两种操作后，无法恢复原有图像。

### 34. pyrDown函数及使用

​         向下取样 ： 将图像的尺度变小，变成原来的四分之一

* **dst  = cv2.pyrDown( src ) **
  * *dst* , 向下取样结果    
  *   *src* ,  原始图像

### 35. pyrUp函数及使用

​         向上取样 ： 将图像的尺度变大，变成原来的四倍

* **dst  = cv2.pyrUp( src ) **
  * *dst* , 向下取样结果    
  *   *src* ,  原始图像

### 36. 取样可逆性研究

​        原始图像          向下取样         dst  = cv2.pyrDown( src )   *dst* ,向下取样结果      *src*, 原始图像

​         M * N    ---> M/2 * N/2

​        原始图像          向上取样         dst  = cv2.pyrUp( src )   *dst* ,向上取样结果      *src*, 原始图像

​         M * N    ---> M*2 * N*2

​        **图像先向上取样再向下取样（或者先向下取样再向上取样）的过程都是不可逆的**

### 37. 拉普拉斯金字塔

* 向下取样 ： 图像尺度不断变小

* 向上取样 ： 图像尺度不断变大

* 拉普拉斯金字塔

  ​    **Li<sub>i</sub> = G<sub>i</sub> - PyrUp( PyrDown ( G<sub>i</sub> )) **

  ​    G<sub>i</sub>,  原始图像

  ​    L<sub>i</sub>,  拉普拉斯金字塔图像

  ![image-20200331154850995](C:\Users\Ovis\AppData\Roaming\Typora\typora-user-images\image-20200331154850995.png)

### 38. 图像轮廓

* **轮廓**

  * 边缘检测能够测出边缘，但是边缘是不连续的。

  * 将边缘连接为一个整体，构成轮廓。

    

* **注意**：

  * 对象是二值图像。所以需要预先进行阈值分割或者边缘检测处理。

  * 查找轮廓需要更改原始图像。因此，通常使用原始图像的一份拷贝操作。
  * 在OpenCV中，是从黑色背景中查找白色对象。因此，对象必须是白色的，背景必须是黑色的。

* **使用函数**

  * cv2.findContours( )

  * cv2.drawContours( )

​      *查找图像轮廓的函数是cv2.findContours( )，通过cv2.drawContours( )将查找到的轮廓绘制到图像上。*

* **cv2.findContours(）**

​            image, contours, hierarchy=cv2.findContours(image,mode,method)

* 参数
  * mage ，修改后的原始图像
  * contours , 轮廓
  * hierarchy , 图像的拓扑信息（轮廓层次）
  *    image , 原始图像                  
  * ​    mode , 轮廓检索模式   
  *    method , 轮廓的近似方法                       
                         
                         
                          

* * *mode, 轮廓检索模式*
    * cv2.RETREXTERNAL   ：  表示只检测外轮廓
    * cv2.RETRLIST             ：检测的轮廓不建立等级关系
    * cv2.RETRCCOMP       ：建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    * cv2.RETRTREE            ：建立一个等级树结构的轮廓。
      		  

* * method，轮廓的近似方法
    
    * cv.CHAINAPPROXNONE           ：到存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max ( abs（×1一×2, abs( y2 - y1 )）== 1
    * cv2.CHAIN-APPROX_SIMPLE     :  压缩水平方向，垂直方向，对角线方向的元素,只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

    * cv2℃HAINAPPROXTC89Ll          :  使用teh-ChinIchain近似算法
    * cv2℃HAINAPPROXTC89KCOS   :  使用teh-ChinIchain近似算法

* **cv2.drawContours( ）**

  ​      r=cv2.drawContours(),contours,contourldx,color[,thickness])

  * r                    :   目标图像，直接修改目标的像素点，实现绘制。

  *  o                   :   原始图像

  * contours       :   需要绘制的边缘数组。

  * contourldx    :   需要绘制的边缘索引，如果全部绘制则为一1。

  * color             :   绘制的颜色，为BGR格式的Scalar。

  * thickness      :   可选，绘制的密度，即描绘轮廓时所用的画笔粗细。


​    

### 39. 直方图

* **普通直方图**
  * 横坐标： 图像中各个像素点的灰度级（0~255，每一个就是一个灰度级）
  * 纵坐标： 具有该灰度级的像素个数

* **归一化直方图**
  * 横坐标： 图像中各个像素点的灰度级
  * 纵坐标： 出现这个灰度级的概率

​     **DIMS: 使用参数的数量**

​             dism = 1

​    **BINS: 参数子集的数目**

​          [0, 255] = [0, 15]U[16,31]U……U[240,255]

​          范围 = bin<sub>1</sub> U bin<sub>2</sub> U ... U bin~16~

​     **RANGE: 统计灰度值的范围，一般为[0,255]**

​         最小值： 0, 黑色

​         最大值： 255, 白色

### 40. 绘制直方图

* **matplotlib**
  * pyplot: 提供了类似于matlab的绘图框架
  * matplotlib.pyplot
  * import matplotlib.pyplot as plt

* **函数ravel** (把二维图像处理成一维数据)
  * hist( 数据源， 像素级)
    * 数据源：图像，必须是一维数组
    * 像素级：一般是256，指[0,255]

### 41. 使用OpenCV统计直方图

* 横坐标： [0, 255]

* 纵坐标： 各个灰度级的像素个数

  [N<sub>0</sub>, N<sub>1</sub>, N<sub>2</sub>, …… N<sub>254</sub>, N<sub>255</sub>, ]

* **函数calcHist**

  * *一般形式：* hist = cv2.calcHist(images, channels, mask, histSize, ranges, accumulate)

    * **hist**:             直方图   (返回的直方图，是一个二维数组)
    * **images**：     原始图像
    * **channels**:     指定通道
      * 通道编号需要用中括号括起来;
      * 输入图像是灰度图时，它的值是[0];
      * 彩色图像可以是[0], [1], [2] 分别对应通道 B, G, R
    * **mask**:           掩码图像
      * 统计整幅图像的直方图， 设为None
      * 统计图像某一部分的直方图时，需要掩码图像
    * **histSize**:       BINS的数量
      * 需要用中括号括起来
    * **ranges**:         现估值范围RANGES
    * **accumulate**: 累计标识(可选参数)
      * 默认值为false
      * 如果被设置为ture， 则直方图在开始分配时就不会被清零。
      * 该参数允许从多个对象中算计单个直方图， 或者用于实时更新直方图
      * 多个直方图的累积结果，用于对一组图像计算直方图。


    例：hist = cv2.calcHist([img],[0],None,[256],[0,255]



### 42. 绘制OpenCV统计直方图

​     使用matplotlib 中的pyplot包

​     import matplotlib.pyplot as plt

​         x = [1,2,3,4,5,6]

​         y = [0.3,0.4,2,5.3,4.5,4]

​         plt.plot(x,y)

   histb = cv2.calcHist([o].[0],None,[256],[0,256])

​    plt.plot(histb,color='b')

### 43. 使用掩膜的直方图

* *一般形式：* hist = cv2.calcHist(images, channels, mask, histSize, ranges)
  * **hist**:             直方图   (返回的直方图，是一个二维数组)
  * **images**：     原始图像
  * **channels**:     指定通道
    * 通道编号需要用中括号括起来;
    * 输入图像是灰度图时，它的值是[0];
    * 彩色图像可以是[0], [1], [2] 分别对应通道 B, G, R
  * **mask**:           掩码图像
    * 统计整幅图像的直方图， 设为None
    * 统计图像某一部分的直方图时，需要掩码图像
  * **histSize**:       BINS的数量
    * 需要用中括号括起来
  * **ranges**:         现估值范围RANGES
  * accumulate: 累计标识(可选参数)

### 44. 掩膜原理及演示

* 或（or) : 并联    与(and) ： 串联

  mask = np.zeros(800, np.unit8)

  mask[300: 500, 300,500] = 255

  生成掩膜图形

  ​    计算结果 = cv2.bitwise_and(图像1， 图像2)

  masked_img = cv2.bitwise_and(img, mask)

### [实验]模板匹配 

* 模板匹配就是在整个图像区域发现与给定子图像匹配的小块区域。
* 所以模板匹配首先需要一个模板图像T（给定的子图像）
* 另外需要一个待检测的图像-源图像S
* 工作方法，在带检测图像上，从左到右，从上向下计算模板图像与重叠子图像的匹配度程度越大，两者相同的可能性越大

   

```
T-灰度变换-二值化-轮廓-外接矩形 
信用卡-灰度-二值化-顶帽-梯度-闭操作-闭操作-轮廓-二值化-切分
模板匹配
```

***

* **imutils : 工具包**

> imutils提供一系列便捷功能进行基本的图像处理功能，如平移，旋转，缩放，骨架，matplotlib图像显示，排序的轮廓，边缘检测

***



* #### argparse : 命令行参数解析包

#####  1. 创建一个解析器

>  使用 `argparse` 的第一步是创建一个 `ArgumentParser` 对象：

```python
>>> parser = argparse.ArgumentParser(description='Process some integers.')
```

##### 2. 添加参数

>  给一个 `ArgumentParser` 添加程序参数信息是通过调用 `add_argument()`方法完成的。通常，这些调用指定 `ArgumentParser` 如何获取命令行字符串并将其转换为对象。这些信息在 `parse_args()`调用时被存储和使用

##### 3. 解析参数

`ArgumentParser` 通过 `parse_args()` 方法解析参数。它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。在大多数情况下，这意味着一个简单的 `Namespace` 对象将从命令行参数中解析出的属性构建：

```python
>>> parser.parse_args(['--sum', '7', '-1', '42'])
Namespace(accumulate=<built-in function sum>, integers=[7, -1, 42])
```

在脚本中，通常 `parse_args()`会被不带参数调用，而 `ArgumentParser`将自动从 `sys.argv` 中确定命令行参数。

***



```
# 根据坐标提取每一个组
group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
cv_show('group',group)
```

其中“+5”，"-5"是为了不把特征值丢掉

***



* **cv2.findContours()函数**

```python
# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓

ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3) 
cv_show('img',img)
print (np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0] #排序，从左到右，从上到下
digits = {}
```

* **cv2.findContours(）**

​              image, contours, hierarchy=cv2.findContours(image,mode,method)

* 参数

  * mage ，修改后的原始图像
  * contours , 轮廓
  * hierarchy , 图像的拓扑信息（轮廓层次）
  *    image , 原始图像                  
  * ​    mode , 轮廓检索模式   
  *    method , 轮廓的近似方法    

* **cv2.drawContours( ）**

  ​      r=cv2.drawContours(),contours,contourldx,color[,thickness])

  * r                    :   目标图像，直接修改目标的像素点，实现绘制。
  * o                   :   原始图像
  * contours       :   需要绘制的边缘数组。
  * contourldx    :   需要绘制的边缘索引，如果全部绘制则为一1。
  * color             :   绘制的颜色，为BGR格式的Scalar。
  * thickness      :   可选，绘制的密度，即描绘轮廓时所用的画笔粗细。

  ***

  

* **refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]**

```python
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes
```

***

```python
for (i, c) in enumerate(refCnts):
   # 计算外接矩形并且resize成合适大小
   (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# 每一个数字对应每一个模板
	digits[i] = roi
```

`enumerate`: 

> enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

```python
enumerate(sequence, [start=0])
```

*参数*

- sequence -- 一个序列、迭代器或其他支持迭代对象。
- start -- 下标起始位置。

*返回值*

​     返回 enumerate(枚举) 对象。

`cv2.boundingRect(img)`

>  得到包覆此轮廓的最小正矩形

***

* **cv2.getStructuringElement( )**

 ```python
# 初始化卷积核读入信用卡图像
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))#根据字体大小设定核的大小
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
 ```

> 返回指定形状和尺寸的结构元素。
>
>    例： kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
>
> ​          函数的第一个参数表示内核的形状，有三种形状可以选择。
>
> ​          矩形：MORPH_RECT;
>
> ​          交叉形：MORPH_CROSS;
>
> ​          椭圆形：MORPH_ELLIPSE;
>
> ​          第二和第三个参数分别是内核的尺寸以及锚点的位置。一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得
>
> getStructuringElement函数的返回值: 对于锚点的位置，有默认值Point（-1,-1），表示锚点位于中心点。element形状唯一依赖锚点位置，其他情况下，锚点只是影响了形态学运算结果的偏移。

***

```python

#礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel) #明亮的区域提取出来
cv_show('tophat',tophat) 
# 梯度运算
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, #ksize=-1相当于用3*3的
	ksize=-1)
```

​	**礼帽（image) = 原始图像（image) - 开运算（image)**

- *得到噪声图像*

  **函数morphologyEX**

​	result = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)

* result ,礼帽结果                       | *img*,源图像

  ​                                                   | cv2.MORPH_TOPHAT,礼帽

  ​									      	 	| kernel,卷积核        例：kernel = np.ones((5,5),np.unit8)

   **dst = cv2.Sobel( src, ddepth, dx, dy, [ ksize ])**

​    dst, 计算结果        src, 原始图像

​						   		ddepth, 处理结果图像深度   

​							 	  dx, x轴方向

​						   		dy, y轴方向

​							    	ksize, 核大小

***

```python
#通过闭操作（先膨胀，再腐蚀，去除内部空洞）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel) 
cv_show('gradX',gradX)
#THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
cv_show('thresh',thresh)
```

>​	**闭运算（image） = 腐蚀（膨胀（image））**
>
>* *先膨胀，后腐蚀*
>
>* *它有助于关闭前景物体内部的小孔，或物体上的小黑点*
>
>**函数morphologyEX**
>
>closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
>
>*closing* ,闭运算结果                               | *img*,源图像
>
>​                                                 | cv2.MORPH_CLOSE,闭运算
>
>​									     		| kernel,卷积核        例：kernel = np.ones((5,5),np.unit8)

> **opencv二值化函数** threshold(src_gray,dst,threshold_value,max_BINARY_value,threshold_type)，threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU)
>
> 这里二值化，即图像像素值变成0或255，THRESH_OTSU是确定阈值分割点，这个是库函数确定的
>
> Ostu方法又名最大类间差方法，通过统计整个图像的直方图特性来实现全局阈值T的自动选取
>
> 算法步骤：
>
> 1)  先计算图像的直方图，即将图像所有的像素点按照0~255共256个bin，统计落在每个bin的像素点数量
>
> 2)  归一化直方图，也即将每个bin中像素点数量除以总的像素点
>
> 3)  i表示分类的阈值，也即一个灰度级，从0开始迭代
>
> 4)  通过归一化的直方图，统计0~i 灰度级的像素(假设像素值在此范围的像素叫做前景像素) 所占整幅图像的比例w0，并统计前景像素的平均灰度u0；统计i~255灰度级的像素(假设像素值在此范围的像素叫做背景像素) 所占整幅图像的比例w1，并统计背景像素的平均灰度u1；
>
> 5)  计算前景像素和背景像素的方差 g = w0*w1*(u0-u1) (u0-u1)
>
> 6)  i++；转到4)，直到i为256时结束迭代
>
> 7）将最大g相应的i值作为图像的全局阈值

***



* **模板匹配函数**：

```python
# 在模板中计算每一个得分
for (digit, digitROI) in digits.items():
   # 模板匹配
   result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF)
   (_, score, _, _) = cv2.minMaxLoc(result)
   scores.append(score)
```

matchTemplate(InputArray image, InputArray templ, OutputArray result, int method);

        image：输入一个待匹配的图像，支持8U或者32F。
    
        templ：输入一个模板图像，与image相同类型。
    
        result：输出保存结果的矩阵，32F类型。
    
        method：要使用的数据比较方法。
**result**:

​    result是一个结果矩阵，假设待匹配图像为 **I**，宽高为(W,H)，模板图像为 **T**，宽高为(w,h)。那么result的大小就为(W-w+1,H-h+1) 。



**method**: 

<img src="https://img-blog.csdnimg.cn/2019122516072160.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9uaWNraGFuLmJsb2cuY3Nkbi5uZXQ=,size_16,color_FFFFFF,t_70" alt="img" style="zoom: 33%;" />

>   方差匹配方法：完全匹配会得到1， 完全不匹配会得到0。
>
>     归一化方差匹配方法：完全匹配结果为0。   
>    
>     相关性匹配方法：完全匹配会得到很大值，不匹配会得到一个很小值或0。
>    
>     归一化的互相关匹配方法：完全匹配会得到1， 完全不匹配会得到0。相关系数匹配方法：完全匹配会得到一个很大值，完全不匹配会得到0，完全负相关会得到很大的负数。   
>    
>     归一化的相关系数匹配方法：完全匹配会得到1，完全负相关匹配会得到-1，完全不匹配会得到0。

### 48. matplotlib pyplotshow函数的使用

* **imshow**

`           imshow ( X, cmap = None )`

* * x                     : 要绘制的图像
  * cmap             : colormap, 颜色色谱， 默认为RGBA(A) 颜色空间

* 显示问题

  * 灰度图像

    colormap, 颜色图谱，默认为RGB(A)颜色空间，使用参数`cmap = plt.cm.gray`

  * 彩色图像

    colormap,颜色图谱，默认为RGB(A)颜色空间，如果使用opencv读入的图像，默认空间为BGR, 需要调整色彩空间为RGB

### 49. 直方图均衡化对比

在一个窗口内显示四个子窗口，显示原始图像，原始图像均衡化直方图等

```python
import matplotlib.pyplot as plt
```

* **subplot**

  `subplot ( nrows, ncols, plot_number )`

  * nrows            : 行数
  * ncols             : 列数
  * plot_number: 窗口序号

* **imshow**

  `imshow ( X, cmap = None )`

  * x                     : 要绘制的图像
  * cmap             : colormap, 颜色色谱， 默认为RGBA(A) 颜色空间

```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('image\\boat.bmp',cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(img)
plt.subplot(221)
plt.imshow(img.cmap = plt.cm.gray).plt.axis('off')
plt_subplot(222)
plt.imshow(equ.cmap = plt.cm.gray).plt.axis('off')
plt_subplot(223)
plt.hist(img.ravel(),256)
plt_subplot(224)
plt.hist(equ.ravel(),256)
```



### 50. 傅里叶变换的理论基础

* 时域角度

  横坐标是**时间**

* 频域角度

  横坐标是**频率**的倒数

* 任何连续周期信号，可以由一组适当的正弦曲线组合而成。——傅里叶

* 傅里叶变换：从时间角度看的信号可以转换为从频率角度看的信号，两者同时还是可逆的。

### 51.numpy实现傅里叶变换

`numpy.fft.fft2`

* 实现傅里叶变换
* 返回一个复数数组（ complex ndarray )

`numpy.fft.ffushift`

* 将零频率分量转移到频谱中心

`20*np.log(np.abs(fshift))`

* 设置频谱的范围
* （给复数数组重新标定范围，设置区间，转换成灰度图像，映射到0~255之间）

`````````python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image\\lena.bmp',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
result = 20*np.log(np.abs(shift))
plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('original')
plt.axis('off')
plt.subplot(122)
plt.imshow(result, cmap = 'gray')
plt.title('result')
plt.axis('off')
plt.show()
`````````

* ##### 注意

  * 傅里叶得到低频、高频信息，针对低频、高频处理能够实现不同的目的。
  * 傅里叶过程是可逆的，图像经过傅里叶变换、逆傅里叶变换后，能够恢复到原始图像
  * 在频域对图像进行处理，在频域的处理会反映在逆变换的图像上

   

### 52. numpy实现逆傅里叶变换

`numpy.fft.ifft2`

* 实现逆傅里叶变换，返回一个复数数组（complex ndarray)

`numpy.fft.ifftshifr`

* fftshift函数的逆函数
* （将整个图像的低频移到左上角）

`iimg = np.abs( 逆傅里叶变换结果 )`

* 设置值的范围

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image\\boat.bmp',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)  #将低频移到中心
ishift = np.fft.ifftshift(fshift)  #将中心移到左上角
iimg = np.fft.fft2(ishift) #返回的是复数，无法显示
iimg = np.abs(iimg) #转换为绝对值
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('original'),plt.axis('off')
plt.subplot(122),plt.imshow(iimg, cmap = 'gray')
plt.title('iimg'),plt.axis('off')
plt.show()
```

### 53. 高通滤波演示

* **低频、高频**

  * 低频对应图像内变化缓慢的灰度分量。例如，在一幅大草原的图像中，低频对应着广袤的颜色趋于一致的草原。
  * 高频对应图像内变化越来越快的灰度分量，是由灰度的尖锐过渡造成的。例如，在一幅大草原的图像中，其中狮子的边缘等信息。
  * 衰减高频而通过低频，低通滤波器，将模糊一幅图像。
  * 衰减低频而通过高频，高通滤波器，将增强尖锐细节，但是会导致图像的对比度降低。

* **滤波**

  * 接受（通过）或者拒绝一定频率的分量
  * 通过低频的滤波器称为低通滤波器
  * 通过高频的滤波器称为高通滤波器

* **频域滤波**

  * 修改傅里叶变换以达到特殊目的，然后计算 IDFT 返回到图像域。

    * [IDFT]:逆傅里叶变换

  * 特殊目的： 图像增强、图像去噪、边缘检测、特征提取、压缩、加密等。

* 高通滤波器（去掉低频）

  ``` python
  rows, cols = img.shape
  crow,ccol = int(rows/2),int(cols/2)
  fshift[crow - 30:corw + 30, ccol - 30:ccol + 30] = 0
  ```

  ```python
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt
  img = cv2.imread('image\\lena.bmp',0)
  f = np.fft.fft2(img)
  fshift = np.fft.fftshift(f)  #将低频移到中心
  rows, cols = img.shape
  crow,ccol = int(rows/2),int(cols/2)
  fshift[crow - 30:corw + 30, ccol - 30:ccol + 30] = 0
  ishift = np.fft.ifftshift(fshift)  #将中心移到左上角
  iImg = np.fft.fft2(ishift) #返回的是复数，无法显示
  iImg = np.abs(iImg) #转换为绝对值
  plt.subplot(121),plt.imshow(img,cmap = 'gray')
  plt.title('original'),plt.axis('off')
  plt.subplot(122),plt.imshow(iimg, cmap = 'gray')
  plt.title('iimg'),plt.axis('off')
  plt.show()
  ```

### 54. OpenCV实现傅里叶变换

`返回结果 = cv2.dft(原始图像， 转换标识)`

* 返回结果
  * 是双通道的
  * 第1个通道是结果的实数部分
  * 第2个通道是结果的虚数部分
* 原始图像： 输入图像要首先转换成np.float32格式。 np.float32(img)
* 转换标识:  flags = cv2.DFT_COMPLEX_OUTPUT, 输出一个复数阵列。

`返回值 = cv2.magnitude( 参数1， 参数2 )`

* 计算幅值

  * 参数1： 浮点型X坐标值， 也就是实部

  * 参数2： 浮点型Y坐标值， 也就是虚部

    $dst( I ) =  \sqrt{x(I) ^2 + y( I ) ^2} $

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image\\lena.bmp',0)
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT) #傅里叶变换
dfshift = np.fft.fftshift(dft)  #平移
result = 20*np.log(cv2.magnitude( dftShift[:,:,0]， dftShitft[:,:,1] ))  #计算幅度
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('original'),plt.axis('off')
plt.subplot(122),plt.imshow(iimg, cmap = 'gray')
plt.title('result'),plt.axis('off')
plt.show()
```



### 55. OpenCV实现逆傅里叶变换

`返回结果 = cv2.idft( 原始数据 )`

* 返回结果： 取决于原始数据的类型和大小
* 原始数据： 实数或复数均可

`返回值 = cv2.magnitude ( 参数1， 参数2 )`

* 计算幅值
  * 参数1： 浮点型X坐标值， 也就是实部
  * 参数2： 浮点型Y坐标值， 也就是虚部

​                $dst( I ) =  \sqrt{x(I) ^2 + y( I ) ^2} $

`numpy.fft.ifftshift`

* fftshift函数的逆函数

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image\\lena.bmp',0)
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT) #傅里叶变换
dfshift = np.fft.fftshift(dft)  #平移
ishift = np.fft.ifftshift(dfshift) #移动
iImg = cv2.idft(ishift)
iImg = cv2.magnitude( iImg[:,:,0]， iImg[:,:,1] ))  #计算幅度
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('original'),plt.axis('off')
plt.subplot(122),plt.imshow(iimg, cmap = 'gray')
plt.title('inverse'),plt.axis('off') # 显示逆变换结果
plt.show()
```

### 56. 低通滤波示例

低通滤波器 x 频谱图像 = 滤波结果图像

* 生成代码

  ```python
  rows, cols = img.shape
  crow,ccol = int(rows/2),int(cols/2)
  mask = np.zeros((rows, cols, 2), np.unit8)
  mask[crow - 30:corw + 30, ccol - 30:ccol + 30] = 1
  ```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image\\lena.bmp',0)
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT) #傅里叶变换
dfshift = np.fft.fftshift(dft)  #平移
rows, cols = img.shape
crow,ccol = int(rows/2),int(cols/2)
mask = np.zeros((rows, cols, 2), np.unit8) #构造掩膜图像
fshift[crow - 30:corw + 30, ccol - 30:ccol + 30] = 1
fShift = dftShift*mask #滤波
ishift = np.fft.ifftshift(fShift)  #将中心移到左上角
iImg = cv2.idft(ishift) #返回的是复数，无法显示
iImg = cv2.magnitude( iImg[:,:,0]， iImg[:,:,1] ) #计算幅度
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('original'),plt.axis('off')
plt.subplot(122),plt.imshow(iimg, cmap = 'gray')
plt.title('iimg'),plt.axis('off')
plt.show()
```

### [实验] 图像处理特征检测

* 应用： 机器人运动规划

* 如果用局部特征来表示的话，这些特征应该具有什么样的特性？

  * 尺度不变性

  #### 场景识别

  1. 视觉局部特征

     * 不变性：一种基于尺度空间的、对图像缩放、旋转甚至仿射变换保持不变性的图像局部特征描述方法。

     * 优点

       * 独特性(Distinctiveness)
       * 不变性(Invariant)

     * 1. 关键点(KeyPoints)
       2. 特征描述器(Descriptor)

     * Key Points -  Harries

       * 在两个方向上有比较激励的变化，角点

     * DOG算子

       *对高斯算子的差分*

       * 图像尺度空间表示

         高斯卷积核

         $G(x, y, \sigma) = \frac{1}{2 \pi\sigma^2}e^{-(x^2 + y^2)/2\sigma^2}$

         

       * 一幅二维图像，不同尺度在尺度空间中的表示可由图像与不同的高斯核卷积得到

         $$L(x,y,z) = G(x, y, \sigma) * L(x, y)$$

       * DOG(Difference-Of-Gaussian)算子

         $$D（x,y. \sigma) = L(x, y, k\sigma) - L(x,y, \sigma)$$

       * 在图中找一个极值点：同一个点在不同差分图像上，相邻的九个点都是最大的

       

       * ##### searching maximum point

         * 在DOG(Difference-Of-Gaussian)尺度空间中检测局部极值以作为特征点

       * ##### feature filtering

         * 去除对比度低的点

         * 去除边界点

           $$H = \begin{bmatrix}D_{xx} &D_{xy}\\D_{xy}&D_{yy}\end{bmatrix}$$

           $$\frac{Tr(H)^2}{Det(H)} < \frac{(r+1)^2}{r}$$

         * 亚像素级定位

           $$D(X) = D + \frac{\partial D^T}{\partial X} X + \frac{1}{2}X^T\frac{\partial^2 D}{\partial X^2}X$$

       * ##### Orientation of the features

         * 利用特征点邻域像素的梯度方向分布特性为每个关键指定方向参数
         * 以特征点为中心的邻域窗口内用直方图统计邻域像素的梯度方向
         * 梯度直方图的范围是0~360度，其中每10度一个柱，总共36个柱
         * 每个特征点有三个信息：**位置、所处尺度、方向**。由此可以确定一个SIFT特征区域
         * 现在的问题：怎么描述特征

       * ##### feature descriptor

         * 以关键点为中心取8 $\times$ 8的窗口，把这个窗口切成$2 \times 2$的子窗口，然后统计每个子窗口的方向直方图

         ##### Main direction of KeyPoints

         * 确定关键点的方向采用梯度直方图统计法，统计以关键点为原点，一定区域内的图像像素点对关键方向生成所作的贡献。
         * 关键点的主方向与辅方向
           * 关键点==主方向==：极值点周围区域梯度直方图的主峰值也是特征点方向
           * 关键点==辅方向==：在梯度方向直方图中，当存在另一个相当于主峰值80%能量的峰值时，则将这个方向认为是该关键点的辅方向
           * 这可以增强匹配的鲁棒性，Lowe的论文指出大概有15%关键点具有多方向，但这些点对匹配的稳定性至为关键。

         

         ##### SIFT的改进——SURF

         * SIFT

           通过SIFT算法，在大小方向都改变的情况下，仍能找到特征.

           但由于不具有实时性，所以有了SURF

         * SURF

         * DOG与Hession矩阵

           更快

           把二阶微分后的图像进行近似

           也是求极值，不过是在矩阵上求极值

           让图像保持不变，增大模板，来获取不同尺度上特征

         * 特征方向计算

         * 特征描述

           利用haar小波，x方向，y方向

           16*4 = 64维向量

           * ORB

             $$m_{pq} = \sum_{x,y}x^py^qI(x,y)$$

           Feature Matching

           ​    穷举匹配

           * 模板图中关键点描述子：
           
         * 实时图中关键点描述子:
           
         * 关键点的匹配可以采用穷举法来完成，但是耗费时间太多，一般采用kd数的数据结构来完成搜索。
           
           搜索的内容是以模板图像的关键点为基准，搜索与目标图像的特征点最邻近的原图像特征点和次邻近的原图像特征点。
           
           kd数是一个平衡二叉树
           
           
           
             
           
             
           
### 57. 卷积神经网络

 * Classification 分类

 * Retrieval 相关推荐

 * Detection  图像检测[ 分类和回归 ]

   * self-driving cars 
     *  GPU图像处理单元——显卡[显存] Titan X 8000 12G/ Tegra X1 1080 5000 8G 

 * Segmentation 分割

   > 应用：
   >
   > * 人脸识别
   > * 姿势识别
   > * 标志识别
   > * 手写字体识别
   > * 图像变换

   ##### 卷积神经网络

   神经网络——>+深度  = 卷积神经网络

   [ INPUT - CONV - RELU - POOL - FC ]

   * 输入层
   * 卷积层
   * 激活函数
   * 池化层
   * 全连接层

   > 32 x 32 x 3 image ——> 5 x 5 x 3 filter  $ \omega $ (卷积核)
   >
   > 1. 把原图分成很多小块
   > 2. 卷积核来给每一个小块进行特征提取
   > 3. 把小块提取出一个值[ 特征值 ]
   > 4. 得到一个特征值组成的特征图
   > 5. 注意：最后维度保持一致，深度不变
   > 6. 使用其他更多filter, 有n个filter就有n层

   * **卷积过程**

   filter size ： h w

   原始输入：外面填充了一层0

   卷积大小和filter大小相同

   卷积中心和图像中心对齐（所以填充才能对齐）

   内积：对应位置数字相乘，再求和，得出一个数字，值

   RGB三层和3个filter求的值=三个数相加，再加一个b (1, bias)得到的值，填充到结果图像中  [ $\omega x + b $ ]

   方块（窗口）滑动：实际：原图分割成小块，同时计算

   * **滑动步长stride : **
     * 7  x 7 ——> 3 x 3
     * stride 2 小一点，特征更多一点，更丰富
     * 太小，stride = 1, 能得到更多特征，计算量大，得不偿失
     * 太大，漏掉中间的特征
     * stride小于等于卷积核的大小
   * 这样的过程，有些信息多次利用了，怎么多次利用边缘信息？
   * **pad **1 ： 边缘填充
     * 例： 5 x 5 ——> 7 x 7, 在原始值上加一层padding 
     * 使得原始图像的边缘值更多利用
     * 为什么填充的是0？0没有意义的，没有学习意义，没有影响，不考虑padding项，其他值，由于是边缘，也每什么意义
     * pad = n , 加上 n 圈 0

   >* Input  = 7 x 7
   >* Filter = 3 x 3
   >* Pad  = 1
   >* Output  = ? [w x h]

   多个filter时，要大小一样的 



### [实验] 物体识别dnn_blob 

blob——难以名状的一团

> dnn模块使用caffe模型
> 1、通过readNet()函数加载caffe模型
> 2、读取图像，并将调用blobFromImage，将图像转换为4维的blob，其中mean= cv::Scalar(125, 123, 124)。
> 3、设置网络的输入数据net.setInput(blob)
> 4、执行网络推断Mat prob = net.forward()
> 5、获取推断结果列表中概率最大的索引，得到图像的标签，将标签转换为图像识别结果。对于32x32的rgb图像，在没有GPU的情况下，resnet56推断耗时5ms左右。
> ————————————————
> [原文链接](https://blog.csdn.net/zhongqianli/article/details/85691361)

##### blob_from_images.py

1. 导入包 

   ```python
   # 导入工具包
   import utils_paths
   import numpy as np
   import cv2
   
   ```

2. 标签文件处理

   ```python
   # 标签文件处理
   rows = open("synset_words.txt").read().strip().split("\n")
   classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
   ```

   strip: 丢空格

   classes： 把种类的内容取出来

3. Caffe所需配置文件

   ```python
   net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
   	"bvlc_googlenet.caffemodel")
   ```

   * **cv2.dnn.readNetFromCaffe**: 使用“读取”方法从磁盘直接加载序列化模型, 即通过readNet()函数加载caffe模型

   * > 读取模型，一个网络模型文件，一个参数文件

   ==Caffe==:  Convolution Architecture For Feature Embedding (Extraction) —— 卷积 建筑架构 特征 嵌入 拔出 ——快速特征嵌入的卷积结构

   >Caffe: Deep Learning 工具箱， C++语言架构，CPU和GPU无缝交换，Python和matlab的封装

4. 图像路径

   ```python
   imagePaths = sorted(list(utils_paths.list_images("images/")))
   ```

5. 图像数据预处理

   ```python
   image = cv2.imread(imagePaths[0])
   resized = cv2.resize(image, (224, 224))
   # image scalefactor size mean swapRB 
   blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
   print("First Blob: {}".format(blob.shape))
   ```

   * > ```python3
     > cv2.dnn.blobFromImag
     > e 用于对输入网络的图像进行预处理，主要是三部分，1.减均值 2.缩放 3.通道变换(可选)，对于imageNet训练集而言，三通道均值为(104, 117, 123)
     > ```
   > ```
   > 
   > ```

   *  **cv2.dnn.blobFromImage**

     > * 函数原型：cv2.dnn.blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]])
     >
     > * 函数作用：对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道等，返回一个4通道的blob(blob可以简单理解为一个N维的数组，用于神经网络的输入)
     >
     > * 函数参数：
     >
     >   * **image**:输入图像（1、3或者4通道）
     >         可选参数
     >
     >   * **scalefactor**:图像各通道数值的缩放比例
     >     size:输出图像的空间尺寸,如size=(200,300)表示高h=300,宽w=200
     >   * **mean**:用于各通道减去的值，以降低光照的影响(e.g. image为bgr3通道的图像，mean=[104.0, 177.0, 123.0],表示b通道的值-104，g-177,r-123)
     >   * **swapRB**:交换RB通道，默认为False.(cv2.imread读取的是彩图是bgr通道)
     >   * **crop**:图像裁剪,默认为False.当值为True时，先按比例缩放，然后从中心裁剪成size尺寸
     >   * **ddepth**:输出的图像深度，可选CV_32F 或者 CV_8U.
     >
     > *  返回：
     >
     >   4维的数组

6. 得到预测结果

   ```python
   net.setInput(blob)
   preds = net.forward()
   ```

   > * 设置网络的输入数据net.setInput(blob)
   >
   > * 执行网络推断Mat prob = net.forward()
   >
   > * print(preds) preds相当于一个数组，就是我们将图片input网络后，网络吐出了一个记录了1000多个种类得分的数组

7. 排序

   ```python
   idx = np.argsort(preds[0])[::-1][0]
   text = "Label: {}, {:.2f}%".format(classes[idx],
   	preds[0][idx] * 100)
   cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
   	0.7, (0, 0, 255), 2)		
   ```

   > * cv2.putText()
   >
   >   将文本信息添加到图片上，参数分别是，图片，文本，呈现位置，字体，大小，颜色，颜色厚度

8. Batch数据制作

   数据是一组一组的

   1. 方法一样，数据是一个batch

      ```python
      for p in imagePaths[1:]:
      	image = cv2.imread(p)
      	image = cv2.resize(image, (224, 224))
      	images.append(image)
      ```

      

   2. blobFromImages函数

      ```python
      blob = cv2.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
      print("Second Blob: {}".format(blob.shape))
      ```

      cv2.dnn.blobFromImages比上面多了s，处理一组数据

   3. 获取预测结果

      ```python
      net.setInput(blob)
      preds = net.forward()
      for (i, p) in enumerate(imagePaths[1:]):
      	image = cv2.imread(p)
      	idx = np.argsort(preds[i])[::-1][0]
      	text = "Label: {}, {:.2f}%".format(classes[idx],
      		preds[i][idx] * 100)
      	cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
      		0.7, (0, 0, 255), 2)
      	cv2.imshow("Image", image)
      	cv2.waitKey(0)
      ```

      网络输入——前向传播——for循环

#####       bvlc_googlenet.prototxt

* googlenet 网络模型架构
* layer

### [实验]人脸识别

1. 离线安装dlib

<img src="O:\classTask\opencv\images\img20200521094000.png" alt="img20200521094000" style="zoom:50%;" />

2. 运行报错

   <img src="C:\Users\Ovis\AppData\Roaming\Typora\typora-user-images\image-20200521094137477.png" alt="image-20200521094137477" style="zoom: 67%;" />

   > 网上解决办法：之后发现是.whl的文件有残缺或者dlib的版本过旧导致的
   > 解决方法很简单：（所有包都适用）
   > 打开anaconda prompt或者cmd
   > 先卸载原本安装的包 pip uninstall dlib
   > 然后直接在线安装包 pip install dlib
   > 安装完成后直接使用命令python + import dlib查看是否成功
   > 有时会提示要先安装cmake包，按照上面的方法同理即可
   > ————————————————
   > CSDN博主「. 水怪」的原创文章
   > 原文链接：https://blog.csdn.net/weixin_42510892/article/details/97107674

   在线安装

   <img src="C:\Users\Ovis\AppData\Roaming\Typora\typora-user-images\image-20200521094655717.png" alt="image-20200521094655717" style="zoom:50%;" />

​       安装失败

> [https://pypi.python.org/pypi/dlib/19.6.0](https://link.zhihu.com/?target=https%3A//pypi.python.org/pypi/dlib/19.6.0) 
> 下载 dlib-19.6.0-cp36-cp36m-win_amd64.whl 成功安装 dlib 但是import 时候失败
>
> 尝试 **pip install dlib==[19.6.1](https://link.zhihu.com/?target=https%3A//pypi.python.org/pypi/dlib/19.6.1)**   成功import。
>
> 
>
> 作者：马蹄急
> 链接：https://www.zhihu.com/question/34524316/answer/350130733
> 来源：知乎
>

<img src="O:\classTask\opencv\md_images\image-20200521100952642.png" alt="image-20200521100952642" style="zoom:50%;" />

3. 再次运行

   报错结果

<img src="O:\classTask\opencv\md_images\image-20200521101214187.png" alt="image-20200521101214187" style="zoom: 67%;" />

<img src="O:\classTask\opencv\md_images\image-20200521101610041.png" alt="image-20200521101610041" style="zoom:67%;" />

4. 下载了各自各样的版本dlib

   依然报体同样的错误

5. install dilb  19.6.0 版本

   通过pip3 list 可以看到dlib在里面

   通过python3 import dlib执行成功

   <img src="O:\classTask\opencv\md_images\image-20200521111019452.png" alt="image-20200521111019452" style="zoom:50%;" />

6. 在IDE中依然运行失败

   换了个版本使用 19.7.0

   <img src="O:\classTask\opencv\md_images\image-20200521112913810.png" alt="image-20200521112913810" style="zoom:50%;" />

7. 运行报错

   命令行解析报错

   <img src="O:\classTask\opencv\md_images\image-20200521113055096.png" alt="image-20200521113055096" style="zoom:80%;" />

<img src="O:\classTask\opencv\md_images\image-20200611222505198.png" alt="image-20200611222505198" style="zoom: 33%;" />

8. 运行结果

   <img src="O:\classTask\opencv\md_images\image-20200612152001064.png" alt="image-20200612152001064" style="zoom:33%;" />

9. 代码理解

   ```
   # 导入工具包
   from collections import OrderedDict
   import numpy as np
   import argparse
   import dlib
   import cv2
   ```

   * 参数设定

   ```
   # https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
   # http://dlib.net/files/
   # 参数
   # run configuration
   # --image images/liudehua.jpg
   # --shape-predictor shape_predictor_68_face_landmarks.dat
   
   ap = argparse.ArgumentParser()
   ap.add_argument("-p", "--shape-predictor", required=True,
                   help="path to facial landmark predictor")
   ap.add_argument("-i", "--image", required=True,
                   help="path to input image")
   args = vars(ap.parse_args())
   ```

   * 标注的点

     ​	OrdereDict 点位预测

   ```
   FACIAL_LANDMARKS_68_IDXS = OrderedDict([
       ("mouth", (48, 68)),
       ("right_eyebrow", (17, 22)),
       ("left_eyebrow", (22, 27)),
       ("right_eye", (36, 42)),
       ("left_eye", (42, 48)),
       ("nose", (27, 36)),
       ("jaw", (0, 17))
   ])
   
   FACIAL_LANDMARKS_5_IDXS = OrderedDict([
       ("right_eye", (2, 3)),
       ("left_eye", (0, 1)),
       ("nose", (4))
   ])
   ```

   * 把点值转换成np类型

   ```
   def shape_to_np(shape, dtype="int"):
       # 创建68*2
       coords = np.zeros((shape.num_parts, 2), dtype=dtype)
       # 遍历每一个关键点
       # 得到坐标
       for i in range(0, shape.num_parts):
           coords[i] = (shape.part(i).x, shape.part(i).y)
       return coords
   ```

   * 在overlay上进行凸包运算，连线，然后通过addWeight将两张图片按比例叠加

   ```
   
   def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
       # 创建两个copy
       # overlay and one for the final output image
       overlay = image.copy()
       output = image.copy()
       # 设置一些颜色区域
       if colors is None:
           colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                     (168, 100, 168), (158, 163, 32),
                     (163, 38, 32), (180, 42, 220)]
       # 遍历每一个区域
       for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
           # 得到每一个点的坐标
           (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
           pts = shape[j:k]
           # 检查位置
           if name == "jaw":
               # 用线条连起来
               for l in range(1, len(pts)):
                   ptA = tuple(pts[l - 1])
                   ptB = tuple(pts[l])
                   cv2.line(overlay, ptA, ptB, colors[i], 2)
           # 计算凸包
           else:
               hull = cv2.convexHull(pts)
               cv2.drawContours(overlay, [hull], -1, colors[i], -1)
       # 叠加在原图上，可以指定比例
       cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
       return output
   ```

   * dlib库检测出人脸

   ```
   
   # 加载人脸检测与关键点定位，http://dlib.net/
   # detector = dlib.get_frontal_face_detector()
   # 功能：人脸检测画框
   # 参数：无
   # 返回值：默认的人脸检测器
   detector = dlib.get_frontal_face_detector()
   predictor = dlib.shape_predictor(args["shape_predictor"])
   ```

   * 将高按成指定宽的比例统一缩放

   ```
   # 读取输入数据，预处理
   image = cv2.imread(args["image"])
   (h, w) = image.shape[:2]
   width = 500
   r = width / float(w)
   dim = (width, int(h * r))
   image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   ```

   * 将检测出的人脸部位提取ROI区域表示出来

   ```
   # 人脸检测
   rects = detector(gray, 1)
   
   # 遍历检测到的框
   for (i, rect) in enumerate(rects):
       # 对人脸框进行关键点定位
       # 转换成ndarray
       shape = predictor(gray, rect)
       shape = shape_to_np(shape)
   
       # 遍历每一个部分
       for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
           clone = image.copy()
           cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)
   
           # 根据位置画点
           for (x, y) in shape[i:j]:
               cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)
   
           # 提取ROI区域
           (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
   
           roi = image[y:y + h, x:x + w]
           (h, w) = roi.shape[:2]
           width = 250
           r = width / float(w)
           dim = (width, int(h * r))
           roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)
   
           # 显示每一部分
           cv2.imshow("ROI", roi)
           cv2.imshow("Image", clone)
           cv2.waitKey(0)
   
       # 展示所有区域
       output = visualize_facial_landmarks(image, shape)
       cv2.imshow("Image", output)
       cv2.waitKey(0)
   ```

   

   

   ### 58. 图像检索

   ##### 相似检索

   * 颜色、纹理、形状
   * 局部特征点
   * 词包（Bag Of Visual Word）

   ###### **相似颜色检索**

   > **算法结构：**
   >
   > * 目标： 实现基于*人类颜色感知*的相似排序
   > * 模块：颜色*特征* 提取 & 特征 *相似度* 计算
   >
   > **颜色特征提取**：
   >
   > * 目标：统计图片的*颜色成份* → *颜色聚类直方图*
   > * 方法： 使用*K-means++* 对图片*Lab* 像素值进行聚类
   >
   > **颜色特征相似度计算**
   >
   > * 颜色直方图距离
   >
   >   * EMD(Earth Mover Distance)
   >   * 两个图片的颜色特征直方图之间的视觉相似度
   >   * 检索结果的排序依据
   >
   > * 色差距离
   >
   >   * CIEDE2000
   >
   >   * Lab空间中两个颜色之间的视觉相似度
   >
   >   * EMD距离的基础距离
   >
   >     * EMD距离
   >       * 两个多维特征分布之间的非相似性度量
   >       * 基于针对单特征的地面距离
   >       * 传统运输问题
   >         * 场景：多对多分配
   >           * 物资运送：多个供应商→多个需求客户
   >           * 土堆搬运：多个土堆→多个土坑
   >         * 约束
   >           * 双方的节点总量相等
   >           * 不同节点之间的成本各异
   >         * 目标：完成分配的最小成本
   >
   >   * 色差容忍度(Tolerance)
   >
   >     * 概念：无法感知的色差
   >     * 计算：色差小于JND(Just-Noticeable-Difference)阈值
   >     * 前提：感知均匀的色差距离
   >
   >   * CIE1931颜色空间
   >
   >     * 容忍椭圆
   >
   >     * 非感知均匀
   >
   >       <img src="O:\classTask\opencv\md_images\image-20200613205029305.png" alt="image-20200613205029305" style="zoom: 67%;" />
   >
   >   * CIELab颜色空间
   >
   >     * 视觉感知均匀的颜色模型
   >     * 均匀性更好的距离CIEDE2000

   ###### 相似纹理检索

   > **算法结构**
   >
   > * 目标：实现基于人类纹理感知的相似排序
   > * 模块(与相似颜色检索类似)
   >   * 纹理特征提取
   >     * 特征空间： 多方向、多尺度Gabor滤波器组
   >     * 特征计算：Kmeans++聚类直方图
   >   * 特征相似度计算
   >     * 纹理聚类直方图：EMD
   >     * 纹理距离：L2
   > * Gabor滤波器组
   >   * 6频率(尺度)
   >     * 频率：1，2，3，4，5，6
   >     * 尺寸：25，35，49，69，97，137
   >   * 8方向
   >     * 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5 
   >   * Gabor文理征提取
   >     * 彩色图片灰度化
   >     * 提取灰度图的Gabor滤波器特征
   >       * 6频率(尺度）、8方向的Gabor，
   >       * 48个同尺寸的特征图
   >       * 每个像素对应48维的Gabor特征向量
   >     * 使用Kmeans++聚类所有像素Gabor特征
   >       * K值（10）根据数据集纹理复杂度而定
   >       * 使用KD-tree版加速
   >     * Gabor卷积操作加速
   >       * FFT
   >       * 采用"图片尺寸缩小替代Gabor模板尺寸增大"
   >     * Gabor纹理特征提取可视化

   ###### 相似形状检索

   > Phog形状特征相似度计算
   >
   > * 标准化欧式距离
   >
   >   * Si为样本集特征中每一维对应的标准差
   >
   >     $Dist(P,Q) = \sqrt{\Sigma_i(\frac{P_i - Q_i}{S_i})^2}$
   >
   >   * 直方图相交（Histongram Intersection）
   >
   >     $Sim(P,Q) = \Sigma^{i=n}_{i=1} min(P_i, Q_i)$
   >
   > Phog形状特征提取
   >
   > * 金字塔梯度方向直方图
   >
   >   * 网格：$1\times 1,  2\times 2,  4\times 4$
   >   * 直方图方向数量：9
   >   * 维数：189 = (1+4+16) $\times$ 9
   >
   >   ```mermaid
   >   graph LR
   >   A[图像]  --> B((灰度图))
   >   B --> C(Sobel梯度图)
   >   B --> D(Canny边缘图)
   >   C --> E(每个子区域的直方图)
   >   D --> E
   >   E --> F[级联得到PHOG特征向量]
   >   ```

   ###### 相似局部特征检索

   > **局部特征点特征提取**
   >
   > * 检测出所有
   >   * 局部特征点
   >   * 特征描述子
   > * SIFT特征点
   >   * SURF
   >   * Color SIFT
   >   * Affine SIFT
   > * SIFT描述子之间的相似度匹配
   >   * 基于欧式距离的最近邻
   >     * $d_1< d_2 <d_3 <d_4 < ... $
   >   * 比率条件
   >     * $d_1/d_2 < thresh$
   > * 图之间的相似度匹配
   >   * 两个图SIFT点集之间的匹配对数
   >   * 双向匹配

   ###### 视觉词汇的字典

   > * 由图片集的所有视觉词汇构成
   >   * 视觉词汇的物理含义未知
   >   * 不是现成，需要构建
   >     * 特征检测
   >       * 特征点：SIFT、SURF等
   >     * 特征表示
   >       * SIFT描述子、颜色、纹理等
   >     * 字典生成
   >       * Kmeans等聚类
   > * SIFT视觉词汇的字典应用
   >   * 利用SIFT算法提取图片集中所有视觉词汇
   >   * 利用Kmeans算法对所有词汇聚类，收缩为字典
   >   * 基于字典编码图片特征
   >     * 词汇频数直方图
   >     * 最邻近词汇
   >     * 特征相似距离：L1、L2

   ##### 大数据集的索引加速

   * KD-tree
   * 局部敏感哈希（Locality Sensitive Han）