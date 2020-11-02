import cv2
import numpy as np

# 第一步：使用cv2.VideoCapture读取视频
cap = cv2.VideoCapture('./obj_left_test1.mp4')
# 第二步：cv2.getStructuringElement构造形态学使用的kernel
# 第一个参数表示核的形状。可以选择三种
#         矩形：MORPH_RECT;
#         交叉形：MORPH_CROSS;
#         椭圆形：MORPH_ELLIPSE;
# 第二个参数表示核的尺寸。
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print(kernel)
# 第三步：构造高斯混合模型
#     history：用于训练背景的帧数，默认为500帧，如果不手动设置learningRate，history就被用于计算当前的learningRate，此时history越大，learningRate越小，背景更新越慢；
#     varThreshold：方差阈值，用于判断当前像素是前景还是背景。一般默认16，如果光照变化明显，如阳光下的水面，建议设为25,36，具体去试一下也不是很麻烦，值越大，灵敏度越低；
#     detectShadows：是否检测影子，设为true为检测，false为不检测，检测影子会增加程序时间复杂度，如无特殊要求，建议设为false；
model = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True) # 初始化只有这3个参数，但是还有其他参数需要设置，比如高斯个数

model.setNMixtures(5) # 设置混合高斯的个数
# model.setBackgroundRatio只读参数，默认是0.9，高斯背景模型权重和阈值，nmixtures个模型按权重排序后，
# 只取模型权重累加值大于backgroundRatio的前几个作为背景模型。也就是说如果该值取得非常小，
# 很可能只使用权重最大的高斯模型作为背景(因为仅一个模型权重就大于backgroundRatio了
# model.setVarThresholdGen 只读参数，默认是2.5*2.5，方差阈值，用于是否存在匹配的模型，如果不存在则新建一个

print(model.setfVarInit)



while 1:
    # 第四步：读取视频中的图片，并使用高斯模型进行拟合
    ret, frame = cap.read()
    # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
    # eg：mog->apply(src_YCrCb, foreGround, 0.005);
    # image 源图
    # fmask 前景（二值图像）
    # learningRate 学习速率，值为0-1,为0时背景不更新，为1时逐帧更新，默认为-1，即算法自动更新；
    fgmk = model.apply(frame, learningRate=-1)
    # 第五步：使用形态学的开运算做背景的去除
    fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
    # 第六步：cv2.findContours计算fgmk的轮廓
    contours = cv2.findContours(fgmk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for c in contours:
        # 第七步：进行人的轮廓判断，使用周长，符合条件的画出外接矩阵的方格
        length = cv2.arcLength(c, True)

        if length > 188:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 第八步：进行图片的展示
    cv2.imshow('fgmk', fgmk)
    cv2.imshow('frame', frame)

    if cv2.waitKey(150) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()