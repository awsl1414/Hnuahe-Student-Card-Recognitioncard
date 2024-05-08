# %%
import cv2
import numpy as np
from utils import cv_show, resize, sort_contours

# %%
# 读取模板图像
template = cv2.imread("./images/template/reference.png")
cv_show("template", template)

# %%
# 灰度图像
ref = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cv_show("ref", ref)


# %%
# 二值图像
ref = cv2.threshold(ref, 127, 255, cv2.THRESH_BINARY_INV)[1]
cv_show("ref", ref)

# %%
# 计算轮廓
refCnts, hierarchy = cv2.findContours(
    ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(template, refCnts, -1, (0, 0, 255), 3)
cv_show("template", template)

# %%
# 排序，从左到右，从上到下
refCnts = sort_contours(refCnts, method="left-to-right")[0]

# %%
digits = {}
# 遍历每一个轮廓
for i, c in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y : y + h, x : x + w]
    roi = cv2.resize(roi, (30, 50))
    cv_show("roi", roi)
    # 每一个数字对应每一个模板
    digits[i] = roi


# %%
# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# %%
# 读取输入图像，预处理
image = cv2.imread("./images/cards/card1.jpg")
temp = image.copy()

image = resize(image, width=600)
# 灰度
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show("image_gray", image_gray)

# %%
# 礼帽操作(原图-闭)，突出更明亮的区域
tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show("tophat", tophat)

# %%
# Sobel算子计算水平和垂直梯度
gradX = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=-1)  # ksize=-1相当于使用3x3的卷积核
gradY = cv2.Sobel(tophat, cv2.CV_32F, 0, 1, ksize=-1)

# 梯度值的绝对值
gradX_abs = np.absolute(gradX)
gradY_abs = np.absolute(gradY)

# 归一化处理
(minValX, maxValX) = (np.min(gradX_abs), np.max(gradX_abs))
gradX = 255 * ((gradX_abs - minValX) / (maxValX - minValX))
gradX = gradX.astype("uint8")

(minValY, maxValY) = (np.min(gradY_abs), np.max(gradY_abs))
gradY = 255 * ((gradY_abs - minValY) / (maxValY - minValY))
gradY = gradY.astype("uint8")

# print(np.array(gradX).shape)
cv_show("gradX", gradX)

# print(np.array(gradY).shape)
cv_show("gradY", gradY)

gradXY = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
# print(np.array(gradXY).shape)
cv_show("gradXY", gradXY)

# %%
# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradXY = cv2.morphologyEx(gradXY, cv2.MORPH_CLOSE, rectKernel)
cv_show("gradXY", gradXY)

# %%
# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradXY, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show("thresh", thresh)

# %%
# 再一次闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show("thresh", thresh)

# %%
# 计算轮廓
threshCnts, hierarchy = cv2.findContours(
    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)


# %%

cur_img = image.copy()
cv2.drawContours(cur_img, threshCnts, -1, (0, 0, 255), 2)  # 第25个轮廓是目标轮廓
cv_show("cur_img", cur_img)


# %%
locs = []

# 遍历轮廓
for i, c in enumerate(threshCnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # print(ar)
    # print(w)
    # print(h)

    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 8.0 and ar < 9.0:
        locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])
print(len(locs))

# %%
output = []

# 遍历每一个轮廓中的数字
for i, (gX, gY, gW, gH) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []

    # 根据坐标提取每一个组
    group = image_gray[gY - 5 : gY + gH + 5, gX - 5 : gX + gW + 5]
    group = resize(group, width=600)

    cur_img2 = group.copy()
    # cv_show("group", group)
    # 二值化
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show("group", group)

    # # 平滑处理
    group = cv2.medianBlur(group, 5)
    cv_show("group", group)

    # 颜色反转
    group = cv2.bitwise_not(group)
    cv_show("group", group)
    # 计算轮廓
    digitCnts, hierarchy = cv2.findContours(
        group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 绘制轮廓

    cv2.drawContours(cur_img2, digitCnts, -1, (0, 0, 255), 2)
    # cv2.drawContours(temp, digitCnts, -1, (0, 255, 0), 2)
    cv_show("cur_img2", cur_img2)
    # cv_show("group", group)

    # 排序，从左到右，从上到下
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y : y + h, x : x + w]
        roi = cv2.resize(roi, (32, 54))
        cv_show("roi", roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for digit, digitROI in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(
        image,
        "".join(groupOutput),
        (gX, gY - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )

    # 得到结果
    output.extend(groupOutput)

# %%
print("Student Card #: {}".format("".join(output)))
cv_show("image", image)

# %%
