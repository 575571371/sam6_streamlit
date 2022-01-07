# 2021.4.27版本(test_xcc.py)

# import cv2
import os
import torch
import time
import shutil
# import numpy as np
from xcc_ps.dataloader_test_new import make_loader
from xcc_ps.models.ResNet18 import make_network

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MY_PATH = os.path.dirname(__file__)
# Pic_Path = r'D:/python61/pycharm_code/streamlit61/xcc_ps/data/61/'
# Pic_outPath = r'D:/python61/pycharm_code/streamlit61/xcc_ps/output/'
# Model_Path = r'D:/python61/pycharm_code/streamlit61/xcc_ps/50.pkl'
Pic_Path = os.path.join(MY_PATH, 'data/61/')
Pic_outPath = os.path.join(MY_PATH, 'output/')
Model_Path = os.path.join(MY_PATH, '50.pkl')


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def my_ps():
    start = time.perf_counter()

    # 读取照片
    testloader = make_loader(Pic_Path)
    # 加载模型
    model = make_network()
    model.eval()
    # map_location = lambda storage, loc: storage
    # model.load_state_dict(torch.load(Model_Path, map_location=map_location))
    model.load_state_dict(torch.load(Model_Path, map_location=lambda storage, loc: storage))
    # device = torch.device("cuda")
    device = torch.device("cpu")
    model.to(device)
    result = dict()
    # 将每张照片与其他照片进行比较，记录其优于其他照片的次数
    with torch.no_grad():
        for idx, (dataA, dataB, nameA, nameB) in enumerate(testloader):
            dataA, dataB = dataA.to(device), dataB.to(device)
            pred = model(dataA, dataB)
            predict = torch.argmax(pred, 1)  # predict表示分数大的图片的序号，即更好的那张照片的序号
            if predict.float() == 1.0:  # 后一张比前一张好
                # result[nameA[0]] = result.get(nameA[0], 0) - 1  # 避免出现：最差的那张照片没有计入字典
                result[nameA[0]] = result.get(nameA[0], 0)  # 差的那张照片不用减一
                result[nameB[0]] = result.get(nameB[0], 0) + 1
            else:
                result[nameA[0]] = result.get(nameA[0], 0) + 1
                result[nameB[0]] = result.get(nameB[0], 0)
    #             print("nameA = ,nameB = ,predict = ",nameA, nameB, predict)
    #         print(result)
    # return result

    # 获取结果并整理排序输出结果
    result_rev = sorted(result.items(), key=lambda item: item[1], reverse=True)
    for i in range(len(result_rev)):
        print(result_rev[i])
    model_stop = time.perf_counter()

    # 输出并保存优胜次数最高的前2张照片
    # 首先删除输出目录下的旧照片
    del_file(Pic_outPath)

    img = [0, 0]
    for i in range(len(img)):
        name_in = result_rev[i][0]
        name_out = 'out' + result_rev[i][0]
        shutil.copy(Pic_Path + name_in, Pic_outPath + name_out)
        # img[i] = cv2.imread(Pic_outPath + name_out)

    stop = time.perf_counter()
    print('模型计算耗时{:.2f}s,项目总耗时{:.2f}s'.format((model_stop - start), (stop - start)))

# if __name__ == "__main__":
#     start = time.perf_counter()
#
#     # 获取结果并整理排序输出结果
#     result = main()
#     result_rev = sorted(result.items(), key=lambda item: item[1], reverse=True)
#     for i in range(len(result_rev)):
#         print(result_rev[i])
#     model_stop = time.perf_counter()
#
#     # 输出并保存优胜次数最高的前2张照片
#     # 首先删除输出目录下的旧照片
#     del_file(Pic_outPath)
#
#     img = [0, 0]
#     for i in range(len(img)):
#         name_in = result_rev[i][0]
#         name_out = 'out' + result_rev[i][0]
#         shutil.copy(Pic_Path + name_in, Pic_outPath + name_out)
#         # img[i] = cv2.imread(Pic_outPath + name_out)
#
#     stop = time.perf_counter()
#     print('模型计算耗时{:.2f}s,项目总耗时{:.2f}s'.format((model_stop - start), (stop - start)))

# cv2.namedWindow("frame", 0)  # 0可调大小，注意：窗口名必须与imshow里面的窗口名一致
# cv2.imshow("frame", np.hstack([img[0], img[1]]))

# cv2.waitKey(0)
# cv2.destroyAllWindows()
