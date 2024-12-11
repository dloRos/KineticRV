import os
import numpy as np
import cv2 as cv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from u2net import U2NET
from data_loader import RescaleT, ToTensor, ToTensorLab, SalObjDataset

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time, re


def normalize(x):
    ma = x.max()
    mi = x.min()
    out = (x - mi) / (ma - mi)
    return out


def save_output(image_path, y_pred, dsize=None, save_dir=r'./'):

    os.makedirs(save_dir, exist_ok=True)
    image_name = image_path.split(os.sep)[-1]
    save_path = os.path.join(save_dir, image_name)

    y_pred = y_pred.squeeze()  # (h,w)
    y_pred = y_pred.cpu().data.numpy()
    y_pred = (np.clip(y_pred, 0, 1) * 255).astype('uint8')
    if dsize is None:
        image_shape = cv.imread(image_path).shape
        y_pred = cv.resize(y_pred, dsize=(image_shape[1], image_shape[0]))
    else:
        y_pred = cv.resize(y_pred, dsize=(dsize[1], dsize[0]))
    cv.imwrite(save_path, y_pred)
    print('输出图片保存至:%s' % save_path)


def image_name_sort(image_names):
    num_list = []
    for image_name in image_names:
        if image_name.endswith('png') or image_name.endswith('jpg'):
            a = image_name.split('.')[-2]
            # num = int(a[len(no_num_image_name):])
            num = int(re.sub('\D', '', a))
            num_list.append(num)
    # print('old_image_names:', image_names, len(image_names))
    ids = np.argsort(num_list)
    new_image_names = []
    for i in range(len(num_list)):
        new_image_names.append(image_names[ids[i]])
    # print('new_image_names:', new_image_names), len(new_image_names)
    return new_image_names


def segment(image_dir, save_dir, model_weights, dsize=None, threshold=800000, save_format='png'):

    os.makedirs(save_dir, exist_ok=True)

    model = U2NET(in_ch=3, out_ch=1).cuda()
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    t1 = time.time()
    ImgTypeList = ['jpg', 'JPG', 'bmp', 'png', 'jpeg', 'rgb', 'tif']
    for dir_path, dir_names, image_names in os.walk(image_dir):
        # print(dir_path, dir_names, image_names)
        image_path_list = [os.path.join(dir_path, image_name) for image_name in image_names if
                           image_name.split('.')[-1] in ImgTypeList]
        image_path_list = image_name_sort(image_path_list)
        # print('image_path_list:', image_path_list)
        if len(image_path_list) > 0:

            test_salobj_dataset = SalObjDataset(img_name_list=image_path_list,
                                                lbl_name_list=[],
                                                transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
                                                )
            test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1)
            l = len(test_salobj_dataloader)
            t2 = time.time()

            print('开始转换:%s' % dir_path)
            for i, data_test in enumerate(test_salobj_dataloader):
                inputs_test = data_test['image']
                inputs_test = inputs_test.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_test = Variable(inputs_test.cuda())
                else:
                    inputs_test = Variable(inputs_test)
                # print('inputs_test:', inputs_test.shape, inputs_test.min(), inputs_test.max())
                d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)
                y_pred = d1[:, 0, :, :]
                y_pred = normalize(y_pred)

                save_output(image_path_list[i], y_pred, dsize, save_dir)

            print('已完成，耗时:%s' % (time.time() - t2))
    print('已完成全部，共耗时:%s' % (time.time() - t1))


def main():
    ang_le = ['6_0','6_100','18_0','18_100','30_0','30_100']
    iden_ty = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
    for _ang_le in ang_le:
        for _iden_ty in iden_ty:
            for nm in range(21):
                segment(image_dir=r'video_seq\%s\%s\%d'% (_ang_le,_iden_ty, nm),
                        save_dir=r'result\%s\%s\%d'% (_ang_le,_iden_ty, nm),
                        model_weights=r'gait.pth', dsize=None, threshold=100000, save_format='png')


if __name__ == '__main__':
    main()
