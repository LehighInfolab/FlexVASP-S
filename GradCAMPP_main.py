import os.path
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gradcam import *
# import cv2
import matplotlib.pyplot as plt
import random
# model
from DataSets_Preprocessing.DataSets_Preprocessing import DeepVaspS
from model.model import ConvNeuralNet
import argparse
import re
import time
import cv2
from grad_cam.pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from grad_cam.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from grad_cam.pytorch_grad_cam.utils.image import (show_cam_on_image,preprocess_image)
from torchvision.models import resnet50



if __name__ == '__main__':
    start_time = time.time()
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        print("torch.cuda.is_available:", torch.cuda.is_available())
        torch.cuda.manual_seed_all(seed)

    # Enolase
    EnolaseMuconate = ["2PGW", "1BKH", "3DGB", "3FJ4", "3I4K", "2ZAD", "3DEQ", "3DER", "1JPM", "1TKK"]
    EnolaseMandelate = ["2OX4", "1MDR", "1mdl", "1mns", "2OG9"]
    Enolase = ["1E9I", "1PDY", "1IYX", "1EBH", "1TE6", "1W6T", "2XSX", "2PA6", "3OTR", "3QN3"]

    # nonredundent enolase
    nonredunEnolaseMuconate = ["2PGW", "1BKH", "3DGB", "3I4K", "2ZAD", "1JPM"]  # 6
    nonredunEnolaseMandelate = ["2OX4", "1MDR", "2OG9"]  # 3
    nonredunEnolase = ["1E9I", "1PDY", "1IYX", "1EBH", "1TE6", "1W6T", "2XSX", "2PA6", "3OTR", "3QN3"]  # 10

    # ENOLASE_LABELS_SET = [EnolaseMandelate, EnolaseMuconate, Enolase]
    ENOLASE_LABELS_SET = [nonredunEnolaseMandelate, nonredunEnolaseMuconate, nonredunEnolase]

    # draw PCA plot
    # pca_enolase(ENOLASE_LABELS_SET,num_sample=1000)

    SerProtTrypsin = ["2f91", "1fn8", "2eek", "1h4w", "1bzx", "1aq7", "1ane", "1aks", "1trn", "1a0j"]
    SerProtChymotrypsin = ["1eq9", "4cha", "1kdq", "8gch"]
    SerProtElastase = ["1elt", "1b0e"]
    SERPROT_LABELS_SET = [SerProtTrypsin, SerProtChymotrypsin, SerProtElastase]
    # pca_serineProt(SerProt_LABELS_SET)

    args = argparse.ArgumentParser(description='DeepVASP-S parameter')
    args.add_argument('-s', '--superfamily', default='Serprot', type=str,
                      help='config training superfamily dataset; Enolase or Serprot ')
    args.add_argument('-p', '--path', default='./saved_model/Serprot/', type=str,
                      help='config output model path ')
    args.add_argument('-t', '--test', default=0, type=int,
                      help='set test protein index')
    args.add_argument('-n', '--num_sample', default=500, type=int,
                      help='number of samples')
    args.add_argument('-th', '--toph', default=600, type=int,
                      help='top h values in saliency map')


    args = args.parse_args()
    # print(args.superfamily,args.path,args.test)
    print(args)
    superfamily = args.superfamily
    # LABELS_SET = ENOLASE_LABELS_SET
    if args.superfamily == "Enolase":
        LABELS_SET = ENOLASE_LABELS_SET
    elif args.superfamily == "Serprot":
        LABELS_SET = SERPROT_LABELS_SET


    proteinset=[y for x in LABELS_SET for y in x][args.test]
    print(proteinset)


    num_classes=3

    # data = data[:1]
    if args.superfamily == "Enolase":
        k = 15
    elif args.superfamily == "Serprot":
        k=25

    # npy_index = 0
    for npy_index in range(0,500):
        file_name = 'F:/PycharmProjects/DeepVASP_pytorch/DataSet/{}New/{}_union_{}/{}-union-{}.npy'.format(args.superfamily, proteinset, k, proteinset, str(npy_index))

        data =  np.load(file_name)
        if args.superfamily == "Enolase":
            data = np.reshape(data, (1, 1, 34, 29, 30))
        elif args.superfamily == "Serprot":
            data = np.reshape(data,(1 , 1, 35,38,43))

        data = torch.from_numpy(data).type(torch.FloatTensor)

        # data = torch.unsqueeze(data, 0)
        model = ConvNeuralNet(num_classes, superfamily=args.superfamily)

        if args.superfamily == "Enolase":
            model.load_state_dict(torch.load("./saved_model/Enolase/Union/best_model_fold1_Enolase_{}_Union15.ckpt".format(proteinset)))
        elif args.superfamily == "Serprot":
            model.load_state_dict(torch.load("./saved_model/Serprot/Union/best_model_fold1_Serprot_{}_Union25.ckpt".format(proteinset)))

        out = model(data)
        print(data.shape, out.shape)
        print(model, model.conv_layer1)

        #Set the Model to Evaluation Mode, In PyTorch, model.eval() disables dropout layers and batch normalization during inference, ensuring stable activations.
        model.eval()
        if args.superfamily == "Enolase":
            data_size = (34,29,30)
        elif args.superfamily == "Serprot":
            data_size = (35,38,43)
        # model_dict = dict(model_type='resnet', arch=model, layer_name='conv_layer1', input_size=data_size) #data_size=(28, 34,  30)
        # gradcampp = GradCAMpp(model_dict)
        model = resnet50(pretrained=True)
        target_layers = [model.layer4[-1]]
        print("target_layers:",target_layers)

        rgb_img = cv2.imread("F:\PycharmProjects\DeepVASP_pytorch\grad_cam\examples\\both.png", 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        targets = [ClassifierOutputTarget(281)]

        # Construct the CAM object once, and then re-use it on many images.
        with GradCAM(model=model, target_layers=target_layers) as cam:
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # You can also get the model outputs without having to redo inference
            model_outputs = cam.outputs

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(data, class_idx=0)

        print(mask.shape, logit.shape, logit)

        if args.superfamily == "Enolase":
            mask = mask.reshape((34 * 29 * 30))
            saliency = np.empty((34 * 29 * 30, 2))
        elif args.superfamily == "Serprot":
            mask=mask.reshape((35*38*43))
            saliency = np.empty((35 * 38 * 43, 2))

        for i in range((mask.shape)[0]):
            saliency[i]=[i,mask[i]]

        sorted_indices = np.argsort(saliency[:, 1])
        sorted_saliency = saliency[sorted_indices]

        top = []
        top_50_values = sorted_saliency[-50:]
        top_100_values = sorted_saliency[-100:]
        top_150_values = sorted_saliency[-150:]
        top_300_values = sorted_saliency[-300:]
        top_450_values = sorted_saliency[-450:]
        top_600_values = sorted_saliency[-600:]
        # top = [top_50_values,top_100_values,top_150_values,top_300_values,top_450_values,top_600_values]

        if args.toph == 50:
            top = [top_50_values]
        elif args.toph == 100:
            top = [top_100_values]
        elif args.toph == 150:
            top = [top_100_values]
        elif args.toph == 300:
            top = [top_300_values]
        elif args.toph == 450:
            top = [top_450_values]
        elif args.toph == 600:
            top = [top_600_values]

        # print(mask.min(), mask.max())
        # print(top_150_values)
        # print(heatmap.shape, cam_result.shape)
        #
        # heatmap = heatmap.permute(1, 2, 0).numpy()
        # plt.figure()
        # plt.imshow(heatmap)
        # plt.show()  # display it
        # filename="F:/PycharmProjects/DeepVASP_pytorch/DataSet/SerprotNew/{}-CNN/{}_1-025.SURF-clean.SURF.cnn".format(proteinset,proteinset)
        # filename = "./DataSet/{}New/{}-CNN/{}_1-025.SURF-clean.SURF.cnn".format(args.superfamily,proteinset,proteinset)
        #
        # with open(filename, encoding='gbk', errors='ignore') as voxel_file:
        #     lines = voxel_file.readlines()
        #     # Reads dimensions from
        #     #
        #     # line 18
        #     match = re.search(r'BOUNDS xyz dim: \[([0-9]+) ([0-9]+) ([0-9]+)]', lines[17])
        #     match_x_bounds = re.search(r'BOUNDS xneg/xpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[19])
        #     match_y_bounds = re.search(r'BOUNDS yneg/ypos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[20])
        #     match_z_bounds = re.search(r'BOUNDS zneg/zpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[21])
        #     x_dim = int(match.group(1))
        #     y_dim = int(match.group(2))
        #     z_dim = int(match.group(3))
        #     # data = np.zeros((x_dim * y_dim * z_dim))
        #     # indexdata = np.zeros((x_dim * y_dim * z_dim))
        #     # # Reads voxel data starting at line 25
        #     line_nums=[]
        #     for i in range(24, len(lines)):
        #         line_num, val = lines[i].split()
        #         line_num = int(line_num)
        #         val = float(val)
        #         #use line_nums to hold first index in cnn file
        #         line_nums.append(line_num)

        all_indexes = []
        for top_n_values in top:
            indexes=[]
            for n in range(len(top_n_values)):
                index = int(top_n_values[n][0])
                indexes.append(index)
            all_indexes.append(indexes)

        # m=1
        for topN in all_indexes:
            # keepindex= [x for x in topN if x in line_nums ]
            keepindex = [x for x in topN]
            if not os.path.exists('./Gradcampp/{}/{}'.format(args.superfamily,proteinset)):
                os.makedirs('./Gradcampp/{}/{}'.format(args.superfamily,proteinset))
            with open('./Gradcampp/{}/{}/{}_gradcam_{}_{}.txt'.format(args.superfamily,proteinset,proteinset,npy_index,args.toph), "w") as f:
                for n in range(len(keepindex)):
                    f.write(str(keepindex[n])+'\n')
                    # m=m+1
    end_time = time.time()

    print("Done")
    print("耗时: {:.2f}秒".format(end_time - start_time))
