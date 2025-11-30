from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from imbalanced import ImbalancedDatasetSampler
# from torchsampler import ImbalancedDatasetSampler
# from model.model import Net
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
import statistics
from DataSets_Preprocessing.DataSets_PreprocessingUnion import DeepVaspS
from model.model import ConvNeuralNet
from trainer.train import Trainer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


def main(superfamily, LABELS_SET, leave_out_index, model_path, num_epochs, seed,args=None):
    batch_size = 64
    num_classes = 3
    learning_rate = 0.001
    # num_epochs = 3
    # leave_out_index = 2

    if superfamily == "Enolase":
        # test_data = DeepVaspS("./DataSet/EnolaseNew/", LABELS_SET, leave_out_index, Train=False)
        train_data = DeepVaspS("./DataSet/EnolaseNew/", LABELS_SET, leave_out_index, k=args.intersection, Train=True,num_sample=args.num_sample)
        test_data = DeepVaspS("./DataSet/EnolaseNew/", LABELS_SET,
                              leave_out_index,k=args.intersection,  Train=False, num_sample=args.num_sample)
    elif superfamily == "Serprot":
        # test_data = DeepVaspS("./DataSet/SerprotNew/", LABELS_SET, leave_out_index, Train=False)
        train_data = DeepVaspS("./DataSet/SerprotNew/", LABELS_SET, leave_out_index, k=args.intersection, Train=True,num_sample=args.num_sample)
        test_data = DeepVaspS("./DataSet/SerprotNew/", LABELS_SET,
                              leave_out_index, k=args.intersection, Train=False, num_sample=args.num_sample)

    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)
    # print(train_data.labels.unique(return_counts=True))

    # Set Loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # for train, val in kfold.split(xid, ys):
    model = ConvNeuralNet(num_classes, superfamily=superfamily)

    # Set optimizer with optimzer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01, momentum=0.9)

    device = torch.device('cuda:%d'%(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    # cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Perform cross-validation
    fold_num = 1
    val_loss = []
    val_acc = []
    for train_index, val_index in kfold.split(train_data.data, train_data.labels):
        # x_index = range(len(train_data))
        # y_label = train_data.labels.tolist()
        train_set = [train_data[i] for i in train_index]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                  sampler=ImbalancedDatasetSampler(train_set))
        val_set = [train_data[i] for i in val_index]
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        print('============================Fold [{}]======================================='.format(fold_num))
        leave_out_protein = [y for x in LABELS_SET for y in x][leave_out_index]
        best_val_loss, best_val_acc = run(train_loader, val_loader, test_loader,num_epochs, model, criterion, optimizer, device,
                                          fold_num, model_path=model_path,superfamily=superfamily,leave_out_protein=leave_out_protein)
        val_loss.append(best_val_loss)
        val_acc.append(best_val_acc)
        fold_num += 1

    val_loss = np.asarray(val_loss)
    val_acc = np.asarray(val_acc)
    print(val_loss, val_acc)
    print("average loss of validation: ", np.mean(val_loss), "average acc of validation: ", np.mean(val_acc))
    print("val_loss std:", statistics.stdev(val_loss), " val_acc std:", statistics.stdev(val_acc))

    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    # test_loss = []
    # test_acc = []

    print('============================Final Test=======================================')
    finaltest(superfamily, leave_out_index, LABELS_SET, model_path, num_classes, device, num_sample=args.num_sample,leave_out_protein =leave_out_protein,k=args.intersection)


# We use the pre-defined number of epochs to determine how many iterations to train the network on
def run(train_loader, val_loader, test_loader, num_epochs, model, criterion, optimizer, device, fold_num, model_path,superfamily,leave_out_protein):
    tr_losses = np.zeros((num_epochs,))
    tr_accs = np.zeros((num_epochs,))

    val_losses = np.zeros((num_epochs,))
    val_accs = np.zeros((num_epochs,))
    val_recalls = np.zeros((num_epochs,))
    val_f1s= np.zeros((num_epochs,))
    val_precisions = np.zeros((num_epochs,))

    test_losses = np.zeros((num_epochs,))
    test_accs = np.zeros((num_epochs,))
    test_recalls= np.zeros((num_epochs,))
    test_f1s= np.zeros((num_epochs,))
    test_precisions= np.zeros((num_epochs,))

    trainer = Trainer(train_loader, val_loader, model, criterion, optimizer, device)

    best_val_loss = 1e6
    best_val_acc = -1
    pbar = tqdm(range(0, num_epochs), ncols=70)
    for epoch in pbar:
        # i, acc, losses = train(Dataloader)
        i, acc, losses = trainer.train()
        tr_losses[epoch] = losses / i
        tr_accs[epoch] = acc / i
        print('Epoch [{}/{}] Train , acc: {:.4f}, loss:{:.4f}'.format(epoch + 1, num_epochs, tr_accs[epoch],
                                                                      tr_losses[epoch]))

        # v_i, v_acc, v_losses, y_trues, y_preds = val(test_loader)
        v_i, v_acc, v_recall, v_f1, v_precision,v_losses, y_trues, y_preds = trainer.val()
        cnf = confusion_matrix(y_trues, y_preds)
        val_losses[epoch] = v_losses / v_i
        val_accs[epoch] = v_acc / v_i
        val_recalls[epoch] = v_recall / v_i
        val_f1s[epoch] = v_f1 / v_i
        val_precisions[epoch] =  v_precision / v_i
        print('Epoch [{}/{}] Val , acc: {:.4f}, recall: {:.4f}, f1: {:.4f},precision: {:.4f}, loss:{:.4f}'.format(epoch + 1, num_epochs,
                                                                                                                  val_accs[epoch],val_recalls[epoch],val_f1s[epoch],val_precisions[epoch],
                                                                                                                  val_losses[epoch]))
        #test
        testtrainer = Trainer(test_loader, test_loader, model, criterion, optimizer, device)
        t_i, t_acc, t_recall, t_f1, t_precision,t_loss, ty_trues, ty_preds = testtrainer.val()
        tcnf = confusion_matrix(ty_trues, ty_preds)
        test_losses[epoch] = t_loss / t_i
        test_accs[epoch] = t_acc / t_i
        test_recalls[epoch] = t_recall / t_i
        test_f1s[epoch] = t_f1 / t_i
        test_precisions[epoch] =  t_precision / t_i
        print('Epoch [{}/{}] Test , acc: {:.4f}, recall: {:.4f}, f1: {:.4f},precision: {:.4f}, loss:{:.4f}'.format(epoch + 1, num_epochs,
                                                                                                                   test_accs[epoch],test_recalls[epoch], test_f1s[epoch], test_precisions[epoch],
                                                                                                                   test_losses[epoch]))


        current_val_loss = v_losses / v_i
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_val_acc = val_accs[epoch]
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model_fold{}_{}_{}.ckpt'.format(fold_num,superfamily,leave_out_protein)))

    return best_val_loss, best_val_acc


def finaltest(superfamily, leave_out_index, LABELS_SET, model_path, num_classes, device,num_sample,leave_out_protein,k):
    criterion = nn.CrossEntropyLoss()
    test_data = []
    if superfamily == "Enolase":
        test_data = DeepVaspS("./DataSet/EnolaseNew/", LABELS_SET,
                              leave_out_index,k, Train=False,num_sample=num_sample)
        model = ConvNeuralNet(num_classes, superfamily='Enolase')
    elif superfamily == "Serprot":
        test_data = DeepVaspS("./DataSet/SerprotNew/", LABELS_SET,
                              leave_out_index,k, Train=False, num_sample=num_sample)
        model = ConvNeuralNet(num_classes, superfamily='Serprot')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01, momentum=0.9)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    avg_loss = np.empty(0)
    avg_acc = np.empty(0)
    avg_f1 = np.empty(0)
    avg_recall = np.empty(0)
    avg_precision = np.empty(0)

    for n in range(1, 6):
        # model_dict = torch.load(os.path.join(model_path, 'best_model_fold_{}.ckpt'.format(n)))
        model_dict = torch.load(os.path.join(model_path,'best_model_fold{}_{}_{}.ckpt'.format(n, superfamily, leave_out_protein)))
        model.load_state_dict(model_dict)
        model = model.to(device)
        model.eval()
        trainer = Trainer(test_loader, test_loader, model, criterion, optimizer, device)
        # v_i, v_acc, v_losses, y_trues, y_preds = trainer.val()
        v_i, v_acc, v_recall, v_f1, v_precision, v_losses, y_trues, y_preds = trainer.val()
        cnf = confusion_matrix(y_trues, y_preds)
        # print(v_losses,v_acc)
        test_loss = v_losses / v_i
        test_acc = v_acc / v_i
        test_recall = v_recall / v_i
        test_f1 =  v_f1  / v_i
        test_precision = v_precision / v_i
        avg_loss = np.append(avg_loss, test_loss)
        avg_acc = np.append(avg_acc, test_acc)
        avg_recall = np.append(avg_recall, test_recall)
        avg_f1 = np.append(avg_f1, test_f1)
        avg_precision = np.append(avg_precision,test_precision)
        print('Test model {}... acc: {:.4f}, recall: {:.4f}, f1: {:.4f}, precision: {:.4f}, loss:{:.4f}'.format(n,test_acc,test_recall,test_f1,test_precision,test_loss))
    print('Test Average... acc: {:.4f}, recall: {:.4f}, f1: {:.4f}, precision: {:.4f}, loss:{:.4f}'.format(np.mean(avg_acc),np.mean(avg_recall),np.mean(avg_f1),np.mean(avg_precision), np.mean(avg_loss)))


def pca_enolase(ENOLASE_LABELS_SET,num_sample):
    # data_dir, labels_set = "./DataSet/EnolaseNew/", ENOLASE_LABELS_SET
    data_dir, labels_set = "./DataSet/EnolaseNew/", [ENOLASE_LABELS_SET[0]]
    print(labels_set)
    data, labels = DeepVaspS.sample_numpy_load(data_dir, labels_set,num_sample)
    data = data.reshape((data.shape[0], -1))
    pca = PCA(n_components=2)
    X_r = pca.fit(data).transform(data)


    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure()
    # colors = ["navy", "turquoise", "darkorange",'']
    lw = 2
    # target_names = ["2OX4", "1MDR", "1mdl", "1mns", "2OG9",
    #                 "2PGW", "1BKH", "3DGB", "3FJ4", "3I4K", "2ZAD", "3DEQ", "3DER", "1JPM", "1TKK",
    #                 "1E9I", "1PDY", "1IYX", "1EBH", "1TE6", "1W6T", "2XSX", "2PA6", "3OTR", "3QN3"]
    target_names = [y for x in ENOLASE_LABELS_SET for y in x]
    for i, target_name in zip(range(4), target_names[0:4]):
        plt.scatter(
            X_r[i * 1000:(i + 1) * 1000, 0], X_r[i * 1000:(i + 1) * 1000, 1], alpha=0.8, lw=lw, label=target_name
        )

    # for i, target_name in zip(range(5), target_names[0:5]):
    #     plt.scatter(
    #         X_r[i * 1000:(i + 1) * 1000, 0], X_r[i * 1000:(i + 1) * 1000, 1], c="red",alpha=0.8, lw=lw, label=target_name
    #     )
    #
    # for i, target_name in zip(range(5,15), target_names[5:15]):
    #     plt.scatter(
    #         X_r[i*1000:(i+1)*1000, 0], X_r[i*1000:(i+1)*1000, 1],c="yellow",alpha=0.8, lw=lw, label=target_name
    #     )
    #
    # for i, target_name in zip(range(15,25), target_names[15:25]):
    #     plt.scatter(
    #         X_r[i*1000:(i+1)*1000, 0], X_r[i*1000:(i+1)*1000, 1],c="blue",alpha=0.8, lw=lw, label=target_name
    #     )

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA_Enolase")
    plt.show()


def pca_serineProt(SerProt_LABELS_SET,num_sample):
    SerProtTrypsin = ["2f91", "1fn8", "2eek", "1h4w", "1bzx", "1aq7", "1ane", "1aks", "1trn", "1a0j", "1eq9", "4cha",
                      "1kdq", "8gch", "1elt", "1b0e"]
    data_dir, labels_set = "./DataSet/SerprotNew/", SerProt_LABELS_SET
    data, labels = DeepVaspS.sample_numpy_load(data_dir, labels_set,num_sample)
    data = data.reshape((data.shape[0], -1))
    pca = PCA(n_components=2)
    X_r = pca.fit(data).transform(data)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure()
    # colors = ["navy", "turquoise", "darkorange"]
    lw = 2
    plt.figure()
    # colors = ["navy", "turquoise", "darkorange",'']
    target_names = ["2f91", "1fn8", "2eek", "1h4w", "1bzx", "1aq7", "1ane", "1aks", "1trn", "1a0j",
                    "1eq9", "4cha", "1kdq", "8gch",
                    "1elt", "1b0e"]

    for i, target_name in zip(range(10), target_names[0:10]):
        plt.scatter(
            X_r[i * 1000:(i + 1) * 1000, 0], X_r[i * 1000:(i + 1) * 1000, 1], c="red", alpha=0.8, lw=lw,
            label=target_name
        )
    for i, target_name in zip(range(10, 14), target_names[10:14]):
        plt.scatter(
            X_r[i * 1000:(i + 1) * 1000, 0], X_r[i * 1000:(i + 1) * 1000, 1], c="blue", alpha=0.8, lw=lw,
            label=target_name
        )

    for i, target_name in zip(range(14, 16), target_names[14:16]):
        plt.scatter(
            X_r[i * 1000:(i + 1) * 1000, 0], X_r[i * 1000:(i + 1) * 1000, 1], c="yellow", alpha=0.8, lw=lw,
            label=target_name
        )

    plt.legend(loc="upper right", shadow=False, scatterpoints=1)
    plt.title("PCA_Serine Protease")
    plt.show()

def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # set random seed for numpy
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs

if __name__ == '__main__':

    seed = 12306
    seed_everything(seed)
    random.seed(seed)
    torch.manual_seed(seed)
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

    ENOLASE_LABELS_SET = [EnolaseMandelate, EnolaseMuconate, Enolase]
    # ENOLASE_LABELS_SET = [nonredunEnolaseMandelate, nonredunEnolaseMuconate, nonredunEnolase]

    # draw PCA plot
    # pca_enolase(ENOLASE_LABELS_SET,num_sample=1000)

    SerProtTrypsin = ["2f91", "1fn8", "2eek", "1h4w", "1bzx", "1aq7", "1ane", "1aks", "1trn", "1a0j"]
    SerProtChymotrypsin = ["1eq9", "4cha", "1kdq", "8gch"]
    SerProtElastase = ["1elt", "1b0e"]
    SERPROT_LABELS_SET = [SerProtTrypsin, SerProtChymotrypsin, SerProtElastase]
    # pca_serineProt(SerProt_LABELS_SET)

    args = argparse.ArgumentParser(description='DeepVASP-S parameter')
    args.add_argument('-s', '--superfamily', default='Enolase', type=str,
                      help='config training superfamily dataset; Enolase or Serprot ')
    args.add_argument('-p', '--path', default='./saved_model/Enolase/', type=str,
                      help='config output model path ')
    args.add_argument('-t', '--test', default=1, type=int,
                      help='set test protein index')
    args.add_argument('-e', '--epoch', default=5, type=int,
                      help='set number of epochs')
    args.add_argument('-cuda', '--cuda', default=0, type=int,
                      help='cuda')
    args.add_argument('-n', '--num_sample', default=500, type=int,
                      help='number of samples')
    args.add_argument('-k', '--intersection', default=15, type=int,
                      help='number of snapshorts used to generate intersection')


    args = args.parse_args()
    # print(args.superfamily,args.path,args.test)
    print(args)
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    superfamily = args.superfamily
    # LABELS_SET = ENOLASE_LABELS_SET
    if args.superfamily == "Enolase":
        LABELS_SET = ENOLASE_LABELS_SET
    elif args.superfamily == "Serprot":
        LABELS_SET = SERPROT_LABELS_SET
    proteinset=[y for x in LABELS_SET for y in x][args.test]
    print(proteinset)
    main(superfamily, LABELS_SET, leave_out_index=args.test, model_path=args.path, num_epochs=args.epoch, seed=seed,args=args)
    # device = torch.device('cuda:%d'%(args.cuda) if torch.cuda.is_available() else 'cpu')
    # leave_out_index = args.test
    # leave_out_protein = [y for x in ENOLASE_LABELS_SET for y in x][leave_out_index]
    # finaltest(superfamily, leave_out_index, LABELS_SET, model_path=args.path, num_classes=3, device=device, num_sample=args.num_sample,leave_out_protein =leave_out_protein)

    print("finish")
