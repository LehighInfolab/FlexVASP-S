import torch
import numpy as np
import re
import argparse


def gen_saliency_surf(args,protein):
    # filename="D:/PycharmProjects/FrequencyHistogramEnoSerp/EnolaseNew/Enolase-029551.SURF"
    # saliency = "D:/PycharmProjects/FrequencyHistogramEnoSerp/EnolaseNew/EnolaseSaliency.txt"
    #/share/ceph/hawk/byc210_proj/yal421/DeepVASP/Gradcampp/Serprot/1aks/1aks_gradcam_499_100.txt
    #./share/ceph/hawk/byc210_proj/yal421/DeepVASP/  Gradcampp/Serprot/
    filename = args.path+"Gradcampp/"+args.superfamily+"/"+protein+"/"+"{}_gradcam_{}_{}.txt".format(protein,args.idx_sample,args.toph)
    with open(filename, 'r') as f:
        indexes = f.readlines()

    Geometries=[]
    Topologies=[]
    i=0
    for indx in indexes:
        #/share/ceph/hawk/byc210_proj/yal421/DeepVASP/
        with open(args.path+args.superfamily + "Cubes/" + args.superfamily + "-{}.SURF".format("{:06}".format(int(indx.split('\n')[0]))), 'r') as file:
            lines = file.readlines()
            for m in range(15,23):
                Geometries.append(lines[m])
            for n in range(31,43):
                topology=lines[n].split('\n')[0].split()
                top1 = int(topology[0]) + i * 8
                top2 = int(topology[1]) + i * 8
                top3 = int(topology[2]) + i * 8
                topology = str(top1)+' '+str(top2)+' '+str(top3)+'\n'
                Topologies.append(topology)
        i=i+1
    geobegin=lines[0:14]
    Geonum=["GEOMETRY:"+str(int(lines[14].split(":")[1].split("\n")[0])*i)+"\n"]
    topbegin = lines[23:30]
    Topnum=["TOPOLOGY:"+str(int(lines[30].split(":")[1].split("\n")[0])*i)+"\n"]
    topend = lines[44:49]




    # with open("SaliencyEnolase.surf",'w') as surf:
    with open(args.path+"Gradcampp/"+args.superfamily+"/"+protein+"/"+"{}_saliency_{}_{}.surf".format(protein,args.idx_sample,args.toph),"w") as surf:
        for i in range(len(Newline)):
            surf.write(Newline[i])

    print("{}_{}_{} saliency may Done".format(protein,args.toph,args.idx_sample))


if __name__ == '__main__':

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

    args = argparse.ArgumentParser(description='DeepVASP-S parameter')
    args.add_argument('-s', '--superfamily', default='Serprot', type=str,
                      help='config training superfamily dataset; Enolase or Serprot ')
    args.add_argument('-p', '--path', default='./share/ceph/hawk/byc210_proj/yal421/DeepVASP/', type=str,
                      help='config output model path ')
    args.add_argument('-t', '--test', default=0, type=int,
                      help='set test protein index')
    args.add_argument('-n', '--idx_sample', default=500, type=int,
                      help='number of samples')
    args.add_argument('-th', '--toph', default=600, type=int,
                      help='top h values in saliency map')

    args = args.parse_args()

    if args.superfamily == "Enolase":
        LABELS_SET = ENOLASE_LABELS_SET
    elif args.superfamily == "Serprot":
        LABELS_SET = SERPROT_LABELS_SET

    protein = [y for x in LABELS_SET for y in x][args.test]
