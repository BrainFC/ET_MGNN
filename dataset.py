import os
import pickle
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from random import shuffle, randrange, choices
from nilearn import image, maskers, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat
import re
def read_netts(file_path):
    with open(file_path, 'r') as file:

        netts_list=[]
        for line in file:
           
        # 处理每一行
           ts_list = str(line).split()
           ts_float = [float(value) for value in ts_list]
           netts_list.append(ts_float)
        netts_matrix = np.array(netts_list)
        return netts_matrix.T


def read_txt(file_path):

    with open(file_path, 'r') as file:
        netts_list = []
        for line in file:
            # 处理每一行
            ts_list = str(line).split()
            ts_float = [float(value) for value in ts_list]
            netts_list.append(ts_float)
        netts_matrix = np.array(netts_list)
        return netts_matrix



def read_abide_sc(file_path,bntype,percent):
    if bntype != 'FC':
        sc_dict = loadmat(file_path)
        sc_matrix = sc_dict['dataTable']
    else:
        print("read netcc")
        sc_tmp = np.loadtxt(file_path)
        sc_matrix = sc_tmp[2:2+246,:]
    # percentile_values = np.percentile(sc_matrix, 95, axis=1)
    percentile_values = np.percentile(sc_matrix, percent, axis=1)

    sc_result = np.where(sc_matrix >= percentile_values[:, np.newaxis], 1, 0)
    return sc_result


def read_adni_sc(file_path,bntype,percent):
    if bntype != 'FC':
        sc_dict = loadmat(file_path)
        sc_matrix = sc_dict['dataTable']
    else:# fc
        print("read netcc")
        sc_tmp = np.loadtxt(file_path)
        sc_matrix = sc_tmp[2:2+246,:]
    # percentile_values = np.percentile(sc_matrix, 95, axis=1)
    sc_matrix = np.real(sc_matrix)

    # 下面俩个代码随时屏蔽，区别在于是否对SC进行二值化
    # percentile_values = np.percentile(sc_matrix, percent, axis=1)
    # sc_matrix = np.where(sc_matrix >= percentile_values[:, np.newaxis], 1, 0)
    return sc_matrix


def get_adni_sbj():
    sbj_list = []
    # 打开并读取txt文件
    with open('/home/image015/BrainCode/data/Disease/ADNI/ADNI_sbj_NEW.txt', 'r') as file:
        # 按行读取文件内容
        lines = file.readlines()
    # 循环遍历每一行内容
    for line in lines:
        # 处理每一行内容，这里以打印为例
        # print(line.strip())  # 使用strip()方法去除行尾的换行符
        sbj_list.append(line.strip())
    return sbj_list


def is_in_abide_sc(dir_path, file_name,bntype):
    subject_id = file_name.split('_')[1]
    if bntype != 'FC':
       sc_file_name = subject_id + '.mat'
    else:
        sc_file_name = 'corr_' + subject_id + '_000.netcc'

    file_path = os.path.join(dir_path, sc_file_name)
    return os.path.isfile(file_path), file_path,subject_id

def is_in_adni_sc_new(dir_path, file_name,bntype):
    parts = file_name.split('.')
    subject_id = parts[0]
    sc_file_name = 'corr_' + subject_id + '_000.netcc'
    file_path = os.path.join(dir_path, sc_file_name)

    return os.path.isfile(file_path), file_path,subject_id

class DatasetABIDEII(torch.utils.data.Dataset):
    def __init__(self, sourcedir, dynamic_length=None, k_fold=None, dx='dx_group', percent = 85,percent_sc = 50, bntype = 'FC'):
        super().__init__()
        #830_old ABIDE,   901 ABIDE_REST_NEW
        # self.filename = 'abide2_901_sbj'+ str(percent) + '-' + bntype +percent_sc
        self.bntype = bntype # 必须放在使用之前
        self.filename = 'new_abide2_916_sbj' + str(percent) + '_percent_sc_'+ str(percent_sc) + '_'+ str(dynamic_length)

        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.dx = dx

        self.percent = percent

        if os.path.isfile(os.path.join(sourcedir, 'processed', f'{self.filename}.pth')):
            # print("=== run here ===os.path ")
            self.timeseries_list, self.sc_list, self.label_list,self.sbj_list = torch.load(os.path.join(sourcedir, 'processed',f'{self.filename}.pth'))
            print("self.label_list.count(0)",self.label_list.count(0))
            print("self.label_list.count(1)",self.label_list.count(1))
            # print("=== run here ===self.timeseries_list[0].shape[1]",self.timeseries_list[0].shape[1])
        else:
            # print("=== run here ===")
            self.timeseries_list = []
            self.label_list = []
            self.sc_list = []
            self.sbj_list = []
            sub_list = []
            shape_list = []
            netts_list = [f for f in os.listdir(os.path.join(sourcedir, 'Disease/ABIDE_II','ABIDE_REST_NEW')) if f.endswith('netts')]
            print(netts_list)
            netts_list.sort()
            ab_data = pd.read_excel(os.path.join(sourcedir, 'Disease/ABIDE_II', 'ABIDE_2_participants.xlsx'))
            for subject in tqdm(netts_list, ncols=60, desc=f'prep:{dx.lower()[:3]}'):
                dir_path = os.path.join(sourcedir, 'Disease/ABIDE_II', 'ABIDE_' + self.bntype)
                # exist, sc_path = is_in_sc(dir_path, subject) # original read sc
                exist, sc_path,subject_id = is_in_abide_sc(dir_path, subject,self.bntype)
                print("====",exist,subject_id, sc_path)

                if exist:
                    # print("=== subject ===",subject)
                    subject_path = os.path.join(sourcedir, 'Disease/ABIDE_II','ABIDE_REST_NEW', subject)
                    # subject_id = re.search(r'\d+', subject.split('_')[1]).group() # before had resolved the subjectID
                    label = ab_data[ab_data['participant_id'] == subject_id][self.dx].values[0].astype(int)
                    # label = ad_data[ad_data['SUB_ID'] == subject_id][self.dx].values[0].astype(int)
                    timeseries = read_netts(subject_path)
                    print("====subject id:", subject_id ,"exist, timeseries_len",timeseries.shape)
                    if timeseries.shape[1] != 246 or timeseries.shape[0] < 120:#存放不滿足條件的
                    # if timeseries.shape[1] == 246 and timeseries.shape[0] > 85:#存放不滿足條件的
                        sub_list.append(subject_id)
                        shape_list.append(timeseries.shape)
                        continue
                        # break
                        # sc_matrix = read_sc(sc_path) # read SC
                    sc_matrix = read_abide_sc(sc_path, bntype, self.percent)
                    self.timeseries_list.append(timeseries)
                    label -= 1
                    self.label_list.append(label)
                    self.sc_list.append(sc_matrix)
                    self.sbj_list.append(subject_id)

            print("self.label_list.count(0)",self.label_list.count(0))
            print("self.label_list.count(1)",self.label_list.count(1))

            torch.save((self.timeseries_list, self.sc_list, self.label_list,self.sbj_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
            # torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
        if type(k_fold) is type(None):
            k_fold = 0
        if k_fold >1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None
        self.folds = list(range(k_fold))
        # print("timeseries_list:",self.timeseries_list[0])
        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        sc_matrix = self.sc_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            if self.train:
                timeseries = timeseries[0:0+self.dynamic_length]

        label = self.label_list[self.fold_idx[idx]]
       
        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32),
                'sc_matrix': torch.tensor(sc_matrix, dtype=torch.float32), 'label': torch.tensor(label)}


class DatasetADNI(torch.utils.data.Dataset):
    def __init__(self, sourcedir, dynamic_length=None, k_fold=None, percent = 85,percent_sc=50,task='label', bntype = 'FC',cltype = 'AN'):
        super().__init__()
        # self.filename = 'adni-246-' + str(ds)
        self.bntype = bntype
        self.cltype = cltype
        self.percent_sc = percent_sc
        # 這個用MBR结果比较好 adni-246-903_test
        self.filename = ('adni-246-1021_test' + str(percent) + '_percent_sc_'+ str(percent_sc) + '_'
                         + self.bntype + '_' + self.cltype)

        # self.filename = 'adni-246-wei-sc_fc' + str(percent) + '_' + self.bntype + '_' + self.cltype
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task = task
        self.percent = percent

        self.sbj_scfc = get_adni_sbj()


        if os.path.isfile(os.path.join(self.sourcedir, 'processed', f'{self.filename}.pth')):
            print("=== load pth ===",os.path.join(self.sourcedir, 'processed', f'{self.filename}.pth'))
            self.timeseries_list, self.sc_list, self.label_list, self.sbj_list= torch.load(
                os.path.join(self.sourcedir, 'processed', f'{self.filename}.pth'))
            print("lable 0 count:",self.label_list.count(0))
            print("lable 1 count:",self.label_list.count(1))
            print("lable 2 count",self.label_list.count(2))
        else:
            self.timeseries_list = []
            self.label_list = []
            self.sc_list = []
            self.sbj_list = []
            sub_list = [] # 下面这行时间序列，固定不变
            # netts_list = [f for f in os.listdir(os.path.join(self.sourcedir, 'Disease/ADNI/ADNI_TS')) if f.endswith('netts')]

            netts_list = [
                f for f in os.listdir(os.path.join(self.sourcedir, 'Disease/ADNI/ADNI_TS_NEW')) #ADNI_TS (之前的)
                # if f.endswith('netts') and f[5:15] in self.sbj_scfc]
                if f.endswith('txt') and f[0:10] in self.sbj_scfc]

            netts_list.sort()

            # clean_netts_list(netts_list)

            ds_data = pd.read_csv(os.path.join(sourcedir, 'Disease/ADNI/ADNI_subj_info_Fmri.csv'))


            for subject in tqdm(netts_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):

                dir_path = os.path.join(sourcedir, 'Disease/ADNI', 'ADNI_'+self.bntype)

                exist, sc_path,subject_id = is_in_adni_sc_new(dir_path, subject,bntype)

                print("====",exist,subject_id, sc_path)
                if exist:
                    subject_path = os.path.join(sourcedir, 'Disease/ADNI/ADNI_TS_NEW', subject)
                    # subject_id = subject.split('_')[1]
                    if subject_id in ds_data['Subject'].values:
                        label = ds_data[ds_data['Subject'] == subject_id]['DX'].values[0].astype(int)
                        print("subject_id:", subject_id, "label:", label)
                        # timeseries = read_netts(subject_path)
                        timeseries = read_txt(subject_path)


                        if timeseries.shape[1] != 246:
                            sub_list.append(subject_id)
                            continue
                        # 修改一下，不对SC进行二值化
                        sc_matrix = read_adni_sc(sc_path,bntype,self.percent)

                        # 'AN','AM','NM' : AD(0) , MCI(1), NC(2)
                        if self.cltype == 'AN' and label != 1 : # 0 ,NC, 1, AD
                            label = 1 if label == 0 else 0
                            self.label_list.append(label)
                            self.timeseries_list.append(timeseries)
                            self.sc_list.append(sc_matrix)
                            self.sbj_list.append(subject_id)
                        elif self.cltype == 'AM' and label != 2: # 0 MCI, 1, AD
                            # label = 1 if label ==0 else 1
                            label = (label + 1) % 2
                            self.label_list.append(label)
                            self.timeseries_list.append(timeseries)
                            self.sc_list.append(sc_matrix)
                            self.sbj_list.append(subject_id)
                        elif self.cltype == 'NM' and label != 0: # 0,NC , 1, MCI
                            label = 0 if label == 2 else label
                            self.label_list.append(label)
                            self.timeseries_list.append(timeseries)
                            self.sc_list.append(sc_matrix)
                            self.sbj_list.append(subject_id)
                        elif self.cltype == 'AMN': # add all ADNI subjects
                            self.label_list.append(label)
                            self.timeseries_list.append(timeseries)
                            self.sc_list.append(sc_matrix)
                            self.sbj_list.append(subject_id)

                    else:
                        print("=====",subject_id,"not in adni xlsx")

            print("lable 0 count:",self.label_list.count(0))
            print("lable 1 count:",self.label_list.count(1))
            print("lable 2 count",self.label_list.count(2))

            print("subject with not 246 node:",sub_list)
            torch.save((self.timeseries_list, self.sc_list, self.label_list,self.sbj_list),
                       os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
        if type(k_fold) is type(None):
            k_fold = 0
        if k_fold > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None
        self.folds = list(range(k_fold))
        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)

    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        sc_matrix = self.sc_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (
                    np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            sampling_init = 0
            timeseries = timeseries[sampling_init:sampling_init + self.dynamic_length]

        label = self.label_list[self.fold_idx[idx]]

        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32),
                'sc_matrix': torch.tensor(sc_matrix, dtype=torch.float32), 'label': torch.tensor(label)}
