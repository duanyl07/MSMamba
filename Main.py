import datetime
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from thop import profile
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import LDA_SLIC
import MSMamba
import dgl
import numpy.ma as ma
import os
from spectral import spy_colors
from utils.visualization import visualize_segmentation
from utils.segmentation import get_false_color
os.chdir("D:\code\A_papercode\MSMamba")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# FLAG =1, indian
# FLAG =2, paviaU
# FLAG =3, salinas
samples_type = ['ratio', 'same_num'][1]
#参数在这里 
KNN=True
K_neigs=15
parts=20
beta=1e-3
# method=['MSMamba','GM-CNN noCE','GM','GM noCE','CNN','GM1','CEGMamba_M'] #消融
dim=128
flag={}
for (curr_train_ratio,FLAG, superpixel_scale,compactness1,compactness2,method,c1,c2) in [
            (10,1,500,0.03,0.1,'MSMamba',5,7),     #Hanchuan
            (10,2,100,0.03,0.1,'MSMamba',5,7),    #Xuzhou
            (10,3,500,0.005,0.1,'MSMamba',5,7),      #subHouston
            (10,4,400,0.005,0.03,'MSMamba',5,7)]:   #LongKou
            flag['CEGMamba']=True
            flag['CNN']=False
            flag['CE']=True
            flag['stage1']=True
            flag['GM1']=True
            flag['GM']=True
            flag['GM2']=True
            if method=='CNN':
                flag['GM']=False
                flag['CNN']=True
            if method=='MSMamba':
                flag['GM1']=True
                flag['GM']=True
                flag['GM2']=True
                flag['CNN']=True
            if method=='GM':
                flag['GM1']=True
                flag['GM']=True
                flag['GM2']=True
            torch.cuda.empty_cache()
            OA_ALL = []
            AA_ALL = []
            KPP_ALL = []
            AVG_ALL = []
            Train_Time_ALL=[]
            Test_Time_ALL=[]
            flops_all=[]
            params_all=[]
            learning_rate = 5e-4  # 学习率
            max_epoch = 300  # 迭代次数
            # dim=128
            # curr_train_ratio= 30  #0.1
            curr_val_ratio= 5   #0.01
            Seed_List=[0,1,2,3,4]#[0]#随机种子点
            if FLAG == 1:
                data_mat = sio.loadmat('.\\data\\HanChuan\\WHU_Hi_HanChuan.mat')
                data = data_mat['WHU_Hi_HanChuan']
                gt_mat = sio.loadmat('.\\data\\HanChuan\\WHU_Hi_HanChuan_gt.mat')
                gt = gt_mat['WHU_Hi_HanChuan_gt']
                # 参数预设
                train_ratio = curr_train_ratio  # 训练集比例。注意，训练集为按照‘每类’随机选取
                # val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
                class_count = 16  # 样本类别数
                learning_rate = 5e-4  # 学习率
                # max_epoch = 600  # 迭代次数
                dataset_name = "WHU_Hi_HanChuan"  # 数据集名称
                # dim=128
                pass
            if FLAG == 2:
                data_mat = sio.loadmat('.\data\Xuzhou\Xuzhou.mat')
                data = data_mat['all_x']
                # gt_mat = sio.loadmat('D:\code\data\Xuzhou\Xuzhou.mat')
                gt = data_mat['all_y']
                data=data.reshape([260,500,436])
                data=np.transpose(data,(1,0,2))
                gt=gt.reshape([260,500]).T
                
                # 参数预设
                train_ratio = curr_train_ratio  # 训练集比例。注意，训练集为按照‘每类’随机选取
                # val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
                class_count = 9  # 样本类别数
                # learning_rate = 5e-4  # 学习率
                # max_epoch = 600  # 迭代次数
                dataset_name = "Xuzhou"  # 数据集名称
                # dim=128
                # compactness=0.01
                # superpixel_scale = 200
                for i in range(class_count+1):
                    count=np.where(gt==i)
                    print(count[0].shape[0])
                pass
            if FLAG == 3:
                file_path = '.\data\HoustonU\crop_601_-601\HoustonU_601_-601.mat'
                file_path_gt = '.\data\HoustonU\crop_601_-601\HoustonU_gt_601_-601.mat'
                # file_path = 'D:\code\data\data\HoustonU\HoustonU.mat'
                # file_path_gt = 'D:\code\data\data\HoustonU\HoustonU_gt.mat'
                with h5py.File(file_path, 'r') as mat_file:
                    data = mat_file['HoustonU'][:]
                    # data = data.T
                with h5py.File(file_path_gt, 'r') as mat_file1:
                    gt = mat_file1['HoustonU_gt'][:]
                    # gt = gt.T
                # 参数预设
                train_ratio = curr_train_ratio  # 训练集比例。注意，训练集为按照‘每类’随机选取
                class_count = 16 # 20   # 样本类别数
                dataset_name = "subHouston"  # 数据集名称
                pass
            if FLAG == 4:
                data_mat = sio.loadmat('.\data\longkou\WHU_Hi_LongKou.mat')
                data = data_mat['WHU_Hi_LongKou']
                gt_mat = sio.loadmat('.\data\longkou\WHU_Hi_LongKou_gt.mat')
                gt = gt_mat['WHU_Hi_LongKou_gt']
                
                # 参数预设
                train_ratio = curr_train_ratio  # 训练集比例。注意，训练集为按照‘每类’随机选取
                # val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
                class_count = 9  # 样本类别数
                # learning_rate = 5e-4  # 学习率
                # max_epoch = 600  # 迭代次数
                dataset_name = "LongKou"  # 数据集名称
                # superpixel_scale = 100
                # dim=128
                pass
            
            
            if samples_type == 'same_num':
                train_samples_per_class = curr_train_ratio
                val_samples = curr_val_ratio
                train_ratio=curr_train_ratio
                val_ratio = curr_val_ratio
            elif samples_type == 'ratio':
                train_samples_per_class = curr_train_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
                val_samples = class_count
                train_ratio = curr_train_ratio
            
            cmap = cm.get_cmap('jet', class_count + 1)
            plt.set_cmap(cmap)
            m, n, d = data.shape  # 高光谱数据的三个维度 2（610，340，103）

            # 数据standardization标准化,即提前全局BN
            orig_data=data
            height, width, bands = data.shape  # 原始高光谱数据的三个维度
            data = np.reshape(data, [height * width, bands])  ##flatten
            minMax = preprocessing.StandardScaler()
            data = minMax.fit_transform(data)     ##标准化
            data = np.reshape(data, [height, width, bands])
            savepath='results\\' + dataset_name+" "+str(superpixel_scale)+" "+str(compactness1) +" "+str(compactness2) 
            savepath+='\\'+method+'\\'+current_time+' '+str(curr_train_ratio)+" "+str(curr_val_ratio)+' CNN'+str(c1)+'-'+str(c2)#+'chu'+str(parts)+' part'#+'spa_n_'+str(num_spatial_n1)
            savepath+=" "+str(dim)
                    # 保存数据信息
            if not os.path.exists(savepath):
                os.makedirs(savepath)
        
            def Draw_Classification_Map(label, gt,name: str, scale: float = 4.0, dpi: int = 400):
                '''
                get classification map , then save to given path
                :param label: classification label, 2D
                :param name: saving path and file's name
                :param scale: scale of image. If equals to 1, then saving-size is just the label-size
                :param dpi: default is OK
                :return: null
                '''
                fig, ax = plt.subplots()
                numlabel = np.array(label)
                masked_label = ma.masked_equal(numlabel, 0)
                # v = ax.imshow(masked_label.astype(np.int16),  interpolation='nearest')
                v = spy.imshow(classes=masked_label.astype(np.int16), fignum=fig.number)
                ax.set_axis_off()
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
                foo_fig = plt.gcf()  # 'get current figure'
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
                pass
            
            def GT_To_One_Hot(gt, class_count):
                '''
                Convet Gt to one-hot labels
                :param gt:
                :param class_count:
                :return:
                '''
                GT_One_Hot = []  # 转化为one-hot形式的标签
                for i in range(gt.shape[0]):
                    for j in range(gt.shape[1]):
                        temp = np.zeros(class_count,dtype=np.float32)
                        if gt[i, j] != 0:
                            temp[int( gt[i, j]) - 1] = 1
                        GT_One_Hot.append(temp)
                GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
                return GT_One_Hot
            
            def  set_all_seeds(seed) -> None:
                """
                Set the seed for reproducibility. 
                """
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                dgl.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                os.environ['PYTHONHASHSEED'] = str(seed)

            def visualize_predict(gt,predict_label,save_predict_path,save_gt_path,only_vis_label=False):
                row, col = gt.shape[0], gt.shape[1]
                predict = np.reshape(predict_label,(row,col)) + 1
                if only_vis_label:
                    vis_predict = np.where(gt==0,gt,predict)
                else:
                    vis_predict = predict
                #如果 vis_predict 是 PyTorch 张量，转换为 NumPy 数组
                if isinstance(vis_predict, torch.Tensor):
                    vis_predict = vis_predict.cpu().numpy()  # 将张量转换为 NumPy 数组

                # 确保 vis_predict 是整数类型，以便保存为图像
                vis_predict = vis_predict.astype(np.uint8)
                spy.save_rgb(save_predict_path, vis_predict, colors=spy_colors)
                spy.save_rgb(save_gt_path, gt, colors=spy_colors)

            def vis_a_image(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=False):
                visualize_predict(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=only_vis_label)
                visualize_predict(gt_vis,pred_vis,save_single_predict_path.replace('.png','_mask.png'),save_single_gt_path,only_vis_label=True)

            for curr_seed in Seed_List:

                #固定seed
                set_all_seeds(curr_seed)

                # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
                # random.seed(curr_seed)
                gt_reshape = np.reshape(gt, [-1])
                train_rand_idx = []
                val_rand_idx = []
                if samples_type == 'ratio':
                    for i in range(class_count):
                        idx = np.where(gt_reshape == i + 1)[-1]
                        samplesCount = len(idx)
                        rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                        rand_idx = random.sample(rand_list,
                                                np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                        rand_real_idx_per_class = idx[rand_idx]
                        train_rand_idx.append(rand_real_idx_per_class)
                    train_rand_idx = np.array(train_rand_idx,dtype=object) #改
                    train_data_index = []
                    for c in range(train_rand_idx.shape[0]):
                        a = train_rand_idx[c]
                        for j in range(a.shape[0]):
                            train_data_index.append(a[j])
                    train_data_index = np.array(train_data_index)
                    
                    ##将测试集（所有样本，包括训练样本）也转化为特定形式
                    train_data_index = set(train_data_index)
                    all_data_index = [i for i in range(len(gt_reshape))]
                    all_data_index = set(all_data_index)
                    
                    # 背景像元的标签
                    background_idx = np.where(gt_reshape == 0)[-1]
                    background_idx = set(background_idx)
                    test_data_index = all_data_index - train_data_index - background_idx
                    
                    # 从测试集中随机选取部分样本作为验证集
                    val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
                    val_data_index = random.sample(test_data_index, val_data_count)
                    val_data_index = set(val_data_index)
                    test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
                    
                    # 将训练集 验证集 测试集 整理
                    test_data_index = list(test_data_index)
                    train_data_index = list(train_data_index)
                    val_data_index = list(val_data_index)
                
                if samples_type == 'same_num':
                    # for i in range(class_count):
                    #     idx = np.where(gt_reshape == i + 1)[-1]
                    #     samplesCount = len(idx)
                    #     real_train_samples_per_class = train_samples_per_class
                    #     rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                    #     if real_train_samples_per_class > samplesCount:
                    #         real_train_samples_per_class = samplesCount
                    #     rand_idx = random.sample(rand_list,real_train_samples_per_class+val_samples)  # 随机数数量 四舍五入(改为上取整)
                    #     rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                    #     train_rand_idx.append(rand_real_idx_per_class_train)
                    #     rand_real_idx_per_class_val = idx[rand_idx[real_train_samples_per_class:real_train_samples_per_class+val_samples]]
                    #     val_rand_idx.append(rand_real_idx_per_class_val)
                    for i in range(class_count):
                        idx = np.where(gt_reshape == i + 1)[-1]
                        samplesCount = len(idx)
                        current_train_samples_per_class = train_samples_per_class
                        # 创建一个新的变量来存储当前循环的训练集采样数
                        if dataset_name == "indian_" and train_samples_per_class>10:
                            if i == 6 or i == 8:
                                current_train_samples_per_class = 5
                        
                        rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                        if current_train_samples_per_class > samplesCount:
                            current_train_samples_per_class = samplesCount-val_samples
                        
                        rand_idx = random.sample(rand_list, current_train_samples_per_class + val_samples)  # 随机数数量 四舍五入(改为上取整)
                        rand_real_idx_per_class_train = idx[rand_idx[0:current_train_samples_per_class]]
                        train_rand_idx.append(rand_real_idx_per_class_train)
                        
                        rand_real_idx_per_class_val = idx[rand_idx[current_train_samples_per_class:current_train_samples_per_class + val_samples]]
                        val_rand_idx.append(rand_real_idx_per_class_val)

                    # train_rand_idx = np.array(train_rand_idx)
                    train_data_index = []
                    # for c in range(train_rand_idx.shape[0]):
                    for c in range(len(train_rand_idx)):
                        a = train_rand_idx[c]
                        for j in range(a.shape[0]):
                            train_data_index.append(a[j])
                    train_data_index = np.array(train_data_index)

                    val_rand_idx = np.array(val_rand_idx)
                    val_data_index = []
                    for c in range(val_rand_idx.shape[0]):
                        a = val_rand_idx[c]
                        for j in range(a.shape[0]):
                            val_data_index.append(a[j])
                    val_data_index = np.array(val_data_index)
                    
                    train_data_index = set(train_data_index)
                    all_data_index = [i for i in range(len(gt_reshape))]
                    all_data_index = set(all_data_index)
                    
                    # 背景像元的标签
                    background_idx = np.where(gt_reshape == 0)[-1]
                    background_idx = set(background_idx)
                    
                    val_data_index = set(val_data_index)
                    test_data_index = all_data_index - train_data_index - background_idx - val_data_index

                    # 将训练集 验证集 测试集 整理
                    test_data_index = list(test_data_index)
                    train_data_index = list(train_data_index)
                    val_data_index = list(val_data_index)
                
                # 获取训练样本的标签图
                train_samples_gt = np.zeros(gt_reshape.shape)
                for i in range(len(train_data_index)):
                    train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
                    pass
                
                # 获取测试样本的标签图
                test_samples_gt = np.zeros(gt_reshape.shape)
                for i in range(len(test_data_index)):
                    test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
                    pass
                
                Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图
                
                # 获取验证集样本的标签图
                val_samples_gt = np.zeros(gt_reshape.shape)
                for i in range(len(val_data_index)):
                    val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
                    pass

                train_samples_gt=np.reshape(train_samples_gt,[height,width])
                test_samples_gt=np.reshape(test_samples_gt,[height,width])
                val_samples_gt=np.reshape(val_samples_gt,[height,width])

                train_samples_gt_onehot=GT_To_One_Hot(train_samples_gt,class_count)
                test_samples_gt_onehot=GT_To_One_Hot(test_samples_gt,class_count)
                val_samples_gt_onehot=GT_To_One_Hot(val_samples_gt,class_count)

                train_samples_gt_onehot=np.reshape(train_samples_gt_onehot,[-1,class_count]).astype(int)
                test_samples_gt_onehot=np.reshape(test_samples_gt_onehot,[-1,class_count]).astype(int)
                val_samples_gt_onehot=np.reshape(val_samples_gt_onehot,[-1,class_count]).astype(int)

                ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
                # 训练集
                train_label_mask = np.zeros([m * n, class_count])
                temp_ones = np.ones([class_count])
                train_samples_gt = np.reshape(train_samples_gt, [m * n])
                for i in range(m * n):
                    if train_samples_gt[i] != 0:
                        train_label_mask[i] = temp_ones
                train_label_mask = np.reshape(train_label_mask, [m* n, class_count])

                # 测试集
                test_label_mask = np.zeros([m * n, class_count])
                temp_ones = np.ones([class_count])
                test_samples_gt = np.reshape(test_samples_gt, [m * n])
                for i in range(m * n):
                    if test_samples_gt[i] != 0:
                        test_label_mask[i] = temp_ones
                test_label_mask = np.reshape(test_label_mask, [m* n, class_count])

                # 验证集
                val_label_mask = np.zeros([m * n, class_count])
                temp_ones = np.ones([class_count])
                val_samples_gt = np.reshape(val_samples_gt, [m * n])
                for i in range(m * n):
                    if val_samples_gt[i] != 0:
                        val_label_mask[i] = temp_ones
                val_label_mask = np.reshape(val_label_mask, [m* n, class_count])
                def merge_superpixels(mapping1, mapping2):
                    mapping1=torch.from_numpy(mapping1)#.to(device)
                    mapping2=torch.from_numpy(mapping2)#.to(device)
                    matrix_tensor = torch.cat([mapping1,mapping2],dim=-1)
                    return matrix_tensor.to(device)
                ls = LDA_SLIC.LDA_SLIC(data, np.reshape( train_samples_gt,[height,width]), class_count-1)
                tic0=time.time()
                Q, _,_,Seg1= ls.simple_superpixel(scale=superpixel_scale,compactness=compactness1) ##超像素S以及相关系数矩阵Q，根据 segments 判定邻接矩阵A
                Q1,  _,_,Seg2= ls.simple_superpixel(scale=superpixel_scale*1.5,compactness=compactness1) #
                Q2,  _,_,Seg3= ls.simple_superpixel(scale=superpixel_scale,compactness=compactness2) ##超像素S以及相关系数矩阵Q，根据 segments 判定邻接矩阵A
                Q3,  _,_,Seg4= ls.simple_superpixel(scale=superpixel_scale*1.5,compactness=compactness2) #
                toc0 = time.time()      ##Q[207400,2074][HW，超像素个数] S[2074,8] A[2074,2074] Seg[610,340]
                LDA_SLIC_Time=toc0-tic0
                # np.save(dataset_name+'Seg',Seg)
                print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
                false_color = get_false_color(data)
                visualize_segmentation(Seg1, false_color, gt, os.path.join(savepath, "visualize_segmentation1.png"))
                visualize_segmentation(Seg2, false_color, gt, os.path.join(savepath, "visualize_segmentation2.png"))
                visualize_segmentation(Seg3, false_color, gt, os.path.join(savepath, "visualize_segmentation3.png"))
                visualize_segmentation(Seg4, false_color, gt, os.path.join(savepath, "visualize_segmentation4.png"))
                del ls,Seg1,Seg2,Seg3,Seg4
                if flag['stage1']==True: 
                    Q_new=merge_superpixels(Q,Q1) #torch.from_numpy(Q1).to(device)
                    del Q,Q1
                    Q=Q_new
                    Q1_new=merge_superpixels(Q2,Q3)#torch.from_numpy(Q3).to(device) 
                    del Q2,Q3
                    Q1=Q1_new
                else:
                    Q_new=torch.from_numpy(Q).to(device)
                    del Q,Q1
                    Q=Q_new
                    Q1_new=torch.from_numpy(Q2).to(device) 
                    del Q2,Q3
                    Q1=Q1_new
                #compactness1 的序列
                nlist=np.arange(Q.shape[1])
                reversed_sort=np.argsort(nlist)
                nlist_r=np.flip(nlist).copy()
                reversed_sort_r=np.argsort(nlist_r)
                nlist1=[nlist,reversed_sort]
                nlist2=[nlist_r,reversed_sort_r]
                nlist=[nlist1,nlist2]
                #compactness2 的序列
                nlist_1=np.arange(Q1.shape[1])
                reversed_sort_1=np.argsort(nlist_1)
                nlist_r_1=np.flip(nlist_1).copy()
                reversed_sort_r_1=np.argsort(nlist_r_1)
                nlist1_1=[nlist_1,reversed_sort_1]
                nlist2_1=[nlist_r_1,reversed_sort_r_1]
                nlist_1=[nlist1_1,nlist2_1]

                nlist = [nlist,nlist_1]
                # Q1=[]
                Q=[Q,Q1]
                # A=torch.from_numpy(A).to(device)
                # A1=torch.from_numpy(A1).to(device)
                # A=[A,A1]

                #转到GPU
                train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
                test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
                val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
                #转到GPU
                train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
                test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
                val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
                #转到GPU
                train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
                test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
                val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)
                
                
                net_input=np.array( data,np.float32)
                net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)

                if dataset_name == "indian_":
                    net = MSMamba.MSMamba(c1,c2,height, width, bands, class_count, dim, flag, Q,  nlist,model='smoothed')
                else:
                    net = MSMamba.MSMamba(c1,c2,height, width, bands, class_count, dim, flag, Q, nlist)
                parameters=sum(p.numel() for p in net.parameters())
                
                print('参数量:'+str(parameters))
                print("parameters", net.parameters(), len(list(net.parameters())))
                net.to(device)
                # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                # print(f"Total Trainable Parameters: {total_params}")
                def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
                    real_labels = reallabel_onehot
                    we = -torch.mul(real_labels,torch.log(predict))
                    we = torch.mul(we, reallabel_mask)
                    pool_cross_entropy = torch.sum(we)
                    return pool_cross_entropy
                

                zeros = torch.zeros([m * n]).to(device).float()
                def evaluate_performance(network_output,train_samples_gt,train_samples_gt_onehot, require_AA_KPP=False,printFlag=True):
                    if False==require_AA_KPP:
                        with torch.no_grad():
                            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
                            available_label_count=available_label_idx.sum()#有效标签的个数
                            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                            OA= correct_prediction.cpu()/available_label_count
                            
                            return OA
                    else:
                        with torch.no_grad():
                            #计算OA
                            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
                            available_label_count=available_label_idx.sum()#有效标签的个数
                            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                            OA= correct_prediction.cpu()/available_label_count
                            OA=OA.cpu().numpy()
                            mask = train_samples_gt != 0
                            y_true = torch.argmax(train_samples_gt_onehot, dim=1)[mask].cpu().numpy()
                            y_pred = torch.argmax(network_output, dim=1)[mask].cpu().numpy()
                            OA1 = accuracy_score(y_true, y_pred)
                            print(OA)
                            print(OA1)
                            # 计算AA
                            zero_vector = np.zeros([class_count])
                            output_data=network_output.cpu().numpy()
                            train_samples_gt=train_samples_gt.cpu().numpy()
                            train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
                            
                            output_data = np.reshape(output_data, [m * n, class_count])
                            idx = np.argmax(output_data, axis=-1)
                            for z in range(output_data.shape[0]):
                                if ~(zero_vector == output_data[z]).all():
                                    idx[z] += 1
                            # idx = idx + train_samples_gt
                            count_perclass = np.zeros([class_count])
                            correct_perclass = np.zeros([class_count])
                            for x in range(len(train_samples_gt)):
                                if train_samples_gt[x] != 0:
                                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                                    if train_samples_gt[x] == idx[x]:
                                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
                            test_AC_list = correct_perclass / count_perclass
                            test_AA = np.average(test_AC_list)

                            # 计算KPP
                            test_pre_label_list = []
                            test_real_label_list = []
                            output_data = np.reshape(output_data, [m * n, class_count])
                            idx = np.argmax(output_data, axis=-1)
                            idx = np.reshape(idx, [m, n])
                            for ii in range(m):
                                for jj in range(n):
                                    if Test_GT[ii][jj] != 0:
                                        test_pre_label_list.append(idx[ii][jj] + 1)
                                        test_real_label_list.append(Test_GT[ii][jj])
                            test_pre_label_list = np.array(test_pre_label_list)
                            test_real_label_list = np.array(test_real_label_list)
                            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                            test_real_label_list.astype(np.int16))
                            test_kpp = kappa

                            # 输出
                            if printFlag:
                                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                                print('acc per class:')
                                print(test_AC_list)

                            OA_ALL.append(OA)
                            AA_ALL.append(test_AA)
                            KPP_ALL.append(test_kpp)
                            AVG_ALL.append(test_AC_list)
                            
                            # 保存数据信息
                            if not os.path.exists(savepath):
                                os.makedirs(savepath)
                            f = open( savepath+'\\' + dataset_name  + '_results.txt', 'a+')
                            str_results = '\n======================' \
                                        + " learning rate=" + str(learning_rate) \
                                        + " epochs=" + str(max_epoch) \
                                        + " train ratio=" + str(train_ratio) \
                                        + " val ratio=" + str(val_ratio) \
                                        + " ======================" \
                                        + "\nOA=" + str(OA) \
                                        + "\nAA=" + str(test_AA) \
                                        + '\nkpp=' + str(test_kpp) \
                                        + '\nacc per class:' + str(test_AC_list) + "\n"
                                        # + '\ntrain time:' + str(time_train_end - time_train_start) \
                                        # + '\ntest time:' + str(time_test_end - time_test_start) \
                            f.write(str_results)
                            f.close()

                            return OA
                
                # 训练
                optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)#,weight_decay=0.0001
                best_loss=99999

                net.train()
                tic1 = time.perf_counter()
                for i in range(max_epoch+1):
                    optimizer.zero_grad()  # zero the gradient buffers
                    output,_= net(net_input)
                    loss = compute_loss(output,train_samples_gt_onehot,train_label_mask)
                    loss.backward(retain_graph=False)
                    optimizer.step()  # Does the update
                    # net.toynet_ema.update(net.toynet.state_dict())
                    #
                    if i%10==0:
                        with torch.no_grad():
                            net.eval()
                            output,_= net(net_input)
                            # trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                            trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                            valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                            valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                            print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), loss, trainOA, valloss, valOA))

                            if valloss < best_loss :
                                best_loss = valloss

                                torch.save(net.state_dict(),"model\\best_model.pt")
                                print('save model...')

                        torch.cuda.empty_cache()
                        net.train()
                    # #
                toc1 = time.perf_counter()
                print("\n\n====================training done. starting evaluation...========================\n")
                training_time=toc1 - tic1 + LDA_SLIC_Time #分割耗时需要算进去
                Train_Time_ALL.append(training_time)
                
                torch.cuda.empty_cache()
                
                net.load_state_dict(torch.load("model\\best_model.pt"))
                net.eval()        
                # 计算 FLOPs 和参数量
                flops, params = profile(net, inputs=(net_input,))
                print(f"Parameters: {params / 10**6:.2f} MParams")  # Parameters in MParams
                print(f"FLOPs: {flops / 10**9:.2f} GFLOPs")  # FLOPs in GFLOPs
                flops_all.append(flops)
                params_all.append(params)

                with torch.no_grad():

                    tic2 = time.perf_counter()
                    output,output1 = net(net_input)
                    toc2 = time.perf_counter()
                    testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
                    testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot,require_AA_KPP=True,printFlag=False)
                    print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
                    #计算
                    classification_map=torch.argmax(output, 1).reshape([height,width]).cpu()+1
                    Draw_Classification_Map(classification_map,gt,savepath+'\\' + dataset_name+str(testOA))
                    testing_time=toc2 - tic2 + LDA_SLIC_Time #分割耗时需要算进去
                    Test_Time_ALL.append(testing_time)
                    predict=torch.argmax(output, 1)
                    predict=predict.cpu()
                    save_single_predict_path = os.path.join(savepath+'\\predict_{}.png'.format(str(curr_seed+1)))
                    save_single_gt_path = os.path.join(savepath+'\\gt.png')
                    vis_a_image(gt,predict,save_single_predict_path, save_single_gt_path)
                torch.cuda.empty_cache()
                del net
                
            OA_ALL = np.array(OA_ALL)*100
            AA_ALL = np.array(AA_ALL)*100
            KPP_ALL = np.array(KPP_ALL)*100
            AVG_ALL = np.array(AVG_ALL)*100
            Train_Time_ALL=np.array(Train_Time_ALL)
            Test_Time_ALL=np.array(Test_Time_ALL)

            print("\ntrain_ratio={}".format(curr_train_ratio),
                "\n==============================================================================")
            print('OA =', round(np.mean(OA_ALL), 2), '+-', round(np.std(OA_ALL), 2))
            print('AA =', round(np.mean(AA_ALL), 2), '+-', round(np.std(AA_ALL), 2))
            print('Kpp =', round(np.mean(KPP_ALL), 2), '+-', round(np.std(KPP_ALL), 2))
            print('AVG =', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
            print("Average training time:{}".format(np.mean(Train_Time_ALL)))
            print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
            
            # 保存数据信息
            f = open(savepath+'\\' + dataset_name+ '_results.txt', 'a+')
            str_results = '\n\n************************************************\n' \
            +"CE:{} ".format(flag['CE'])\
            +"CNN:{} ".format(flag['CNN'])\
            +"GM:{} ".format(flag['GM'])\
            +"GM1:{} ".format(flag['GM1'])\
            +"GM2:{} ".format(flag['GM2'])\
            +"\ntrain_ratio={}".format(curr_train_ratio) \
            + '\nOA=' + str(round(np.mean(OA_ALL), 2)) + ' +- ' + str(round(np.std(OA_ALL), 2)) \
            +'\nAA=' + str(round(np.mean(AA_ALL), 2)) + ' +- ' + str(round(np.std(AA_ALL), 2)) \
            +'\nKpp=' + str(round(np.mean(KPP_ALL), 2)) + ' +- ' + str(round(np.std(KPP_ALL), 2)) \
            +'\nAVG=' + str(np.mean(AVG_ALL, 0)) + ' +- ' + str(np.std(AVG_ALL, 0))\
            +"\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
            +"\nAverage testing time:{}".format(np.mean(Test_Time_ALL)) \
            + f"\nParameters: {np.mean(params_all) / 10**6:.2f} MParams\n" + f"FLOPs: {np.mean(flops_all) / 10**9:.2f} GFLOPs" \
            + f"\nParameters: {parameters} \n"
            f.write(str_results)
            f.close()