import os
import re
import time
import random
import numpy as np
from numpy.core.defchararray import index
from numpy.core.numeric import indices
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt

_paths = {
    'credit' : './raw_data/default_credit.csv'
}


def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep=',')
    return tp

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

class DataSetLoader(object):
    def __init__(self, name, pred_ratio = 0.5, test_ratio = 0.1):
        self._name = name
        self._train_ratio = 1 - pred_ratio
        self._test_ratio = test_ratio

        
        np.random.seed(0)
        self.all_data = load_data(_paths[self._name])
        # shuffled_idx = np.random.permutation(self.all_data.shape[0])

        # num_train = int(np.ceil(self.all_data.shape[0] * self._train_ratio))
        # self.all_train = self.all_data.iloc[shuffled_idx[:num_train]]
        # self.all_pred = self.all_data.iloc[shuffled_idx[num_train:]]

        # num_test = int(np.ceil(self.all_data.shape[0] * self._test_ratio))
        # self.train = self.all_data.iloc[shuffled_idx[num_test:num_train]]
        # self.test = self.all_data.iloc[shuffled_idx[:num_test]]
        # 取消 shuffled     
    

        num_train = int(np.ceil(self.all_data.shape[0] * self._train_ratio))
        self.all_train = self.all_data.iloc[:num_train]
        self.all_pred = self.all_data.iloc[num_train:]

        num_test = int(np.ceil(self.all_data.shape[0] * self._test_ratio))
        self.train = self.all_data.iloc[num_test:num_train]
        self.test = self.all_data.iloc[:num_test]

        print("All data : {}".format(self.all_data.shape[0]))
        print("\tAll train data : {}".format(self.all_train.shape[0]))
        print("\t\tTrain data : {}".format(self.train.shape[0]))
        print("\t\ttest data : {}".format(self.test.shape[0]))
        print("\tPred data : {}".format(self.all_pred.shape[0]))

        # print(self.all_data)
        # print(self.train)
        # print(self.test)

        #将数据分开 分为local 与 guest
        self.host_train_X = self.all_train.iloc[:,2:15]
        self.host_train_Y = self.all_train.iloc[:,1:2]

        self.guest_train_X = self.all_train.iloc[:,15:25]

        ## 后端数据预处理流程 data_selection view
        # 1. tsne 降维到2维
        # 2. 使用 elbow 算法 算出 需要多少个cluster
        # 3. 用kmeans聚类

        # 1. tsne n to 2 dim
        #     Input : local_feature   shape:  10000*12 
        #     Output: 2D_coordinates  shape:  10000*2
        print("step 1")
        # embedding_exist = False
        embedding_exist = True
        if embedding_exist:
            self.host_coordinate = pd.read_csv("./raw_data/credit_embeddings.csv", sep=',',index_col=0).values
        else:
            self.host_coordinate = self.Muti_tsne(self.host_train_X.values, True)
            df = pd.DataFrame({"center_x":self.host_coordinate[:,0], "certer_y":self.host_coordinate[:,1]},list(self.host_train_X.index))
            df.to_csv("./raw_data/credit_embeddings.csv")
        print(self.host_coordinate)

        # 2. Elbow method choose a K value automatically
        #     Input ：local_feature   shape:  10000*2
        #     Output：class_num       int  :  n
        print("step 2")
        # cluster_num_exist = False
        cluster_num_exist = True

        if cluster_num_exist:        
            self.cluster_num = 18
        else:
            self.cluster_num = self.ElbowMethod(self.host_train_X.values, 20, max_iter=500)

        print("elbow:", self.cluster_num) 

        # 3. 用kmeans聚类
        #     1. 输入：2D_coordinates `10000*2`  class_num `n`
        #     2. 输出：data `10000*1`  
        print("step 3")
        # Kmeans_exist = False
        Kmeans_exist = True
        if Kmeans_exist:
            self.data_class = pd.read_csv("./raw_data/data_class.csv", sep=',',index_col=0)["class"].values
        else:
            kmeans = KMeans(self.cluster_num, random_state=0).fit(self.host_coordinate)
            self.data_class = kmeans.labels_
            df = pd.DataFrame({"center_x":self.host_coordinate[:,0], "certer_y":self.host_coordinate[:,1], "class":self.data_class},index = list(self.host_train_X.index))
            df.to_csv("./raw_data/data_class.csv")            
    
        print(self.data_class)

        # 4. 生成中心坐标
        print("step 4")
        # center_exist = False
        center_exist = True
        if center_exist:
            self.data_center = pd.read_csv("./raw_data/data_center.csv", sep=',',index_col=0).values
        else:
            center_x = kmeans.cluster_centers_[:,0]
            center_y = kmeans.cluster_centers_[:,1]
            df = pd.DataFrame({"center_x":center_x, "certer_y":center_y})
            df.to_csv("./raw_data/data_center.csv")


        # 5.data merge
        print("step 5")
        data_class = pd.read_csv("./raw_data/data_class.csv", sep=',',index_col=0)
        # print(data_class.values)
        # print(self.host_train_X)
        # frames = [self.host_train_X, data_class]
        # df = pd.concat(frames,axis=0,ignore_index=True,join='inner')
        self.train_with_class = self.host_train_X.join(data_class)


        # 6.groupby class
        print("step 6")
        self.group = self.train_with_class.groupby("class")

        # 7.metric
        print("step 7")
        #   1.mean
        self.mean_list = self.train_with_class.groupby("class").agg('mean')
        # print(self.train_with_class.groupby("class").agg('mean'))

        #   2.std
        self.std_list = self.train_with_class.groupby("class").agg('std')
        # print(self.train_with_class.groupby("class").agg('std'))

        #   3.contant diversity
        from sklearn.metrics.pairwise import pairwise_distances
        # from sklearn.metrics.pairwise import cosine_similarity
        group_list = list(self.group)

        # print(group_list[0][1].iloc[:,0:13].values)
        print("7.3 contant diversity")
        sim_list = []
        for i in range(len(group_list)):
            dis = 0
            sim_ele = pairwise_distances(group_list[i][1].iloc[:,0:13].values, metric="cosine")
            num = sim_ele.shape[0]*(sim_ele.shape[0]-1)/2
            for i in range(sim_ele.shape[0]):
                for j in range(i+1, sim_ele.shape[0]):
                    dis = dis + sim_ele[i][j]
            sim_list.append(dis/num)
        self.sim_list = sim_list
        print(sim_list)
        #   4.statistic homogeneity



        #   5.distribution
        #   TODO:计算每一个group中不同维度的特征的数据分布
        #   Input  : group_list    shape:17*[13*n]
        #   Outpot : distribution  shape:17*[13*20] 说明：20是对区间范围的分桶区间
        print("7.5 distribution")
        print(self.host_train_X.shape)

        distribution = []
        for i in range(len(group_list)):
            dim = []
            for j in range(self.host_train_X.values.shape[1]):
                counts, bin_edges = np.histogram(group_list[i][1].iloc[:,j].values, bins=80)
                dim.append(counts)
            distribution.append(dim)
        distribution = np.array(distribution)

        self.distribution = distribution
        print("distribution:",distribution.shape)


        # TODO:对每个group中 每行数据判断一下它是否在3sigma范围内
        # group_large = []
        # for gro in group
        #     group_small = []
        #     flag = True
        #     for i in every line:
        #         for every col in gro:
        #             if x out of 3sigma:
        #                 flag = False
        #         if flag:
        #             group_small.append(x)   
        #     group_large.append(group_small)  
        
        group_large = []
        delete_list = []
        for i, element in enumerate(self.group):
            #print(i, element)
            group_small = []
            delete_num = 0
            print(element[1].values[:,0:13].shape)
            sheet = element[1].values[:,0:13]
            print("sheet.shape:", sheet.shape)
            sheet_std = np.std(sheet, axis=0)
            print("sheet_std, sheet_std.shape:", sheet_std, sheet_std.shape)
            sheet_mean = np.mean(sheet, axis=0)
            print("sheet_mean, sheet_mean.shape:", sheet_mean, sheet_mean.shape)
            for line in range(0, sheet.shape[0]):
                flag = True
                for col in range (0, sheet.shape[1]):
                    if self.out_of_sigma(sheet[line][col], sheet_std[col], sheet_mean[col]):
                        flag = False
                        # print(col)
                if flag:
                    group_small.append(sheet[line])
                else:
                    delete_num += 1
            group_small = np.array(group_small)
            group_large.append(group_small)
            delete_list.append(delete_num)
        
        print("delete_list:",delete_list)
        # print("group:",group_large)
        self.after_filter_group = np.array(np.squeeze(group_large))
        np.save("group_list.npy",self.after_filter_group)
        # self.write_list_to_json(self.after_filter_group, "grouplist")
        print("self.after_filter_group:", len(self.after_filter_group[0]))
        b = np.squeeze(self.after_filter_group)
        print("b.shape:", b[0].shape)


        

        import matplotlib.pyplot as plt
        for i in range(0,12):
            data = self.after_filter_group[i]
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 9))
            axes.violinplot(data,  points=100, widths=0.5,
                                showmeans=True, showextrema=True, showmedians=True)
            # fig.suptitle("Violin0")
            fig.subplots_adjust(hspace=0.4)
            plt.savefig("violin"+str(i)+".png")

    def out_of_sigma(self, ele, ele_std, ele_mean):
        if abs(ele - ele_mean)/ele_std > 3:
            return True

        


    # def write_list_to_json(self, list, json_file_name):
    #     import json
    #     """
    #     将list写入到json文件
    #     :param list:
    #     :param json_file_name: 写入的json文件名字
    #     :param json_file_save_path: json文件存储路径
    #     :return:
    #     """
    #     with open(json_file_name, 'w') as  f:
    #         json.dump(list, f)

    

        # print(group_list[0][1].iloc[:,0:13])


        # print(self.train_with_class)

        # import seaborn as sns
        # import matplotlib.pyplot as plt

        # tips = self.train_with_class
        # ax = sns.violinplot(x="class", y="x1", data=tips)
        # plt.savefig("x0.png")


        # print(tips)

        # plt.show()

        # data = np.array(group_list[0][1].iloc[:,0:13].values)
        # print(data.shape)
        # print(distribution[5].T)


# """         import matplotlib.pyplot as plt
#         for i in range(0,5):
#             data = np.array(group_list[i][1].iloc[:,0:13].values)
#             fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 9))
#             axes.violinplot(data,  points=100, widths=0.5,
#                                 showmeans=True, showextrema=True, showmedians=True)
#             # fig.suptitle("Violin0")
#             fig.subplots_adjust(hspace=0.4)
#             plt.savefig("violin"+str(i)+".png") """


        # plt.show()

        # Plot
        # import matplotlib.pyplot as plt
        # kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
        # plt.hist(group_list[0][1].iloc[:,0].values, **kwargs)
    


    def Muti_tsne(self, X, save = False):
        """ tsne 降维到2维
                输入：local_feature: `10000*13` self.host_train_X
                输出：2D_coordinates `10000*2` """      
        from MulticoreTSNE import MulticoreTSNE as TSNE  
        start = time.time()
        embeddings = TSNE(n_jobs=8, perplexity=400, early_exaggeration = 1000, verbose=1).fit_transform(X)
        end = time.time()
        print("消耗时间：", end-start)

        if save:
            vis_x = embeddings[:, 0]
            vis_y = embeddings[:, 1]  
            dataframe = pd.DataFrame({'x':vis_x,'y':vis_y})
            dataframe.to_csv("./raw_data/credit_embeddings.csv",index=False,sep=',')

        return embeddings

    def ElbowMethod(self, array, max_cluster_number, min_cluster_number=1, init = 'k-means++', 
                    max_iter = 300, n_init = 50, random_state = None, remark=True):

        wcss = np.zeros(max_cluster_number) # a future within_cluster_sum_squares array
        
        for i in range(max_cluster_number):
            n = i+1 # we are to start with at least 1 cluster
            kmeans = KMeans(n_clusters = n, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
            kmeans.fit(array)
            wcss[i] = kmeans.inertia_
        
        cosines = -1 * np.ones(max_cluster_number-2)# the agles are between cluster numbers, so 
        # 1st and last do not count
        
        for i in range(max_cluster_number-2):
        # at first check if the point is below a segment midpoint connecting its neighbors
            if (wcss[i+1] < (wcss[i+2]+wcss[i])/2 ):
                cosines[i]= (-1+(wcss[i]-wcss[i+1])*(wcss[i+2]-wcss[i+1]))/ \
                ((1+(wcss[i]-wcss[i+1])**2)*(1+ (wcss[i+2]-wcss[i+1])**2))**.5
        
        if remark:
            print("""Remark:\n\tRemember that the K-means method is randomized and may yield different results for different runs.
            If each time repeated applications of the method(with different random_state parameters)yield  
            different values for the optimal number then Elbow method with K-means does not work well 
            on the array. The method produces roundish clusters and they may be not suitable for 
            your objective.""")
        
        return (np.flip(np.argsort(cosines[1:-1]))+2+min_cluster_number)[0]


    def metric(self):
        return 0

    # def three_sigma(self):
    #     # 去除每个data set 中3sigma以外的数据
    #     mean = 
    #     std = 
    #     indices = []
    #     for i in listA:
    #         if abs((i-mean)/3/std) < 1:
    #             indices.append(i)

    def filter(self):
        #TODO:对每个group中 每行数据判断一下它是否在3sigma范围内
        # group_large = []
        # for gro in group
        #     group_small = []
        #     flag = True
        #     for i in every line:
        #         for every col in gro:
        #             if x out of 3sigma:
        #                 flag = False
        #         if flag:
        #             group_small.append(x)   
        #     group_large.append(group_small)  
                   

        group_large = []
        for i, element in enumerate(self.group):
            print(i, element)
        
        return 0






if __name__ == '__main__':
    dataset = DataSetLoader(args.data_name, test_ratio=args.test_ratio, pred_ratio=args.pred_ratio)
