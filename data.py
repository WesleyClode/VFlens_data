import os
import re
import time
import random
import numpy as np
from numpy.core.defchararray import index
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
        shuffled_idx = np.random.permutation(self.all_data.shape[0])

        num_train = int(np.ceil(self.all_data.shape[0] * self._train_ratio))
        self.all_train = self.all_data.iloc[shuffled_idx[:num_train]]
        self.all_pred = self.all_data.iloc[shuffled_idx[num_train:]]

        num_test = int(np.ceil(self.all_data.shape[0] * self._test_ratio))
        self.train = self.all_data.iloc[shuffled_idx[num_test:num_train]]
        self.test = self.all_data.iloc[shuffled_idx[:num_test]]

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
            self.cluster_num = 11
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
        # print(self.train_with_class.groupby("class").agg('mean'))

        #   2.std
        # print(self.train_with_class.groupby("class").agg('std'))

        #   3.similarity
        from sklearn.metrics.pairwise import pairwise_distances
        # from sklearn.metrics.pairwise import cosine_similarity
        group_list = list(self.group)

        # print(group_list[0][1].iloc[:,0:13].values)

        sim_list = []
        for i in range(len(group_list)):
            dis = 0
            sim_ele = pairwise_distances(group_list[i][1].iloc[:,0:13].values, metric="cosine")
            num = sim_ele.shape[0]*(sim_ele.shape[0]-1)/2
            for i in range(sim_ele.shape[0]):
                for j in range(i+1, sim_ele.shape[0]):
                    dis = dis + sim_ele[i][j]
            sim_list.append(dis/num)

        print(sim_list)
        #   4.diversity
        #   5.
        # print(list(group))
        # print(train_with_class.groupby("class").agg('mean'))

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

    def data_merge(self):
        df1 = pd.read_csv("./raw_data/data_class.csv", sep=',')
        df2 = pd.read_csv("./raw_data/credit_embeddings.csv", sep=',')
        frames = [self.host_train_X, self.host_train_Y, df2]
        df = pd.concat(frames)

        return df



if __name__ == '__main__':
    dataset = DataSetLoader(args.data_name, test_ratio=args.test_ratio, pred_ratio=args.pred_ratio)
