## 数据说明

- 对本地数据进行降维，12维 10000行数据

- 数据说明：全部数据有30000行、本地维度12维、联邦维度9维

- 训练数据 + 验证数据：10000行

- 测试数据：20000行



## 后端流程

1. tsne 降维到2维
   1. 输入：local_feature: `10000*12`
   2. 输出：2D_coordinates `10000*2`
2. 使用 elbow 算法 算出 需要多少个cluster
   1. 输入：local_feature `10000*2`
   2. 输出：class_num `n`
3. 用kmeans聚类
   1. 输入：2D_coordinates `10000*2`  class_num `n`
   2. 输出：data `10000*1`

