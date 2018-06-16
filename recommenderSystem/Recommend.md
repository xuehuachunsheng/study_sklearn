### 淘宝上有几亿的买家和千万级的商家，我们需要在每个月初给头部的商家（已知商家的名单）推荐一定数量的用户，便于商家有针对性的进行营销活动。要求：请给出具体的方案和实现方法，包括所用的模型、数据、训练和预测的方法。已有的数据是过去半年所有买家的行为数据，即过去半年的每一天里，单个买家在每个商品上的浏览、收藏、推荐，和购买数量。

### Data
刚开始有想过自己爬取或者是购买数据，但是由于无法爬取到买家浏览，收藏商品的数据。
因此数据选用一个较为近似的数据，阿里云天池数据：User Behavior Data on Taobao/Tmall IJCAI16 Contest

该数据包含三张表：
![这里写图片描述](https://github.com/xuehuachunsheng/study_sklearn/blob/master/recommenderSystem/datatable.png)

数据分析：ijcai2016_taobao.csv这张表表示交易数据，从所含字段中可以得到商家的id(Seller id)和商品的id(Item id). 由于存在商品类别的id，且统计得知有72个不同的类别。因此，推荐过程不考虑Item，转而考虑商品的类别，Online_Action_id字段为0表示用户点击，1表示购买。由于题目要求单个买家在每个商品上的浏览、收藏、推荐，和购买数量。但这里没有收藏和推荐的数据，我们以用户点击表示浏览、收藏和推荐。

### 方案步骤
1. 由于题目要求推荐用户给商家，因此，我们只使用用户的交易数据即ijcai2016_taobao.csv；
2. 根据商品的类别，统计用户购买以及点击同类商品的数量。由于有72个不同的类别，因此每个用户构成两个长度为72维的特征向量，一个ub表示购买数，一个uc表示点击数；
3. 根据Seller_id和Category_id字段统计每个商家含有的商品类别id。这是一个长度为72维的二值向量-s；
4. 计算用户的特征向量和商家的特征向量的相似度。然后根据相似度做一个Rank，选取相似度Top-k的用户推荐给商家。
相似度计算方法如下：
![这里写图片描述](https://github.com/xuehuachunsheng/study_sklearn/blob/master/recommenderSystem/equation.gif)
为了防止分母为0导致的NaN问题，给分母加上一个比较小的数epsilon。

主要代码为recommenderSystem/userFeatureDictAndNeighborComputing.py
中间结果：feature.csv -- 用户购买或者点击某一个类别的商品次数。
这里模型很显然是k近邻方法。不过不一样的是这里的近邻计算标准为商家和用户之间的相似度。
Algorithm Input: 每个用户的类别点击购买次数向量(ub和uc)，每个商家的类别特征向量(s)
Algorithm Output: {商家id: 推荐用户id列表}

参考资料：https://blog.csdn.net/BaiJingting/article/details/51473871