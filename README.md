# ucloud
牛客网-随身云时间黑客大数据挖掘比赛

说明

（1）比赛链接：http://www.nowcoder.com/activity/calendar

（2）仅上传部分代码

比赛思路

（1）选取建模用户（仅对互动率高的小部分用户进行建模）

（2）建模思路：先判断有无行为，然后判断该行为是赞或者踩；由于大部分用户行为比较一致，所以后者可直接统计判断

（3）有无行为建模：样本不均衡的二分类问题，时间衰减采样

（4）特征设计：用户特征 + 帖子特征 + 交互特征

（5）模型选择：rf + gbdt

（6）模型融合：软投票（计算平均概率值）
