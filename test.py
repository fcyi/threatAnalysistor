from pymcdm.methods import MARCOS
from pymcdm.weights import critic_weights
import numpy as np


def norm(vec_, type_):
    min_, max_ = vec_.min(), vec_.max()
    if type_ == 0:  # maxToMax，正向标准化
        if max_ == min_:
            if max_ > 1 or min_ < 0:
                vec_ = np.ones_like(vec_)*0.5
            else:
                pass
        else:
            scale_ = 1. / (max_ - min_)
            vec_ -= min_
            vec_ *= scale_
    elif type_ == 1:  # minToMax，负向标准化
        if max_ == min_:
            if max_ > 1 or min_ < 0:
                vec_ = np.ones_like(vec_) * 0.5
            else:
                pass
        else:
            scale_ = 1. / (min_ - max_)
            vec_ -= max_
            vec_ *= scale_

    elif type_ == 2:  # 正向标准化1
        if max_ == min_:
            if max_ > 1 or min_ < 0:
                vec_ = np.ones_like(vec_) * 0.5
            else:
                pass
        elif max_ == 0:
            min_ = -min_
            vec_ += min_
            vec_ /= min_
        else:
            vec_ /= max_
    elif type_ == 3:  # 负向标准化1
        if max_ == min_:
            if max_ > 1 or min_ < 0:
                vec_ = np.ones_like(vec_) * 0.5
            else:
                pass
        elif (min_ <= 0 and max_ >= 0) or min_ == 0 or max_ == 0:
            vec_ -= min_
            min_, max_ = vec_.min(), vec_.max()
            mid_ = (min_+max_) / 2
            vec_ += mid_
            min_ = vec_.min()
            vec_ = min_ / vec_
        else:
            vec_ = min_ / vec_
    else:
        raise Exception('method have not complement!')

    return vec_


class CRITIC:
    def __init__(self, array_, normTypeList_):
        """
        :param array_: m*n, the number of objects is m, the number of objects is n
        :param normTypeList_:  norm direction
        """
        self.n_ = array_.shape[1]
        assert self.n_ == len(normTypeList_), 'ndims not correspondence'
        self.array_ = array_.copy()
        self.normTypeList_ = normTypeList_

    def weight_critic(self):
        # 1.无量纲过程（规范化）
        for dim_ in range(self.n_):
            self.array_[:, dim_] = norm(self.array_[:, dim_], self.normTypeList_[dim_])

        # 2.计算对比强度
        V_ = np.std(self.array_, axis=0)

        # 3. 计算冲突性
        # A2 = list(map(list, zip(*self.array)))  # 矩阵转置
        # A2_ = self.array_.T
        r_ = np.corrcoef(self.array_, rowvar=False)  # 求皮尔逊相关系数，np.corrcoef默认是对行计算皮尔逊系数、转置一下对列做，除非设置参数rowvar
        f_ = np.sum(1 - r_, axis=0)

        # 4. 计算信息承载量
        C_ = V_ * f_

        print(V_)
        print(C_)

        # 5. 计算权重
        arrayWeight_ = C_ / np.sum(C_)

        return arrayWeight_

    def weight_critic_plus(self, ewmWeight_):
        # 基于熵权法对CRITIC加权法进行改善
        # 参考：https://blog.csdn.net/weixin_53972936/article/details/123337354?spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-123337354-blog-106742082.235%5Ev43%5Epc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-123337354-blog-106742082.235%5Ev43%5Epc_blog_bottom_relevance_base3&utm_relevant_index=13
        # 实际测试效果并没有达到该参考给出的效果，怀疑此改进方式本身就不靠谱
        # 1.无量纲过程（规范化）
        for dim_ in range(self.n_):
            self.array_[:, dim_] = norm(self.array_[:, dim_], self.normTypeList_[dim_])

        # 2.计算对比强度
        V_ = np.std(self.array_, axis=0)

        # 3.计算相关性
        r_ = np.corrcoef(self.array_, rowvar=False)  # 求皮尔逊相关系数，np.corrcoef默认是对行计算皮尔逊系数、转置一下对列做，除非设置参数rowvar
        r_ = np.abs(r_)
        rABSum_ = np.sum(1.-r_, axis=0)

        # 4.计算信息矩阵
        VES_ = ewmWeight_ + V_
        total_ = np.sum(VES_)
        rABSumT_ = total_ * rABSum_

        VEWSR_ = VES_ * rABSumT_
        informVolume_ = VEWSR_ / rABSumT_

        arrayWeight_ = informVolume_ / np.sum(informVolume_)

        return arrayWeight_


class EWM:
    def __init__(self, array_, normTypeList_) -> None:
        """初始化"""
        self.m_ = array_.shape[0]  # m个样本
        self.n_ = array_.shape[1]  # n个指标
        assert self.n_ == len(normTypeList_), 'ndims not correspondence'
        self.array_ = array_.copy()
        self.normTypeList_ = normTypeList_
        self.entropy_ = None

    def weight_ewm(self):
        """
        模型计算权重
        @X: 训练数据样本, np.ndarray
        @normal: 是否需要标准化X数据，默写进行标准化
        @max_val_list: 限定样本指标标准化时最大值
        @min_val_list: 限定样本指标标准化时最小值
        """
        # 1.无量纲过程（规范化）
        for dim_ in range(self.n_):
            self.array_[:, dim_] = norm(self.array_[:, dim_], self.normTypeList_[dim_])

        # 计算比重
        P_ = self.array_ / np.sum(self.array_, axis=0)
        P_ = np.clip(P_, a_min=1e-100, a_max=None)
        # 计算熵值
        e_ = -1 / np.log(self.m_) * np.sum(P_ * np.log(P_), axis=0)
        self.entropy_ = e_.copy()
        # 差异系数
        d_ = 1 - e_
        # 计算权重
        arrayWeight_ = d_ / np.sum(d_)
        return arrayWeight_


objectW = np.array([
        [1, 1, 1, 0.85, 0.86, 0.43, 1, 1, 1],
        [0, 0.16, 0.24, 0.46, 0.86, 0.25, 0, 0.13, 1],
        [0.27, 0.68, 0.10, 0.54, 0.14, 0.06, 0.11, 0.10, 0.48],
        [1, 0.68, 0, 0.85, 0.86, 0.25, 0.81, 0.60, 0.74],
        [0.27, 0, 0.56, 0.54, 0.5, 0.43, 1, 1, 0.74],
        [0.33, 0.79, 0.61, 0.51, 0.14, 0.11, 0.16, 0.05, 0],
        [0.33, 1, 1, 0.51, 0.5, 0.18, 0.06, 0, 0.26],
        [0, 0, 0, 0.46, 0.14, 0.06, 0, 0, 0]
    ])

wz33_ = critic_weights(objectW)

cric = CRITIC(objectW, [0]*objectW.shape[1])
ewm = EWM(objectW, [0]*objectW.shape[1])
ewm.weight_ewm()
wz331_ = cric.weight_critic_plus(ewm.entropy_)

print(wz33_)
print(wz331_)
