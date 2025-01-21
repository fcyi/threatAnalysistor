import numpy as np
import matplotlib.pyplot as plt


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


class AHP:
    """相关信息的传入和准备"""

    def __init__(self, array):
        # 记录矩阵相关信息
        self.array = array.copy()
        # 记录矩阵大小
        self.n = array.shape[0]
        assert self.n <= 20, 'shape is invalid or too big'
        erro = False
        for r_ in range(self.n):
            for c_ in range(r_, self.n):
                if r_ == c_:
                    if array[r_, c_] == 1:
                        erro = True
                        break
                else:
                    if array[r_, c_] * array[c_, r_] != 1:
                        erro = True
                        break
            if erro:
                break
        assert not erro, 'array have meet error'

        # 初始化RI值，用于一致性检验
        self.RI_list = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46,
                        1.49, 1.52, 1.54, 1.56, 1.58, 1.59, 1.5943, 1.6064,
                        1.6133, 1.6207, 1.6292]
        self.eig_val = None
        self.eig_vector = None
        # 矩阵的最大特征值
        self.max_eig_val = None
        # 矩阵最大特征值对应的特征向量
        self.max_eig_vector = None

    def test_consist(self):
        """一致性判断"""
        if self.n <= 2:
            return True
        # 矩阵的一致性指标CI
        if self.max_eig_val is not None:
            CI_val = (self.max_eig_val - self.n) / (self.n - 1)
        else:
            weight_ = self.cal_weight__by_geometric_method()
            AW_ = self.array * weight_
            AWSum_ = np.sum(AW_, axis=1)
            lbdMax_ = np.mean(AWSum_ / weight_)
            CI_val = (lbdMax_ - self.n) / (self.n - 1)

        # 矩阵的一致性比例CR
        CR_val = CI_val / (self.RI_list[self.n - 1])
        # # 打印矩阵的一致性指标CI和一致性比例CR
        # print("判断矩阵的CI值为：" + str(CI_val))
        # print("判断矩阵的CR值为：" + str(CR_val))
        # # 进行一致性检验判断
        # if self.n <= 2:  # 当只有两个子因素的情况
        #     print("仅包含两个子因素，不存在一致性问题")
        # else:
        #     if CR_val < 0.1:  # CR值小于0.1，可以通过一致性检验
        #         print("判断矩阵的CR值为" + str(CR_val) + ",通过一致性检验")
        #         return True
        #     else:  # CR值大于0.1, 一致性检验不通过
        #         print("判断矩阵的CR值为" + str(CR_val) + "未通过一致性检验")
        #         return False
        return CR_val < 0.1

    def cal_weight_by_arithmetic_method(self):
        """算术平均法求权重"""
        # 求矩阵的每列的和
        col_sum = np.sum(self.array, axis=0)
        # 将判断矩阵按照列归一化
        array_normed = self.array / col_sum
        # 计算权重向量
        # array_weight = np.sum(array_normed, axis=1) / self.n
        array_weight = np.sum(array_normed, axis=1)
        array_weight /= np.sum(array_weight)
        # 打印权重向量
        print("算术平均法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight

    def cal_weight__by_geometric_method(self):
        """几何平均法求权重"""
        # 求矩阵的每列的积
        col_product = np.product(self.array, axis=0)
        # 将得到的积向量的每个分量进行开n次方
        array_power = np.power(col_product, 1 / self.n)
        # 将列向量归一化
        array_weight = array_power / np.sum(array_power)
        # 打印权重向量
        print("几何平均法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight

    def cal_weight__by_eigenvalue_method(self):
        """特征值法求权重"""
        # 矩阵的特征值和特征向量
        self.eig_val, self.eig_vector = np.linalg.eig(self.array)
        # 矩阵的最大特征值
        self.max_eig_val = np.max(self.eig_val)
        # 矩阵最大特征值对应的特征向量
        self.max_eig_vector = self.eig_vector[:, np.argmax(self.eig_val)].real
        # 将矩阵最大特征值对应的特征向量进行归一化处理就得到了权重
        array_weight = self.max_eig_vector / np.sum(self.max_eig_vector)
        # 打印权重向量
        print("特征值法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight


class IAHP:
    """相关信息的传入和准备"""

    def __init__(self, array):
        # 记录矩阵大小
        self.m, self.n = array.shape[:2]
        assert array.shape.__len__() == 3 and self.m == self.n and array[0, 0].shape[0] == 2, 'shape have meet error'
        erro = False

        for r_ in range(self.n):
            for c_ in range(r_, self.n):
                if r_ == c_:
                    if array[r_, c_, 0] != 1 or array[r_, c_, 0] != 1:
                        erro = True
                        break
                else:
                    if array[r_, c_, 0] > array[r_, c_, 1]:
                        erro = True
                        break
            if erro:
                break

        assert not erro, 'array have meet error'

        # 记录矩阵相关信息
        self.array = array.copy()
        self.k_ = None
        self.l_ = None

    def test_consist(self):
        """一致性判断"""
        return (self.k_ <= 1) and (self.l_ >= 1)

    def weight_iahp(self):
        ALower_ = np.copy(self.array[:, :, 0])
        AUpper_ = np.copy(self.array[:, :, 1])
        kVec_ = np.sum(AUpper_, axis=0)
        lVec_ = np.sum(ALower_, axis=0)
        self.k_ = np.sqrt(np.sum(1. / kVec_))
        self.l_ = np.sqrt(np.sum(1. / lVec_))

        eigValL_, eigVecL_ = np.linalg.eig(ALower_)
        eigValU_, eigVecU_ = np.linalg.eig(AUpper_)
        eigVecML_ = eigVecL_[:, np.argmax(eigValL_)].real
        eigVecMU_ = eigVecU_[:, np.argmax(eigValU_)].real

        eigVecML_ /= eigVecML_.sum()
        eigVecMU_ /= eigVecMU_.sum()

        arrayWeight_ = self.k_ * eigVecML_ + self.l_ * eigVecMU_
        arrayWeight_ /= np.sum(arrayWeight_)
        return arrayWeight_


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


# Function: Rank
def ranking(flow):
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show()
    return


def marcos_method(dataset, weights, criterion_type=None, graph=True, verbose=True):
    # MARCOS (Measurement of Alternatives and Ranking according to COmpromise Solution)
    # dataset: m*n, m个目标的不同指标值
    # weights: 不同指标的权重
    X       = np.copy(dataset)/1.0
    n       = X.shape[1]
    weights = np.copy(weights)

    # 对数据进行标准化，并且获取每一列上的最大最小值
    minV_ = np.min(X, axis=0)
    maxV_ = np.max(X, axis=0)
    # worst = [minV_[j] if criterion_type[j] == 'max' else maxV_[j] for j in range(n)]
    # best = [maxV_[j] if criterion_type[j] == 'max' else minV_[j] for j in range(n)]
    # worst = [worst[j]/best[j] if criterion_type == 'max' else best[j]/worst[j] for j in range(n)]
    # best = [best[j]/best[j] for j in range(n)]
    worst = np.array([minV_[j]/maxV_[j] for j in range(n)])  # saai
    best = np.array([1]*n)  # sai
    if criterion_type is not None:
        for j in range(n):
            X[:, j] = norm(X[:, j], 2 if criterion_type[j] == 2 else 3)  # nj = xj / saj

    # 对权值矩阵以及最大最小向量按列进行加权
    best = best*weights
    worst = worst*weights
    V     = X * weights  # vij = nij*wj

    S     = V.sum(axis = 1)  # Sj
    k_n   = S / np.sum(worst)  # K- = Sj / Saaj
    k_p   = S / np.sum(best)  # K+ = Sj / Saj
    # print('*'*10)
    # print(worst)
    # print(best)
    # print(worst.sum())
    # print(S)
    # print(k_n)
    # print(k_p)
    f_k_n = k_p / (k_p + k_n)  # f(K-)
    f_k_p = k_n / (k_p + k_n)  # f(K+)
    f_k   = (k_p + k_n) / (1 + ((1 - f_k_p) / f_k_p) + ((1 - f_k_n) / f_k_n))  # f(K)
    if verbose:
        for i in range(0, f_k.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(f_k[i], 2)))
    if graph:
        flow = np.copy(f_k)
        flow = np.reshape(flow, (f_k.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, f_k.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return f_k, f_k_p, f_k_n


def test_iaph(A):
    print('=======================================================================')
    ALower_ = np.copy(A[:, :, 0])
    AUpper_ = np.copy(A[:, :, 1])
    kVec_ = np.sum(AUpper_, axis=0)
    lVec_ = np.sum(ALower_, axis=0)
    k_ = np.sqrt(np.sum(1. / kVec_))
    l_ = np.sqrt(np.sum(1. / lVec_))
    assert (k_ <= 1) and (l_ >= 1), 'the correspondence of matrix can not meet requires'

    eigValL_, eigVecL_ = np.linalg.eig(ALower_)
    eigValU_, eigVecU_ = np.linalg.eig(AUpper_)
    eigVecML_ = eigVecL_[:, np.argmax(eigValL_)].real
    eigVecMU_ = eigVecU_[:, np.argmax(eigValU_)].real

    eigVecML_ /= eigVecML_.sum()
    eigVecMU_ /= eigVecMU_.sum()

    sig_ = k_ * eigVecML_ + l_ * eigVecMU_
    print(k_*eigVecML_)
    print(l_*eigVecMU_)
    sig_ /= np.sum(sig_)
    return sig_


def comprise_iahp(weightSub_, wwList_):
    nIdxs_ = wwList_.__len__()
    wwSub_ = [weightSub_[i_]*wwList_[i_] for i_ in range(nIdxs_)]
    wwSubL_ = []
    for i_ in range(nIdxs_):
        wwSubL_ += wwSub_[i_].tolist()
    wwSubA_ = np.array(wwSubL_)
    wwSubA_ /= np.sum(wwSubA_)
    return wwSubA_


def comprise_iahp_critic(wIahp_, wCritic_):
    wCom_ = np.sqrt(wIahp_ * wCritic_)
    wCom_ /= np.sum(wCom_)
    return wCom_


def plot_weight(sub_, obj_, com_, savePath_=None):
    # 能力标签
    assert sub_.shape[0] == obj_.shape[0] and sub_.shape[0] == com_.shape[0], 'dim meet errors'
    dims_ = sub_.shape[0]
    abilityLabel_ = [str(i) for i in range(dims_)]
    # fig_ = plt.figure(4, figsize=(16, 16))
    fig_ = plt.figure(figsize=(12, 12))
    # 生成4个子图
    ax1_ = fig_.add_subplot(1, 1, 1, projection='polar')
    # 平均分成6份,首尾相连
    theta_ = np.linspace(0, 2 * np.pi, dims_, endpoint=False)
    theta_ = np.append(theta_, theta_[0])

    ax1_.plot(theta_, np.append(sub_, sub_[0]), 'red', label='subjective')
    # ax1.fill(theta_, np.append(sub_, sub_[0]), 'red', alpha=0.3)  # 填充
    ax1_.plot(theta_, np.append(obj_, obj_[0]), 'blue', label='objective')
    # ax1.fill(theta_, np.append(sub_, sub_[0]), 'red', alpha=0.3)  # 填充
    ax1_.plot(theta_, np.append(com_, com_[0]), 'green', label='comprise')
    # ax1.fill(theta_, np.append(sub_, sub_[0]), 'red', alpha=0.3)  # 填充

    # 显示刻度
    # ax1_.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.], fontsize=14)
    max_ = np.max(np.array([sub_.max(), obj_.max(), com_.max()]))
    min_ = np.min(np.array([sub_.min(), obj_.min(), com_.min()]))
    seg_ = (max_ - min_) / 5.
    ax1_.set_rgrids([seg_+min_, 2*seg_+min_, 3*seg_+min_, 4*seg_+min_], fontsize=14)

    ax1_.set_xticks(theta_[:-1])
    ax1_.set_xticklabels(abilityLabel_, fontsize=14)
    ax1_.set_title('Evaluation index weight radar chart', color='black', size=15, y=1.1)

    plt.legend(loc=(0.88, 0.82), fontsize=14)
    if not savePath_:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(savePath_)
        plt.close()


def create_multi_bars(labels, datas, tick_step=1, group_gap=0.2, bar_gap=0, savePath_=None):
    '''
    labels : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    # kind_ = ['ppp', 'ppo', 'poo', 'opo', 'oto', 'ooo']
    kind_ = ['ppp', 'ooo']
    # x为每组柱子x轴的基准位置
    x = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    print(group_num)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    plt.figure()
    # 绘制柱子
    for index, y in enumerate(datas):
        plt.bar(x + index*bar_span, y, bar_width, label=kind_[index])
    plt.ylabel('Scores')
    plt.title('multi datasets')
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    plt.xticks(ticks, labels)
    plt.legend()
    if not savePath_:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(savePath_)
        plt.close()


if __name__ == '__main__':
    # # exa = np.array([
    # #     [3,   3,   4],
    # #     [2, 1, 1],
    # #     [1, 3,   1]
    # # ], dtype=np.float64)
    # exa = np.array([
    #     [92, 0.8, 0.52, 0.86, 6],
    #     [12, 0.73, 0.38, 0.48, 37],
    #     [68, 0.15, 0.75, 0.28, 49],
    #     [17, 0.16, 0.97, 0.25, 50],
    #     [42, 0.09, 0.82, 0.18, 17],
    #     [20, 0.65, 0.86, 0.88, 39],
    #     [83, 0.19, 0.67, 0.71, 85],
    #     [28, 0.59, 0.74, 0.39, 44],
    #     [93, 0.7, 0.24, 0.1, 47],
    #     [42, 0.23, 0.69, 0.54, 67]
    # ])
    #
    # print(exa.shape)
    #
    # cric = CRITIC(exa, [0, 0, 0, 0, 0])
    # cricW = cric.weight_critic()
    # print(cricW)
    #
    # f_k = marcos_method(exa, cricW, [0, 0, 0, 0, 0])

    A = np.array([
        [  [1, 1],  [1/2, 1],      [1, 2], [1/2, 2]],
        [  [1, 2],    [1, 1],    [3/2, 3],   [1, 3]],
        [[1/2, 1], [1/3, 2/3],     [1, 1], [3/2, 2]],
        [[1/2, 2],   [1/3, 1], [1/2, 2/3],   [1, 1]]
    ])
    # A = np.array([
    #     [    [1, 1],     [2, 3],      [2, 3], [1/3, 1/2]],
    #     [[1/3, 1/2],     [1, 1],      [2, 3],   [1/4, 1/3]],
    #     [  [1/3, 1/2], [1/3, 1/2],     [1, 1], [1/4, 1/3]],
    #     [[2, 3],   [3, 4], [3, 4],   [1, 1]]
    # ])

    B1 = np.array([
        [  [1, 1], [1, 3/2]],
        [[2/3, 1],   [1, 1]]
    ])

    B2 = np.array([
        [[1, 1], [2/3, 2]],
        [[1/2, 3/2], [1, 1]]
    ])

    B3 = np.array([
        [[1, 1], [1 / 3, 1/2]],
        [[2, 3], [1, 1]]
    ])

    B4 = np.array([
        [[1, 1], [1 / 3, 1/2], [1/3, 2]],
        [[2, 3], [1, 1], [1, 3]],
        [[1 / 2, 3], [1 / 3, 1], [1, 1]]
    ])

    wa = test_iaph(A)
    wb1 = test_iaph(B1)
    wb2 = test_iaph(B2)
    wb3 = test_iaph(B3)
    wb4 = test_iaph(B4)
    wz2 = comprise_iahp([wb1, wb2, wb3, wb4], wa.tolist())

    objectW = np.array([
        # [1, 1, 1, 0.85, 0.86, 0.43, 1, 1, 1],
        [0, 0.16, 0.24, 0.46, 0.86, 0.25, 0, 0.13, 1],
        [0.27, 0.68, 0.10, 0.54, 0.14, 0.06, 0.11, 0.10, 0.48],
        [1, 0.68, 0, 0.85, 0.86, 0.25, 0.81, 0.60, 0.74],
        [0.27, 0, 0.56, 0.54, 0.5, 0.43, 1, 1, 0.74],
        [0.33, 0.79, 0.61, 0.51, 0.14, 0.11, 0.16, 0.05, 0],
        [0.33, 1, 1, 0.51, 0.5, 0.18, 0.06, 0, 0.26],
        # [0, 0, 0, 0.46, 0.14, 0.06, 0, 0, 0]
    ])
    #
    critic = CRITIC(objectW, [0]*objectW.shape[1])
    wz3 = critic.weight_critic()
    # ewm = EWM(objectW, [0]*objectW.shape[1])
    # ewm.weight_ewm()
    # wz3 = critic.weight_critic_plus(ewm.entropy_)
    wz4 = comprise_iahp_critic(wz2, wz3)
    plot_weight(wz2, wz3, wz4, 'zb1.png')
    # print(critic.weight_critic())

    # wa = np.array([0.2330, 0.1494, 0.2965, 0.3212])
    # wb1 = np.array([0.4487,0.5513])
    # wb2 = np.array([0.4604, 0.5396])
    # wb3 = np.array([0.7126,0.2847])
    # wb4 = np.array([0.4556, 0.1773, 0.3672])
    wz2_ = np.array([0.1045, 0.1284, 0.0688, 0.0806,0.2113,0.0852,0.1463,0.0569, 0.1179])
    wz3_ = np.array([0.1100, 0.1166, 0.1153, 0.1168, 0.1153, 0.0983, 0.1077, 0.1063, 0.1138])
    wz4_ = comprise_iahp_critic(wz2_, wz3_)
    plot_weight(wz2_, wz3_, wz4_, 'zb2.png')

    objW = objectW.copy()
    fk, fkp, fkn = marcos_method(objectW, wz4, [2] * wz2.shape[0])  # ooo
    fk1_, fkp1_, fkn1_ = marcos_method(objectW, wz4_, [2] * wz2.shape[0])  # ppo
    fk_ = np.array([0.4330, 0.3011, 0.7381, 0.6178, 0.3342, 0.4765])  # ppp

    from pymcdm.methods import MARCOS
    from pymcdm.weights import critic_weights
    # marcos = MARCOS()
    # fk2_ = marcos(objectW, wz4_, [1] * wz2.shape[0])  # ppt

    wz33_ = critic_weights(objectW)
    print('='*10)
    print(wz3)
    # print(wz3n_)
    print(wz3_)
    wz43_ = comprise_iahp_critic(wz2, wz33_)
    fk3_, _, _ = marcos_method(objectW, wz43_, [2] * wz2.shape[0])  # oto

    wz45_ = comprise_iahp_critic(wz2, wz3_)
    fk5_, _, _ = marcos_method(objectW, wz45_, [2] * wz2.shape[0])  # opo

    wz44_ = comprise_iahp_critic(wz2_, wz3)
    fk4_, _, _ = marcos_method(objectW, wz44_, [2] * wz2.shape[0])  # poo

    # 'ppp', 'ppo', 'poo', 'opo', 'oto', 'ooo'
    # fks_ = np.vstack([fk_, fk1_, fk4_, fk5_, fk3_, fk]).tolist()
    fks_ = np.vstack([fk_, fk]).tolist()
    label = ['obj_{}'.format(i+1) for i in range(fk.shape[0])]

    create_multi_bars(label, fks_, bar_gap=0.1, savePath_='compare.png')









