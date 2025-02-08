import math
import numpy as np
import matplotlib.pyplot as plt


def draw_response(xys_):
    plt.figure()
    cot_ = 0
    for xy in xys_:
        x_, y_ = xy[0], xy[1]
        plt.plot(x_, y_, label=str(cot_))
        cot_ += 1
    plt.legend()
    plt.show()


class EEBInfos:
    def __init__(self, infos_):
        self.infos_ = infos_
        self.xy_ = None
        self.land_ = None


class EssentialElementBuilder:
    """type: 0-海上目标, 1-地面目标, 2-空中目标"""
    def __init__(self):
        pass


def ee_distance(dis_, cfgDist_, type_):
    pass


def metric_tmp0(d_, scalar_, bias_, Min_, Max_, power_=1):
    assert Min_ != Max_, "Min_ should not equal to Max_"
    if power_ == 1:
        return bias_ + scalar_ * (d_ - Min_) / (Max_ - Min_)
    else:
        return bias_ + scalar_ * ((d_-Min_) / (Max_-Min_))**power_


def metric_tmp1(d_, scalar_, bias_, D0_):
    return (1.-bias_)*math.exp(scalar_*(d_-D0_))+bias_


def metric_tmp2(d_, scalar_, bias_, scalar1_):
    return scalar1_*math.exp(scalar_*d_)+bias_


def distance_instance0(d_, scalar_, bias_, Ds_, power_=1):
    dee_ = 0
    DLen_ = len(Ds_)
    assert DLen_ >= 1, "Ds_ length must larger than 1"

    if d_ > Ds_[-1]:
        dee_ = bias_
    elif d_ <= Ds_[0]:
        dee_ = 1.
    else:
        DLen_ -= 1
        for i_ in range(DLen_):
            if Ds_[i_] < d_ <= Ds_[i_+1]:
                dee_ = metric_tmp0(d_, -scalar_, bias_+(DLen_-i_)*scalar_, Ds_[i_], Ds_[i_+1], power_)
                break
    return dee_


def distance_instance1(d_, scalar_, bias_, D0_):
    dee_ = 0
    if d_ < D0_:
        dee_ = 1
    else:
        dee_ = metric_tmp1(d_, scalar_, bias_, D0_)
    return dee_


def distance_instance2(d_, scalar_, bias_, D0_, D1_, proportion=0.5):
    dee_ = 0
    D2_ = D1_+(D1_-D0_)*proportion
    pt0_ = (D0_, 1)
    pt1_ = (D1_, (proportion+bias_)/(1+proportion))
    pt2_ = (D2_, bias_)

    if d_ <= D0_:
        dee_ = 1
    elif d_ > D2_:
        dee_ = bias_
    elif D0_ < d_ <= D1_:
        scalar1_ = (pt1_[1]-pt0_[1]) / (1-math.exp(scalar_*(pt0_[0]-pt1_[0])))
        bias1_ = pt1_[1]-scalar1_
        dee_ = metric_tmp2(d_-pt1_[0], scalar_, bias1_, scalar1_)
    elif D1_ < d_ <= D2_:
        scalar1_ = (pt2_[1]-pt1_[1]) / (math.exp(-scalar_*(pt2_[0]-pt1_[0]))-1)
        bias1_ = pt1_[1]-scalar1_
        dee_ = metric_tmp2(d_-pt1_[0], -scalar_, bias1_, scalar1_)
    return dee_


def distance_instance3(d_, rr_, rR_, rb_, rB_):
    # 用于双方飞机的打击范围和探测范围都已知的情况，并且需要这些范围都互不相等，否则就仅基于己方飞机来考虑威胁度量
    assert rr_ <= rR_ and rb_ <= rB_
    dee_ = 0
    if (rB_ > rR_ > rb_) and (rb_ > rr_):
        if 0 <= d_ <= rr_:
            dee_ = metric_tmp0(d_, 0.3, 0.5, 0, rr_)
        elif rr_ < d_ <= rb_:
            dee_ = metric_tmp0(d_, 0.5, 0.3, rb_, rr_)
        elif rb_ < d_ <= rR_:
            dee_ = metric_tmp0(d_, 0.7, 0.3, rb_, rR_)
        elif rR_ < d_ <= rB_:
            dee_ = metric_tmp0(d_, 1, 0, rB_, rR_)
        else:
            dee_ = 0
    elif (rB_ > rb_ > rR_) and (rR_ > rr_):
        if 0 <= d_ <= rr_:
            dee_ = metric_tmp0(d_, 0.3, 0.5, 0, rr_)
        elif rr_ < d_ <= rR_:
            dee_ = metric_tmp0(d_, 0.2, 0.8, rr_, rR_)
        elif rR_ < d_ <= rb_:
            dee_ = 1
        elif rb_ < d_ <= rB_:
            dee_ = metric_tmp0(d_, 1, 0, rB_, rb_)
        else:
            dee_ = 0
    elif (rB_ > rR_ > rr_) and (rr_ > rb_):
        if 0 <= d_ <= rb_:
            dee_ = metric_tmp0(d_, -0.3, 0.5, 0, rb_)
        elif rb_ < d_ <= rr_:
            dee_ = metric_tmp0(d_, 0.1, 0.2, rb_, rr_)
        elif rr_ < d_ <= rR_:
            dee_ = metric_tmp0(d_, 0.7, 0.3, rr_, rR_)
        elif rR_ < d_ <= rB_:
            dee_ = metric_tmp0(d_, 1, 0, rB_, rR_)
        else:
            dee_ = 0
    elif (rR_ > rB_ > rr_) and (rr_ > rb_):
        if 0 <= d_ <= rb_:
            dee_ = metric_tmp0(d_, -0.3, 0.5, 0, rb_)
        elif rb_ < d_ <= rr_:
            dee_ = metric_tmp0(d_, 0.1, 0.2, rb_, rr_)
        elif rr_ < d_ <= rB_:
            dee_ = metric_tmp0(d_, 0.3, 0, rB_, rr_)
        else:
            dee_ = 0
    elif (rR_ > rr_ > rB_) and (rB_ > rb_):
        if 0 <= d_ <= rb_:
            dee_ = metric_tmp0(d_, -0.3, 0.5, 0, rb_)
        elif rb_ < d_ <= rB_:
            dee_ = metric_tmp0(d_, 0.2, 0, rB_, rb_)
        else:
            dee_ = 0
    elif (rR_ > rB_ > rb_) and (rb_ > rr_):
        if 0 <= d_ <= rr_:
            dee_ = metric_tmp0(d_, 0.5, 0.5, 0, rr_)
        elif rr_ < d_ <= rb_:
            dee_ = metric_tmp0(d_, 0.7, 0.3, rr_, rb_)
        elif rb_ < d_ <= rB_:
            dee_ = metric_tmp0(d_, 0.3, 0, rB_, rb_)
        else:
            dee_ = 0

    return dee_


def test_():
    # D0_ = 10
    # D1_ = 50
    # D2_ = 100
    # bias_ = 0.4
    # scalar_ = 0.3
    #
    # x_ = list(range(D2_+int(0.1*D2_)))
    # y_ = [distance_instance0(xt_, scalar_, bias_, (D0_, D1_, D2_)) for xt_ in x_]
    # draw_response([[x_, y_]])
    #
    # # D0_ = 400
    # # scalar_ = -0.16
    # # bias_ = 0.2
    # # xs_ = list(range(D0_ + int(2 * D0_)))
    # # y0_ = [distance_instance1(xt_, scalar_, bias_, D0_) for xt_ in xs_]
    # # scalar_ = -0.008
    # # bias_ = 0.15
    # # D0_ = 10
    # # y1_ = [distance_instance1(xt_, scalar_, bias_, D0_) for xt_ in xs_]
    # # D0_ = 200
    # # D1_ = 400
    # # scalar_ = 0.01
    # # y2_ = [distance_instance2(xt_, scalar_, bias_, D0_, D1_) for xt_ in xs_]
    # # draw_response([
    # #     [xs_, y0_], [xs_, y1_],
    # #     [xs_, y2_]])

    # rB_ > rR_ > rb_) and (rb_ > rr_
    # rB_ > rb_ > rR_) and (rR_ > rr_
    # rB_ > rR_ > rr_) and (rr_ > rb_
    pass




if __name__ == '__main__':
    test_()
    pass




