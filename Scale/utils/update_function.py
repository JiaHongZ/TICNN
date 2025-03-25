import torch


def neighborhood_fn(input, current_sigma):
    '''e^(-(input / sigma^2))'''
    input.div_(current_sigma ** 2)
    input.neg_()
    input.exp_()
    return input

def updateW_linear(code, W, updateW_neurons, rw, sigma):
    # 选top10 以学习率rw更新
    dists = torch.sort(code, descending=False, out=None)
    bmu_indexes = dists.indices[:updateW_neurons]
    # print(bmu_indexes)
    # 抑制周围的3个神经元 rw * 0.5
    location = torch.Tensor([x for x in range(len(W))]).cuda()
    # print(bmu_indexes)
    for i in range(updateW_neurons):
        # print(W[bmu_indexes[i]])
        W[bmu_indexes[i]] = W[bmu_indexes[i]] + rw*2 # 因为后面会减去一个rw
        # print('aa',W[bmu_indexes[i]])
        # print(bmu_indexes[i])
        distance_square = location - bmu_indexes[i]
        distance_square.pow_(2)
        lr_locations = neighborhood_fn(distance_square, sigma)
        # print(lr_locations.mul_(rw))
        W -= lr_locations.mul_(rw)*0.1
    return W

def updateW_linear_addbound(code, W, updateW_neurons, rw, sigma):
    # 选top10 以学习率rw更新
    dists = torch.sort(code, descending=False, out=None)
    bmu_indexes = dists.indices[:updateW_neurons]
    # print(bmu_indexes)
    # 抑制周围的3个神经元 rw * 0.5
    location = torch.Tensor([x for x in range(len(W))]).cuda()
    # print(bmu_indexes)
    for i in range(updateW_neurons):
        # print(W[bmu_indexes[i]])
        W[bmu_indexes[i]] = W[bmu_indexes[i]] + rw*2 # 因为后面会减去一个rw
        # print('aa',W[bmu_indexes[i]])
        # print(bmu_indexes[i])
        distance_square = location - bmu_indexes[i]
        distance_square.pow_(2)
        lr_locations = neighborhood_fn(distance_square, sigma)
        # print(lr_locations.mul_(rw))
        W -= lr_locations.mul_(rw)*0.1
    W[W>2] = 2
    W[W<0.5] = 0.5
    return W

def updateW_linear_addbound_down(code, W, updateW_neurons, rw, sigma):
    # 选top10 以学习率rw更新
    dists = torch.sort(code, descending=False, out=None)
    bmu_indexes = dists.indices[:updateW_neurons]
    # print(bmu_indexes)
    # 抑制周围的3个神经元 rw * 0.5
    location = torch.Tensor([x for x in range(len(W))]).cuda()
    # print(bmu_indexes)
    for i in range(updateW_neurons):
        # print(W[bmu_indexes[i]])
        W[bmu_indexes[i]] = W[bmu_indexes[i]] + rw*2 # 因为后面会减去一个rw
        # print('aa',W[bmu_indexes[i]])
        # print(bmu_indexes[i])
        distance_square = location - bmu_indexes[i]
        distance_square.pow_(2)
        lr_locations = neighborhood_fn(distance_square, sigma)
        # print(lr_locations.mul_(rw))
        W -= lr_locations.mul_(rw)*0.1
    W[W<0.5] = 0.5
    return W

def updateW_linear2(code, W, updateW_neurons, rw, sigma):
    # 选top10 以学习率rw更新
    code_mean = code.mean()
    # 谨慎，这样弄出来的全是-1
#    code[code>code_mean] = 1
#   code[code<=code_mean] = -1
    code[code<=code_mean] = -1
    code[code>code_mean] = 1
    W += code * rw

    return W

def updateW_linear_addbound2(code, W, updateW_neurons, rw, sigma):
    # 选top10 以学习率rw更新
    code_mean = code.mean()
    code[code<=code_mean] = -1
    code[code>code_mean] = 1
    W += code * rw
    W[W>2] = 2
    W[W<0.5] = 0.5
    return W