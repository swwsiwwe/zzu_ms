import torch
import pandas as pd
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import copy


def torch2msDICT(filename):
    list1 = pd.read_csv(filename, usecols=[1])
    val = list1.values
    dict = {}
    for i in range(len(list1)):
        name = copy.deepcopy(val[i][0])
        if name.endswith('beta'):
            name = name[:name.rfind('beta')]
            name = name + 'bias'
        elif name.endswith('gamma'):
            name = name[:name.rfind('gamma')]
            name = name + 'weight'
        elif name.endswith('moving_mean'):
            name = name[:name.rfind('moving_mean')]
            name = name + 'running_mean'
        elif name.endswith('moving_variance'):
            name = name[:name.rfind('moving_variance')]
            name = name + 'running_var'
        dict[name] = val[i][0]
    return dict


def getPth(net):
    # pytorch
    pytorch_model = JAFFNet()
    pytorch_weights_dict = pytorch_model.state_dict().keys()
    param_torch_lst = pd.DataFrame(param_torch)
    param_torch_lst.to_csv('param_torch.csv')
    # ms
    mindspore_model = net()
    prams_ms = mindspore_model.parameters_dict().keys()
    prams_ms_lst = pd.DataFrame(prams_ms)
    prams_ms_lst.to_csv('prams_ms.csv')


def pytorch2mindspore(pth_name='res18_py.pth', ckpt_name='res18_py.ckpt'):
    dict = torch2msDICT('prams_ga_ms.csv')
    par_dict = torch.load(pth_name, map_location=torch.device('cpu'))
    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        print('========================py_name', name)
        if name.endswith('num_batches_tracked'):
            continue
        print('========================ms_name', dict[name])
        param_dict['name'] = dict[name]
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, ckpt_name)


if __name__ == '__main__':
    pytorch2mindspore('ganet.pth', 'ganet.ckpt')
