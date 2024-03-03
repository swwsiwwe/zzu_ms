import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


def pytorch2mindspore(pth_name='res18_py.pth', ckpt_name='res18_py.ckpt'):
    # dict = torch2msDICT('prams_ms.csv')
    par_dict = torch.load(pth_name, map_location=torch.device('cpu'))
    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        print('========================name', name)

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, ckpt_name)


if __name__ == '__main__':
    pytorch2mindspore('best_state_dict.pth', 'best_state_dict.ckpt')
