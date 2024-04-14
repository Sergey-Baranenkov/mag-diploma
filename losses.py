import torch


def calculate_sisdr(ref, est):
    eps = torch.finfo(ref.dtype).eps

    # Убедимся, что ref и est имеют размерность [batch_size, time]
    assert ref.dim() == est.dim() == 2

    # Вычисление Rss
    Rss = torch.sum(ref * ref, dim=1, keepdim=True)

    # Масштабный коэффициент
    a = torch.sum(ref * est, dim=1, keepdim=True) / (Rss + eps)

    # Истинная и остаточная энергия
    e_true = a * ref
    e_res = est - e_true

    # Расчет SI-SDR
    Sss = torch.sum(e_true ** 2, dim=1)
    Snn = torch.sum(e_res ** 2, dim=1)

    sisdr = 10 * torch.log10((eps + Sss) / (eps + Snn))

    return torch.mean(sisdr)


# Функция потерь SI-SDR
def si_sdr_loss(output_dict, target_dict):
    return -calculate_sisdr(target_dict['segment'], output_dict['segment'])  # Минимизируем отрицательный SI-SDR


def l1(output, target):
    return torch.mean(torch.abs(output - target))


def l1_wav(output_dict, target_dict):
    return l1(output_dict['segment'], target_dict['segment'])


def get_loss_function(loss_type):
    if loss_type == "l1_wav":
        return l1_wav
    elif loss_type == "si_sdr":
        return si_sdr_loss
    else:
        raise NotImplementedError("Error!")
