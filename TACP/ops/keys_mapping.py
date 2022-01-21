
'''
Implementation of 'Temporal-attentive Covariance Pooling Networks for Action Recognition'
Authors: Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu and Peihua Li.

This file is modified from non-local code.
'''


def adaptive_mapping_old_style_keys(checkpoint, k_new, args):
    if  'resnet152' in args.arch:
        checkpoint = keys_mapping_old_preact(checkpoint, k_new)
    elif 'tea' in args.arch :
        checkpoint = keys_mapping_old_tea(checkpoint, k_new)
    else : #resnet50
        checkpoint = keys_mapping_old_resnet(checkpoint, k_new)

    return checkpoint


def keys_mapping_old_preact(checkpoint, k_new):
    checkpoint = keys_mapping_old_tea(checkpoint, k_new)

    return checkpoint


def keys_mapping_old_resnet(checkpoint, k_new, replace_dict=[], k_old=None):
    if k_old is None:
        k_old = checkpoint.keys()
    # replace_dict = []
    for i in k_old :
        if 'iSQRT' in i:
            i_new = i.replace('layer4.iSQRT.', 'TCP.')

            i_candidates = [i_new.replace('att_module.conv_1', 'TCP_att.TCA.g1'),
                            i_new.replace('att_module.conv_2', 'TCP_att.TCA.g2'),
                            i_new.replace('att_module.sp_att.conv_theta', 'TCP_att.TSA.conv_phi_1'),
                            i_new.replace('att_module.sp_att.conv_phi', 'TCP_att.TSA.conv_phi_2'),
                            i_new.replace('att_module.sp_att.conv_g', 'TCP_att.TSA.conv_phi0'),
                            i_new.replace('att_module.sp_att', 'TCP_att.TSA'),
                            i_new.replace('att_module', 'TCP_att'),
                            i_new]
            i_new = [i for i in i_candidates if i in k_new]
            assert len(i_new) != 0, 'invalid TCP layers' + i

            i_new = i_new[0]

            replace_dict.append((i, i_new))
        elif i in k_new:
            replace_dict.append((i, i))
        else :
            raise KeyError('invalid resume layer ' + i)

    for k, k_new_i in replace_dict:
            v = checkpoint.pop(k)
            checkpoint[k_new_i] = v

    return checkpoint





def keys_mapping_old_tea(checkpoint, k_new, replace_dict=[]):

    k_old = checkpoint.keys()

    for i in k_old :
        if 'iSQRT' in i:
            i_new = i.replace('iSQRT.', 'TCP.')

            i_candidates = [i_new.replace('att_module.conv_1', 'TCP_att.TCA.g1'),
                            i_new.replace('att_module.conv_2', 'TCP_att.TCA.g2'),
                            i_new.replace('att_module.sp_att.conv_theta', 'TCP_att.TSA.conv_phi_1'),
                            i_new.replace('att_module.sp_att.conv_phi', 'TCP_att.TSA.conv_phi_2'),
                            i_new.replace('att_module.sp_att.conv_g', 'TCP_att.TSA.conv_phi0'),
                            i_new.replace('att_module.sp_att', 'TCP_att.TSA'),
                            i_new.replace('att_module', 'TCP_att'),
                            i_new.replace('1.1.', '_bn1.'),
                            i_new.replace('2.1.', '_bn2.'),
                            i_new.replace('.1.', '.'),
                            i_new.replace('.0.', '.'),
                            ]
            i_new = [i for i in i_candidates if i in k_new]
            assert len(i_new) != 0, 'invalid TCP layers' + i

            i_new = i_new[0]
            # print(i + '    ==>     ' + i_new)

            replace_dict.append((i, i_new))
        elif i in k_new:
            replace_dict.append((i, i))
        else :
            raise KeyError('invalid resume layer ' + i)

    for k, k_new_i in replace_dict:
            v = checkpoint.pop(k)
            checkpoint[k_new_i] = v

    return checkpoint


def adaptive_mapping_pretrained_keys(sd,model_dict, args):
    if  'resnet152' in args.arch:
        sd = keys_mapping_preact(sd, model_dict)
    elif 'tea' in args.arch :
        sd = keys_mapping_tea(sd, model_dict)
    else : #resnet50
        sd = keys_mapping_resnet(sd, model_dict)

    return sd

def keys_mapping_tea(sd, model_dict):
    replace_dict = []

    for k, v in sd.items():
        if k not in model_dict and k.replace('module.', 'module.base_model.') in model_dict:
            # print('=> Load after add .base_model : ', k)
            replace_dict.append((k, k.replace('module.', 'module.base_model.')))  # (k_old, k_new)
        elif 'layer_reduce' in k :
            #print('==> load after add .base_model.layer4  :', k)
            if '.1.' in k: #bn
                new_k = k.replace('module.iSQRT.', 'module.base_model.TCP.')
                new_k = new_k.replace('.1.','.')
                new_k = new_k.replace('reduce','reduce_bn')
                replace_dict.append((k, new_k))
                # print(' {}    ==>    {}'.format(k, new_k))
            elif '.0.' in k: #conv
                new_k = k.replace('module.iSQRT.', 'module.base_model.TCP.')
                new_k = new_k.replace('.0.','.')
                replace_dict.append((k, new_k))
                # print('{}    ==>    {}'.format(k, new_k))
            else :
                raise KeyError('lost' + k)
        else:
            print('skip ' + k)

    for k, k_new in replace_dict:
        sd[k_new] = sd.pop(k)

    return sd



def keys_mapping_preact(sd, model_dict):
    sd = keys_mapping_tea(sd, model_dict)

    return sd



def keys_mapping_resnet(sd, model_dict):
    replace_dict = []

    for k, v in sd.items():
        if k not in model_dict and k.replace('module.', 'module.base_model.') in model_dict:
            # print('=> Load after add .base_model : ', k)
            replace_dict.append((k, k.replace('module.', 'module.base_model.')))  # (k_old, k_new)
        elif k.replace('module.', 'module.base_model.TCP.') in model_dict:
            # print('==> load after add .base_model.layer4  :', k)
            replace_dict.append((k, k.replace('module.', 'module.base_model.TCP.')))  # (k_old, k_new)
        else:
            print('skip ' + k)

    for k, k_new in replace_dict:
        sd[k_new] = sd.pop(k)

    return sd
