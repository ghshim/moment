import os
from os.path import join as pjoin

import torch

from models.mask_transformer.new_transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer

import utils.eval_t2m as eval_t2m
from utils.fixseed import fixseed
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

import numpy as np  

text_list = [
    'A person walks forward.',
    'A person walks slowly forward.',
    'A person walks briskly forward.',

    'A person is running on a treadmill.',
    'A person is running fast on a teadmill.',
    'A person is running slowly on a treadmill.',

    'A person picks something up from the floor.',
    'A person carefully picks something up from the floor.',
    'A person quickly picks something up from the floor.',

    'A person waves with their right hand.',
    'A person waves happily with their right hand.',
    'A person waves sadly with their right hand.',

    'A person sits down on the floor.',
    'A person sits down carefully on the floor.',
    'A person sits down sadly on the floor.',

    'A person turns to the right.',
    'A person turns quickly to the right.',
    'A person turns carefully to the right.'
]

token_list = [
    ['A/DET', 'person/NOUN', 'walks/Act_VIP', 'forward/Loc_VIP'],
    ['A/DET', 'person/NOUN', 'walks/Act_VIP', 'slowly/Desc_VIP', 'forward/Loc_VIP'],
    ['A/DET', 'person/NOUN', 'walks/Act_VIP', 'briskly/ADV', 'forward/Loc_VIP'],

    ['A/DET', 'person/NOUN', 'is/AUX', 'running/Act_VIP', 'on/ADP', 'a/DET', 'treadmill/NOUN'],
    ['A/DET', 'person/NOUN', 'is/AUX', 'running/Act_VIP', 'fast/Desc_VIP', 'on/ADP', 'a/DET', 'teadmill/NOUN'],
    ['A/DET', 'person/NOUN', 'is/AUX', 'running/Act_VIP', 'slowly/Desc_VIP', 'on/ADP', 'a/DET', 'treadmill/NOUN'],

    ['A/DET', 'person/NOUN', 'picks/Act_VIP', 'something/PRON', 'up/ADP', 'from/ADP', 'the/DET', 'floor/Obj_VIP'],
    ['A/DET', 'person/NOUN', 'carefully/Desc_VIP', 'picks/Act_VIP', 'something/PRON', 'up/ADP', 'from/ADP', 'the/DET', 'floor/Obj_VIP'],
    ['A/DET', 'person/NOUN', 'quickly/Desc_VIP', 'picks/Act_VIP', 'something/PRON', 'up/ADP', 'from/ADP', 'the/DET', 'floor/Obj_VIP'],

    ['A/DET', 'person/NOUN', 'waves/Act_VIP', 'with/ADP', 'their/PRON', 'right/Loc_VIP', 'hand/Body_VIP'],
    ['A/DET', 'person/NOUN', 'waves/Act_VIP', 'happily/Desc_VIP', 'with/ADP', 'their/PRON', 'right/Loc_VIP', 'hand/Body_VIP'],
    ['A/DET', 'person/NOUN', 'waves/Act_VIP', 'sadly/Desc_VIP', 'with/ADP', 'their/PRON', 'right/Loc_VIP', 'hand/Body_VIP'],

    ['A/DET', 'person/NOUN', 'sits/Act_VIP', 'down/Loc_VIP', 'on/ADP', 'the/DET', 'floor/Obj_VIP'],
    ['A/DET', 'person/NOUN', 'sits/Act_VIP', 'down/Loc_VIP', 'carefully/Desc_VIP', 'on/ADP', 'the/DET', 'floor/Obj_VIP'],
    ['A/DET', 'person/NOUN', 'sits/Act_VIP', 'down/Loc_VIP', 'sadly/Desc_VIP', 'on/ADP', 'the/DET', 'floor/Obj_VIP'],

    ['A/DET', 'person/NOUN', 'turns/Act_VIP', 'to/ADP', 'the/DET', 'right/Loc_VIP'],
    ['A/DET', 'person/NOUN', 'turns/Act_VIP', 'quickly/Desc_VIP', 'to/ADP', 'the/DET', 'right/Loc_VIP'],
    ['A/DET', 'person/NOUN', 'turns/Act_VIP', 'carefully/Desc_VIP', 'to/ADP', 'the/DET', 'right/Loc_VIP'],
]

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      pos_emb_dim=opt.pos_emb_dim,
                                      word_emb_dim=opt.word_emb_dim,
                                      text_mode=model_opt.text_mode,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    print(unexpected_keys)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            pos_emb_dim=opt.pos_emb_dim,
                                            word_emb_dim=opt.word_emb_dim,
                                            text_mode=res_opt.text_mode,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    print(unexpected_keys)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

@torch.no_grad()
def inference(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    max_length = torch.from_numpy(np.array([196,] * len(text_list))).cuda()

    anim_dir = os.path.join(out_dir, 'animation')
    joint_dir = os.path.join(out_dir, 'joint')
    os.makedirs(anim_dir, exist_ok=True)
    os.makedirs(joint_dir, exist_ok=True)

    for i, (clip_text, tokens) in enumerate(zip(text_list, token_list)):
        '''Make word embedding'''
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        pos_indices = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = w_vectorizer[token]
            pos_index = np.argmax(pos_oh)  # one-hot → int index
            pos_indices.append(pos_index)
            word_embeddings.append(word_emb[None, :])
        pos_indices = np.array(pos_indices) 
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        pos_indices = torch.from_numpy(pos_indices).long().cuda().unsqueeze(0)
        word_embeddings = torch.from_numpy(word_embeddings).float().cuda().unsqueeze(0)

        '''inference'''
        length = torch.from_numpy(np.array([196,])).cuda()
        mids = trans.generate(clip_text, length // 4, time_steps, cond_scale,
                                temperature=temperature, topk_filter_thres=topkr,
                                gsample=gsample, force_mask=force_mask,
                                sen_emb=None, word_emb=word_embeddings, pos=pos_indices)

        # motion_codes = motion_codes.permute(0, 2, 1)
        # mids.unsqueeze_(-1)
        pred_ids = res_model.generate(mids, clip_text, length // 4, temperature=1, cond_scale=res_cond_scale,
                                      sen_emb=None, word_emb=word_embeddings, pos=pos_indices)
        # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
        # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

        pred_motions = vq_model.forward_decoder(pred_ids)

    
        data = pred_motions.detach().cpu().numpy()
        lengths = max_length.cpu().numpy()
        save_path = os.path.join(anim_dir, f'{i:04d}.gif')
        
        data = val_loader.dataset.inv_transform(data)
        joint = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
        joint_path = os.path.join(joint_dir, f'{i:04d}.npy')
        np.save(joint_path, joint[0])

        plot_3d_motion(save_path, t2m_kinematic_chain, joint[0], title=clip_text, fps=20, radius=4)


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt)

    assert res_opt.vq_name == model_opt.vq_name

    
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    ### Evaluation ###
    out_dir = pjoin('./checkpoints/evaluation', f"{model_opt.name}_{res_opt.name}")
    os.makedirs(out_dir, exist_ok=True)

    # model_dir = pjoin(opt.)
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        t2m_transformer = load_trans_model(model_opt, file)
        t2m_transformer.eval()
        vq_model.eval()
        res_model.eval()

        t2m_transformer.to(opt.device)
        vq_model.to(opt.device)
        res_model.to(opt.device)

        with torch.no_grad():
            inference(eval_val_loader, vq_model, res_model, t2m_transformer,
                        0, eval_wrapper=eval_wrapper,
                        time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                        temperature=opt.temperature, topkr=opt.topkr,
                        force_mask=opt.force_mask, cal_mm=True)

# python eval_t2m_trans.py --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_vq --dataset_name t2m --gpu_id 3 --cond_scale 4 --time_steps 18 --temperature 1 --topkr 0.9 --gumbel_sample --ext cs4_ts18_tau1_topkr0.9_gs