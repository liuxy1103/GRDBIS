import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from openTSNE import TSNE
# from functools import reduce
# import operator
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# show
def draw_fragments_2d(pred):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("the number of instance is %d" % size)
    color_pred = np.zeros([m, n, 3], dtype=np.uint8)
    idx = np.searchsorted(ids, pred)
    seed = [1,11,111]
    for i in range(3):
        np.random.seed(seed[i])
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    return color_pred

def embedding_pca(embeddings, n_components=3, as_rgb=True):
    if as_rgb and n_components != 3:
        raise ValueError("")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')
    return embed_flat

def show_raw_img(img):
    std = np.asarray([0.229, 0.224, 0.225])
    std = std[np.newaxis, np.newaxis, :]
    mean = np.asarray([0.485, 0.456, 0.406])
    mean = mean[np.newaxis, np.newaxis, :]
    if img.shape[0] == 3:
        img = np.transpose(img, (1,2,0))
    img = ((img * std + mean) * 255).astype(np.uint8)
    return img

def show_affs(iters, inputs, pred, target, cache_path, if_cuda=False):
    pred = pred[0].data.cpu().numpy()
    if if_cuda:
        inputs = inputs[0].data.cpu().numpy()
        target = target[0].data.cpu().numpy()
    else:
        inputs = inputs[0].numpy()
        target = target[0].numpy()
    inputs = show_raw_img(inputs)
    pred[pred<0]=0; pred[pred>1]=1
    target[target<0]=0; target[target>1]=1
    pred = (pred * 255).astype(np.uint8)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    target = (target * 255).astype(np.uint8)
    target = np.repeat(target[:,:,np.newaxis], 3, 2)
    im_cat = np.concatenate([inputs, pred, target], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def show_affs_emb(iters, inputs, ema_inputs, pred, target, emb1, emb2, cache_path, if_cuda=False):
    pred = pred[0].data.cpu().numpy()
    if if_cuda:
        inputs = inputs[0].data.cpu().numpy()
        ema_inputs = ema_inputs[0].data.cpu().numpy()
        target = target[0].data.cpu().numpy()
    else:
        inputs = inputs[0].numpy()
        ema_inputs = ema_inputs[0].numpy()
        target = target[0].numpy()
    emb1 = emb1[0].data.cpu().numpy()
    emb1 = embedding_pca(emb1)
    emb1 = np.transpose(emb1, (1,2,0))
    emb2 = emb2[0].data.cpu().numpy()
    emb2 = embedding_pca(emb2)
    emb2 = np.transpose(emb2, (1,2,0))
    inputs = show_raw_img(inputs)
    ema_inputs = show_raw_img(ema_inputs)
    pred[pred<0]=0; pred[pred>1]=1
    target[target<0]=0; target[target>1]=1
    pred = (pred * 255).astype(np.uint8)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    target = (target * 255).astype(np.uint8)
    target = np.repeat(target[:,:,np.newaxis], 3, 2)
    im_cat1 = np.concatenate([inputs, emb1, pred], axis=1)
    im_cat2 = np.concatenate([ema_inputs, emb2, target], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def val_show(iters, pred, target, pred_seg, gt_ins, valid_path):
    pred[pred<0]=0; pred[pred>1]=1
    target[target<0]=0; target[target>1]=1
    pred = pred[:,:,np.newaxis]
    pred = np.repeat(pred, 3, 2)
    target = target[:,:,np.newaxis]
    target = np.repeat(target, 3, 2)
    pred = (pred * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)
    im_cat1 = np.concatenate([pred, target], axis=1)
    seg_color = draw_fragments_2d(pred_seg)
    ins_color = draw_fragments_2d(gt_ins)
    im_cat2 = np.concatenate([seg_color, ins_color], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(valid_path, '%06d.png' % iters))

def val_show_emd(iters, pred, embedding, pred_seg, gt_ins, valid_path):
    pred[pred<0]=0; pred[pred>1]=1
    embedding = np.squeeze(embedding.data.cpu().numpy())
    embedding = embedding_pca(embedding)
    embedding = np.transpose(embedding, (1,2,0))
    pred = pred[:,:,np.newaxis]
    pred = np.repeat(pred, 3, 2)
    pred = (pred * 255).astype(np.uint8)
    im_cat1 = np.concatenate([pred, embedding], axis=1)
    seg_color = draw_fragments_2d(pred_seg)
    ins_color = draw_fragments_2d(gt_ins)
    im_cat2 = np.concatenate([seg_color, ins_color], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(valid_path, '%06d.png' % iters))

def val_show_tsne(iters, pred, embedding, pred_seg, gt_ins, valid_path,if_show_center=True):
    fg = gt_ins>0 # h,w
    fg = fg.reshape(-1, 1)
    embedding = embedding
    embedding = np.squeeze(embedding.data.cpu().numpy())
    _n_feats = embedding.shape[0]
    # fg_ins_embeddings = np.stack(
    #     [embedding[i][np.where(
    #         fg == True)]
    #      for i in range(_n_feats)], axis=1)
    # import pdb
    # pdb.set_trace()
    embedding = np.transpose(embedding, (1, 2, 0)) #h,w,c
    # seg_color = draw_fragments_2d(pred_seg)
    embed_dim = embedding.shape[-1]
    embed_flat = embedding.reshape(-1, embed_dim)
    ins_id = gt_ins.reshape(-1,1)# hxw,1
    if if_show_center:
        id_list = np.unique(ins_id)
        id_list = list(id_list)
        id_list.remove(0)
        emb_center = [np.mean(embed_flat[(ins_id == i)[:,0]],axis=0) for i in id_list]
        emb_center = np.array(emb_center)  #m,c
    ins_color = draw_fragments_2d(gt_ins)  # h,w,3
    ins_color_flat = ins_color.reshape(-1, 3)[fg[:,0]] #hxw,3  __get fg positions

    embed_flat = embed_flat[fg[:,0]] #hxw,c

    print(embed_flat.shape)
    seed = 21474836
    np.random.seed(seed)
    random_list = np.random.choice(range(ins_color_flat.shape[0]), size=int(embed_flat.shape[0]/2))
    print('random_list:',random_list)
    ins_color_flat = \
        ins_color_flat[random_list]

    embed_flat = \
        embed_flat[random_list]


    if if_show_center:
        embed_flat = np.concatenate((embed_flat,emb_center),axis=0) #(hxw+m),c
        color_tmp = np.ones((emb_center.shape[0],3))*0
        ins_color_flat = np.concatenate((ins_color_flat,color_tmp),axis=0)

    print(embed_flat.shape)

    print('start tsne')
    tsne = TSNE(n_components=2, random_state=0)

    tsne_emb =  tsne.fit_transform(embed_flat)  # 进行数据降维,并返回结果 #NxM

    print('start tsne iterations')
    ins_color_flat = np.concatenate((ins_color_flat,np.ones((tsne_emb.shape[0],1))*125),axis=1)

    size = np.ones((tsne_emb.shape[0],))
    if if_show_center:
        m = emb_center.shape[0]
        size[-m:] = 8
        ins_color_flat [-m:,3] = 255
    size = list(size)

    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1],s=size,c=ins_color_flat/255)

    # if if_show_center:
    #     tsne_center = tsne.fit_transform(emb_center)
    #     plt.scatter(tsne_center[:, 0], tsne_center[:, 1], s=5, c='k')

    plt.savefig(os.path.join(valid_path, '%06d' % iters +'_'+'tsne'+'.png'), dpi=1024)
    plt.cla()


# def val_show_tsne(iters, pred, embedding, pred_seg, gt_ins, valid_path):
#
#     embedding = np.squeeze(embedding.data.cpu().numpy())
#     embedding = embedding_pca(embedding)
#     embedding = np.transpose(embedding, (1, 2, 0)) #h,w,c
#     # seg_color = draw_fragments_2d(pred_seg)
#     embed_dim = embedding.shape[-1]
#     ins_color = draw_fragments_2d(gt_ins) #h,w,3
#     ins_color_flat = ins_color.reshape(-1, 3) #hxw,3
#     embed_flat = embedding.reshape(-1, embed_dim) #hxw,c
#     print('start tsne')
#     tsne = TSNE(
#         perplexity=30,
#         metric="euclidean",
#         n_jobs=-1,
#         random_state=42,
#         verbose=True,
#     )
#
#     tsne = tsne.fit(embed_flat)
#
#     # tsne = TSNE()
#     # tsne.fit_transform(embed_flat)  # 进行数据降维,并返回结果 #NxM
#     # tsne = pd.DataFrame(tsne.embedding_, index=embed_flat.index)  # 转换数据格式  #(hxw)x2
#     print('start tsne iterations:',tsne.shape[0])
#     # size = list(np.ones((tsne.shape[0],)))
#
#     ins_color_flat = np.concatenate((ins_color_flat,np.ones((tsne.shape[0],1))),axis=1)
#     plt.scatter(tsne[:, 0], tsne[:, 1],c=ins_color_flat/255)
#     # for i in range(tsne.shape[0]):
#     #     color = []
#     #     c_point = ins_color_flat[i]
#     #     color = color+list(c_point/255)+[1.0]
#     #     plt.scatter(tsne[i, 0], tsne[i, 1], c=color)
#     plt.savefig(os.path.join(valid_path, '%06d' % iters +'_'+'tsne'+'.png'))

def val_show_emd_layers(iters, pred, embedding_list, pred_seg='', gt_ins='', valid_path=''):
    pred[pred<0]=0; pred[pred>1]=1
    i = 0
    for embedding in embedding_list:
        embedding = np.squeeze(embedding.data.cpu().numpy())
        embedding = embedding_pca(embedding)
        embedding = np.transpose(embedding, (1,2,0))
        Image.fromarray(embedding).save(os.path.join(valid_path, '%06d'% iters+'emb_%06d.png'% i ))
        i = i+1