import numpy as np
from aff2ins_util import dense_ind, get_aff, nonoverlap_aff, Downscale
from aff2ins_util import js_similar, gasp_similar
from aff2ins_util import logit_transform as log_trans
import cv2

def logit_transform(p):
    return log_trans(p, 0.5)

def gasp(pars_p, pwcon, pwidx, cls_num=1):
    from nifty.graph import UndirectedGraph
    from GASP.segmentation import run_GASP
  
    assert cls_num == 1
    length = 2
    
    pw_array    = np.asarray( - pwcon, dtype=np.float64).reshape(-1)
    pwidx_array = np.asarray(pwidx, dtype=np.uint64).reshape([-1, length])

    nb_nodes = int(pwidx_array.max() + 1)
    graph = UndirectedGraph(nb_nodes)
 
    graph.insertEdges(pwidx_array)
    unique_edge, cnt = np.unique(pwidx_array[:,0]*(pwidx_array.max()+1)+pwidx_array[:,1], return_counts=True)
    unique_edge_num = len(unique_edge)

    final_node_labels, _ = run_GASP(graph,
                                      pw_array,
                                      linkage_criteria='average',
                                      add_cannot_link_constraints=False,
                                      verbose=False,
                                      print_every=100)

    res = np.zeros([len(final_node_labels), 2], dtype=np.uint32)
    res[:, 1] = final_node_labels
    return res

def multicut(pars_p, pwcon, pwidx,cls_num=1):
    from multicut_cython.multicut import solve_nl_lmp
    length = 2 if cls_num == 1 else 4

    unary_array_solver = 1. * logit_transform(np.asarray(pars_p, dtype=np.float64).reshape([-1, cls_num])).copy(
        order='C')
    pw_array_solver = 1. * (np.asarray(pwcon, dtype=np.float64).reshape([-1, 1])).copy(order='C')
    pwidx_array = np.asarray(pwidx, dtype=np.uint16).reshape([-1, length]).copy(order='C')

    is_sparse_graph = True
    solver_type = False
    do_suppression = False
    logit_in_solver = False

    res = solve_nl_lmp(unary_array_solver, pwidx_array, pw_array_solver,
                       is_sparse_graph, solver_type, do_suppression, logit_in_solver)

    res = np.array(res, dtype=np.uint32)
    return res

class aff2ins(object):
    def __init__(self,
                 aff_pred_probs=[],
                 pars_pred_prob=None,
                 use_hvc=False,
                 aff_hvc=[],
                 obj_range=range(11, 19),
                 use_merge_pars=False,
                 merge_pars=None,
                 merge_obj_range=range(11, 19),
                 group_method='multicut'):
        assert len(aff_pred_probs) > 0
        self.aff_pred_probs = aff_pred_probs

        self.pars_pred_prob = pars_pred_prob
        max_h, max_w = self.pars_pred_prob.shape[len(self.pars_pred_prob.shape)-2:]

        self.stride = []
        for aff_pred_prob in self.aff_pred_probs:
            stride = int(max_h / aff_pred_prob.shape[1])
            pars_pred_prob = Downscale(pars_pred_prob, stride)
            self.stride.append(stride)

        self.use_merge_pars = use_merge_pars
        self.merge_pars1 = None
        self.merge_pars = None
        if use_merge_pars:
            assert merge_pars is not None
            assert merge_pars.shape[1] == max_h
            self.merge_pars = merge_pars
            self.merge_pars1 = Downscale(merge_pars, self.stride[0])
            self.merge_obj_range = merge_obj_range

        self.cls_num = self.pars_pred_prob.shape[0]

        self.obj_range = obj_range
        self.getcon=[]

        assert group_method in ['multicut', 'gasp']
        self.group_method = group_method

    def separate_objects(self, pwidx, pwcon, pars_p, hyperindex=None,use_merge_pars=False):
        assert len(pars_p.shape) == 3
        h=pars_p.shape[1]
        w=pars_p.shape[2]
        hyper_index = np.asarray(range(h*w)).reshape([h,w]) if hyperindex is None else hyperindex

        cls_num = pars_p.shape[0]
        pars_p_r=pars_p.argmax(axis=0)
        pwidx_cls = []
        pwcon_cls = []
        where_cls = []
        pars_cls = []
        if use_merge_pars:
            obj_range=self.merge_obj_range
        else:
            obj_range=self.obj_range
        clses, cls_nums = np.unique(pars_p_r, return_counts=True)
        for cls in clses[cls_nums>8]:
            if cls in obj_range:
                where_cls_s = np.where(pars_p_r==cls)
                hyper_where_cls_s, indices = np.unique(hyper_index[where_cls_s], return_index=True)
                pars_cls_s = pars_p[cls][where_cls_s][indices]
                where_hyper_node_cls_s = np.where(pwidx[:, 2] == cls)
                pwidx_cls_s = pwidx[where_hyper_node_cls_s, :2]
                pwcon_cls_s = pwcon[where_hyper_node_cls_s]

                pwidx_cls_s = np.asarray(pwidx_cls_s)
                j = 0
                pwidx_cls_s_t = pwidx_cls_s.copy()
                for ind in hyper_where_cls_s:
                    pwidx_cls_s[pwidx_cls_s_t == ind] = j
                    j += 1

                if len(pwcon_cls_s) > 0:
                    where_cls.append(hyper_where_cls_s)
                    pars_cls.append(pars_cls_s)
                    pwidx_cls.append(pwidx_cls_s)
                    pwcon_cls.append(pwcon_cls_s)
        ins_result = pars_p_r.copy()

        for unary_array, pwc, pwi, inds in zip(pars_cls, pwcon_cls, pwidx_cls, where_cls):
            if self.group_method == 'multicut':
                res = multicut(unary_array, pwc, pwi)
            else: 
                res = gasp(unary_array, pwc, pwi)

            res = np.array(res, dtype=np.uint32)

            ress = res[:, 1] + np.max(ins_result)

            for ind, res in zip(inds, ress):
                ins_result[hyper_index == ind] = res

        ins_r = dense_ind(ins_result)
        return ins_r.reshape([h, -1])

    def combine_nodes(self, ins_result, parsing_prob, obj_only=True, kernal_size=[3,3]):
            h = parsing_prob.shape[1]
            w = parsing_prob.shape[2]
            upscale = int((h*w/(ins_result.shape[0]*ins_result.shape[1]))**0.5)
            hyper_index = np.asarray(range(h * w)).reshape([h, -1])
            pars_pred = parsing_prob.argmax(axis=0)
            pars_prob_down = Downscale(parsing_prob,upscale).transpose(1, 2, 0)
            pars_pred_down = Downscale(pars_pred,upscale)
            combined_pars_pred_prob = parsing_prob.copy().transpose(1, 2, 0)

            kernel = np.ones((kernal_size[0], kernal_size[0]), dtype=np.uint8)
            kernel_2 = np.ones((kernal_size[1], kernal_size[1]), dtype=np.uint8)

            if obj_only:
                obj = {_ : 0 for _ in range(20)}
                for _ in self.obj_range:
                    obj[_] = 1
                obj_aff = np.vectorize(obj.get)(pars_pred)


            hyper_indexs=[]
            ins_ids, ins_nums = np.unique(ins_result, return_counts=True)
            for ins in ins_ids[ins_nums > kernal_size[0]**2 - 1]:
                if not (obj_only and not (pars_prob_down[ins_result == ins].mean(axis=0).argmax() in self.obj_range)):
                    inner_region = np.zeros_like(ins_result, dtype=np.uint8)

                    inner_region[ins_result == ins] = 1

                    inner_region = cv2.erode(inner_region, kernel)
                    if inner_region.sum() > 0:
                        inner_region_upsample = cv2.resize(inner_region, (0, 0), fx=upscale, fy=upscale,
                                                         interpolation=cv2.INTER_NEAREST)

                        if obj_only:
                            inner_region_upsample = inner_region_upsample * obj_aff
                        if inner_region_upsample.sum() > 0:
                            mean_par_p = parsing_prob.transpose(1, 2, 0)[inner_region_upsample == 1].mean(axis=0)

                            cls_id = mean_par_p.argmax()
                            if not (obj_only and not (cls_id in self.obj_range)):
                                new_hyper_ind = hyper_index.reshape(-1).max() + 1
                                hyper_index[inner_region_upsample == 1] = new_hyper_ind
                                hyper_indexs.append([new_hyper_ind, cls_id, int(inner_region_upsample.sum())])

                                combined_pars_pred_prob[inner_region_upsample == 1] = mean_par_p / mean_par_p.sum()
            hyper_index_dense = dense_ind(hyper_index,stay_shape=True)
            new_hyper_indexs = [[hyper_index_dense[np.where(hyper_index == hi[0])][0], hi[1], hi[2]] for hi in hyper_indexs]

            return hyper_index_dense, combined_pars_pred_prob.transpose(2, 0, 1), new_hyper_indexs

    def combine_getcon_seperate(self, pars_pred_prob = None,
                                maskp=[],
                                combine=True, kernal_size=[3,3], ins_r=None,
                                merge_pars=None, use_merge_pars=False):
        hyper_index = None

        if pars_pred_prob is None:
            stride = int(self.pars_pred_prob.shape[1] / maskp[0].shape[1])
            pars_pred_prob = Downscale(self.pars_pred_prob, stride)

        combined_pars = pars_pred_prob

        hyper_indexs = []
        if combine:
            if not use_merge_pars:
                hyper_index, combined_pars, hyper_indexs = self.combine_nodes(ins_r, pars_pred_prob,
                                                                           obj_only=True,kernal_size=kernal_size)
            else:
                assert merge_pars is not None
                hyper_index, combined_pars, hyper_indexs = self.combine_nodes(ins_r, merge_pars,
                                                                              obj_only=True, kernal_size=kernal_size)

        getcon = GetPWCon(aff_pred_probs=maskp,
                        pars_pred=combined_pars.argmax(axis=0),
                        hyper_index=hyper_index,
                        hyper_set=hyper_indexs,
                        use_merge_pars=use_merge_pars,
                        fully_pars=pars_pred_prob,
                        obj_range=self.obj_range if not use_merge_pars else self.merge_obj_range,
                        group_method=self.group_method)
        pwidx, pwcon = getcon.get_pairwise_conn()

        ins_r = self.separate_objects(pwidx, pwcon, combined_pars, hyper_index, use_merge_pars=use_merge_pars)
        return hyper_index, ins_r, combined_pars

    def make_and_solve_hierarchic(self, cascade_num=3):
        assert not (self.group_method == 'gasp' and cascade_num>1), "cascade not support gasp"
        flag = True

        for ind in range(1, len(self.aff_pred_probs) + 1):
            last_one = ind == len(self.aff_pred_probs)
            
            if len(self.aff_pred_probs) - ind < cascade_num:
                if flag:
                    hyper_ind, ins_r, pars = self.combine_getcon_seperate(maskp=self.aff_pred_probs[-ind:],
                                                                          use_merge_pars=self.use_merge_pars and last_one,
                                                                          merge_pars=self.merge_pars,
                                                                          combine=False)
                    flag = False
                else:
                    hyper_ind, ins_r, pars = self.combine_getcon_seperate(maskp=self.aff_pred_probs[-ind:],
                                           use_merge_pars=self.use_merge_pars and last_one,
                                           merge_pars=self.merge_pars,
                                           combine=True, ins_r=ins_r)
        return ins_r, pars.argmax(axis=0)

class GetPWCon:
    def __init__(self,
                 aff_pred_probs=[],
                 hyper_index=None,
                 hyper_set=[],
                 pars_pred=None,
                 use_merge_pars=False,
                 fully_pars=None,
                 obj_range=range(11, 19),
                 group_method='multicut'):
        assert len(aff_pred_probs) > 0
        self.aff_pred_probs = aff_pred_probs
        self.pars_result = pars_pred
        self.max_h, self.max_w = pars_pred.shape[len(pars_pred.shape)-2:]

        self.hyper_index = np.asarray(range(self.max_h*self.max_w)).reshape([self.max_h,self.max_w]) if hyper_index is None else hyper_index
        self.hyper_set = hyper_set

        self.use_merge_pars=use_merge_pars
        if use_merge_pars:
            assert len(fully_pars.shape)==3, fully_pars.shape
            assert fully_pars.shape[1]==self.max_h, '{} vs. {}'.format(fully_pars.shape[1], self.max_h)
            self.fully_pars=fully_pars
            self.group_method = group_method

        if self.pars_result is not None and len(self.pars_result.shape) == 2:
            obj = {_:0 for _ in range(20)}
            for _ in obj_range:
                obj[_]=1
            self.where_obj=np.vectorize(obj.get)(pars_pred)
        else:
            self.where_obj = np.ones([self.max_h,self.max_w])

        self.aff_size = []
        self.stride = []
        for aff_pred_prob in self.aff_pred_probs:
            stride = self.max_h / aff_pred_prob.shape[1]
            self.stride.append(stride)
            assert stride > 0
            aff_size = int(aff_pred_prob.shape[0] ** 0.5)
            self.aff_size.append(aff_size)

        self.obj_range = obj_range

    def get_pairwise_conn(self):
        self.num_pix = (self.where_obj==1).sum()

        idxcon=[]
        hyper_con_num = len(idxcon)

        for ind in range(len(self.aff_pred_probs)):
            overlap = 0 if ind==0 else int((self.aff_size[ind-1]//2)//(self.stride[ind]/self.stride[ind-1]))
            idxcon.extend(self.get_conn_singlescale(ind, overlap))

        idxcon = np.asarray(idxcon)
        pwidx_array = np.asarray([], dtype=np.uint32)
        pwcon_array = np.asarray([], dtype=np.float64)

        if len(idxcon)>0:
             pwidx_array = idxcon[:, :-1].astype(np.uint32)
             pwcon_array = idxcon[:, -1].astype(np.float64)
        return [pwidx_array, pwcon_array]


    def get_conn_singlescale(self, ind, overlap=0):
        aff_pred = self.aff_pred_probs[ind]
        stride = self.stride[ind]
        aff_size = self.aff_size[ind]
        
        pars_r = Downscale(self.pars_result, stride)
        where_obj = Downscale(self.where_obj, stride)
        hyper_index = Downscale(self.hyper_index, stride).astype(np.int64)
        hyper_index[where_obj == 0] = -1

        base_idx = np.tile(hyper_index, \
                            (aff_size ** 2 - (overlap*2+1)**2, 1, 1)).transpose(1, 2, 0)
        idx = get_aff(hyper_index, aff_size, overlap).transpose(1, 2, 0)
	
        con = aff_pred[nonoverlap_aff(aff_size, overlap)].transpose(1,2,0)

        pars_base = np.tile(pars_r, \
                            (aff_size ** 2 - (overlap*2+1)**2, 1, 1)).transpose(1, 2, 0)
        pars_aff = get_aff(pars_r, aff_size, overlap).transpose(1, 2, 0)
        if self.use_merge_pars:
            assert len(self.fully_pars.shape) == 3
            fully_pars = Downscale(self.fully_pars[self.obj_range], stride)
            cls_num = fully_pars.shape[0]
            pars_prob_base = np.tile(fully_pars.transpose(1,2,0), \
                                     (aff_size ** 2- (overlap*2+1)**2, 1, 1, 1)).transpose(1, 2, 0, 3)
            pars_prob_aff = get_aff(fully_pars.transpose(1,2,0), aff_size, overlap).transpose(1, 2, 0, 3)

        base_idx[base_idx == idx] = -1
        base_idx[idx < 0] = -1
        base_idx[pars_base != pars_aff] = -1

        if (base_idx >= 0).sum() > 0:
            where_valid = np.where(base_idx >= 0)
            base_idx = base_idx[where_valid]
            idx = idx[where_valid]
            pars_base = pars_base[where_valid]
            pars_aff = pars_aff[where_valid]
            con = con[where_valid]
            if self.use_merge_pars:
                pars_prob_base = pars_prob_base[where_valid]
                pars_prob_aff = pars_prob_aff[where_valid]

            dir0 = np.where(base_idx == idx)[0]

            # compute average affinity, a'(u,v) = (a(u,v) + a(v,u)) / 2.
            dir1 = np.where(base_idx < idx)
            dir2 = np.where(base_idx > idx)

            con_dir1 = con[dir1]
            base_idx_dir1 = base_idx[dir1]
            idx_dir1 = idx[dir1]
            pars_base_dir1 = pars_base[dir1]
            pars_aff_dir1 = pars_aff[dir1]
            if self.use_merge_pars:
                pars_prob_base_dir1 = pars_prob_base[dir1]
                pars_prob_aff_dir1 = pars_prob_aff[dir1]

            con_dir2 = con[dir2]
            base_idx_dir2 = base_idx[dir2]
            idx_dir2 = idx[dir2]

            sort1 = np.lexsort((base_idx_dir1, idx_dir1))
            sort2 = np.lexsort((idx_dir2, base_idx_dir2))
            con_1 = con_dir1[sort1]
            base_idx = base_idx_dir1[sort1]
            idx = idx_dir1[sort1]
            pars_base = pars_base_dir1[sort1]
            pars_aff = pars_aff_dir1[sort1]
            if self.use_merge_pars:
                pars_prob_base = pars_prob_base_dir1[sort1]
                pars_prob_aff = pars_prob_aff_dir1[sort1]

            con_2 = con_dir2[sort2]
            base_idx_2 = base_idx_dir2[sort2]
            idx_2 = idx_dir2[sort2]

            con = (con_1 + con_2) / 2.

            if self.use_merge_pars:
                if self.group_method == 'multicut':
                    similar = js_similar(pars_prob_base, pars_prob_aff)
                else:
                    similar = gasp_similar(pars_prob_base, pars_prob_aff)

                cls_base = pars_prob_base.argmax(axis=1)
                cls_aff = pars_prob_aff.argmax(axis=1)
                find_cls = cls_base * 200 + cls_aff
                con_t = con.copy()
                con *= similar

            con = logit_transform(con)

            idxparcon_full = np.asarray(
                [base_idx, idx, pars_base, pars_aff, con]).transpose(1, 0).reshape([-1, 5])
            return idxparcon_full.tolist()

        else:
            return []

def draw_fragments_2d(pred):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("the number of instance is %d" % size)
    color_pred = np.zeros([m, n, 3], dtype=np.uint8)
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    return color_pred

if __name__ == "__main__":
    from PIL import Image
    from gen_affs import Affinity_generator_new
    output_dir = './'

    # parsing_pred_prob = np.load(output_dir + './demo_outputs/output1.npy')
    # aff_pred_prob_down4  = np.load(output_dir + './demo_outputs/output2.npy')
    # aff_pred_prob_down8  = np.load(output_dir + './demo_outputs/output3.npy')
    # aff_pred_prob_down16 = np.load(output_dir + './demo_outputs/output4.npy')
    # aff_pred_prob_down32 = np.load(output_dir + './demo_outputs/output5.npy')
    # aff_pred_prob_down64 = np.load(output_dir + './demo_outputs/output6.npy')

    img = np.asarray(Image.open('./plant015_label.png'))
    img_padding = np.zeros((512, 512), dtype=np.uint8)
    img_padding[:, 6:-6] = img[9:-9, :]
    img_padding_3 = img_padding[:,:,np.newaxis]
    img_padding_3 = np.repeat(img_padding_3, 3, 2)

    affs = Affinity_generator_new(img_padding_3)
    print(affs.shape)

    mask = np.zeros((2, 512, 512), dtype=np.uint8)
    mask[0] = img_padding == 0
    mask[1] = img_padding != 0

    parsing_pred_prob = mask
    aff_pred_prob_down4 = affs[0, :, :256, :256]
    aff_pred_prob_down8 = affs[1, :, :128, :128]
    aff_pred_prob_down16 = affs[2, :, :64, :64]
    aff_pred_prob_down32 = affs[3, :, :32, :32]
    aff_pred_prob_down64 = affs[4, :, :16, :16]

    # img1 = parsing_pred_prob[0]
    # img1 = (img1 * 255).astype(np.uint8)
    # Image.fromarray(img1).save('./img1.png')
    # img2 = parsing_pred_prob[-1]
    # img2 = (img2 * 255).astype(np.uint8)
    # Image.fromarray(img2).save('./img2.png')

    # img_all = aff_pred_prob_down8[0]
    # for i in range(24):
    #     img_all = np.concatenate([img_all, aff_pred_prob_down8[i+1]], axis=0)
    # img_all = (img_all * 255).astype(np.uint8)
    # Image.fromarray(img_all).save('./img_all.png')

    aff_pred_prob_downs = [aff_pred_prob_down4,
                            aff_pred_prob_down8,
                            aff_pred_prob_down16,
                            aff_pred_prob_down32,
                            aff_pred_prob_down64]

    # parsing_pred_prob_down4 = parsing_pred_prob[:, 2::4, 2::4]
    # parsing_pred_prob_down8 = parsing_pred_prob[:, 4::8, 4::8]
    parsing_pred_prob_down4 = parsing_pred_prob[:, 1::2, 1::2]
    parsing_pred_prob_down8 = parsing_pred_prob[:, 2::4, 2::4]

    parsing_pred_prob_merge = np.vstack((parsing_pred_prob[:1],
                                           np.sum(parsing_pred_prob[1:],axis=0,keepdims=True)))
    # parsing_pred_prob_merge_down4 = parsing_pred_prob_merge[:, 2::4, 2::4]
    # parsing_pred_prob_merge_down8 = parsing_pred_prob_merge[:, 4::8, 4::8]
    parsing_pred_prob_merge_down4 = parsing_pred_prob_merge[:, 1::2, 1::2]
    parsing_pred_prob_merge_down8 = parsing_pred_prob_merge[:, 2::4, 2::4]

    run_cgp = aff2ins(
                    aff_pred_prob_downs,
                    parsing_pred_prob_down4,
                    use_merge_pars=True,
                    merge_pars=parsing_pred_prob_merge_down4,
                    obj_range=range(1,2),  # range(1,9)
                    merge_obj_range=range(1,2),  # range(1,9)
                    group_method='multicut')  # multicut, gasp
    ins_r, pars_r = run_cgp.make_and_solve_hierarchic(cascade_num=1)
    # ins_r = ((255/ins_r.max())*ins_r).astype(np.uint8)
    ins_r = draw_fragments_2d(ins_r)
    Image.fromarray(ins_r).save(output_dir + './cvppp.png')
