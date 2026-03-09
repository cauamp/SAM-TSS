import torch
from datasets.helpers import classes_weights, DATASETS_CLASSES_DICT, DATASETS_NUM_CLASSES

import torch.nn.functional as F


class TrainingLoss(torch.nn.Module):

    def __init__(self, args, device):
        super().__init__()

        weights = classes_weights(DATASETS_CLASSES_DICT[args.dataset], args.shallow_dec, device)

        self.win_size = args.win_size
        self.always_decode = args.always_decode

        if args.dataset == "MVSeg":
            #n_losses = 1
            n_losses = 4
            self.last_frame_loss = True
            self.idx_start = self.win_size - 1
        else:
            n_losses = -1
            print("unsupported dataset")
            exit(1)

        self.loss_idx_step = 1
        if args.model_struct == "aux_mem_loss":
            n_losses *= 2
            self.loss_idx_step = 2

        self.losses = [torch.nn.NLLLoss(weights) for i in range(n_losses)]

        assert(len(self.losses) == n_losses)
        assert(self.idx_start < self.win_size)


    def gen_prototypes(self, feat, label, cls_num):
        n, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)                 # B, H*W, N
        label = label.permute(0, 1, 2).contiguous().view(n, h * w)                      # B, H*W

        print(f"[DEBUG gen_prototypes] feat has NaN: {torch.isnan(feat).any()}")
        print(f"[DEBUG gen_prototypes] feat shape: {feat.shape}, min: {feat.min()}, max: {feat.max()}")

        prototypes_batch = []
        for i in range(n):
            # classes = torch.unique(label[i, :].clone().detach())
            classes = list(range(cls_num))
            prototypes = []
            for c in classes:
                prototype = feat[label == c, :]
                temp = prototype.detach()
                if torch.equal(temp.cpu(), torch.zeros(prototype.shape)):
                    prototype = prototype.sum(0, keepdims=True)
                else:
                    prototype = prototype.mean(0, keepdims=True)
                if torch.isnan(prototype).any():
                    print(f"[DEBUG gen_prototypes] NaN in prototype for class {c}, batch {i}")
                    print(f"[DEBUG gen_prototypes] prototype shape: {prototype.shape}, num pixels: {(label == c).sum()}")
                prototypes.append(prototype)
            prototypes = torch.cat(prototypes, dim=0)
            prototypes = prototypes.permute(1, 0).contiguous()
            prototypes_batch.append(prototypes)
        prototypes = torch.stack(prototypes_batch)
        print(f"[DEBUG gen_prototypes] output has NaN: {torch.isnan(prototypes).any()}")
        return prototypes  # [batch_size, N_channels, Classes]

    def metric_learning(self,curr_R_fea, mem_feats, labels):
        batch_size = curr_R_fea.shape[0]
        labels = labels.contiguous().view(batch_size, -1)  # B_s, H*W
        curr_R_fea = curr_R_fea.permute(0, 2, 3, 1)
        curr_R_fea = curr_R_fea.contiguous().view(curr_R_fea.shape[0], -1, curr_R_fea.shape[-1])  # B_s, H*W, D

        print(f"[DEBUG metric_learning] curr_R_fea has NaN: {torch.isnan(curr_R_fea).any()}")

        [mem_R_feas, mem_T_feas, mem_F_feas] = mem_feats
        print(f"[DEBUG metric_learning] mem_R_feas has NaN: {torch.isnan(mem_R_feas).any()}")
        print(f"[DEBUG metric_learning] mem_T_feas has NaN: {torch.isnan(mem_T_feas).any()}")
        print(f"[DEBUG metric_learning] mem_F_feas has NaN: {torch.isnan(mem_F_feas).any()}")

        temperature = 0.1
        loss_batch = []
        for i in range(batch_size):
            label_i = labels[i, ...]  # HW

            mem_R_feas_i = mem_R_feas[i, ...]  # D, T, C
            mem_T_feas_i = mem_T_feas[i, ...]  # D, T, C
            mem_F_feas_i = mem_F_feas[i, ...]  # D, T, C

            multimodal_mem_i = torch.cat([mem_R_feas_i, mem_T_feas_i, mem_F_feas_i], dim=1).permute(1, 2,
                                                                                                    0)  # 3T, C,  D
            T3, C, N = multimodal_mem_i.size()
            multimodal_mem_i_view = multimodal_mem_i.contiguous().view(T3 * C, N)  # 3T*C, D

            anchor_fea_R_i = curr_R_fea[i, ...]  # HW, D

            '''Takcing anchor R features as an example'''
            # anchor_fea_R_i： （HW, N）；  multimodal_mem_i：（3TC, N）
            # memory label
            y_contrast = torch.zeros((T3 * C, 1)).float().cuda()         # 3TC, 1
            sample_ptr = 0
            for ii in range(C):
                # if ii == 0: continue
                y_contrast[sample_ptr:sample_ptr + T3, ...] = ii
                sample_ptr += T3
            contrast_feature = F.normalize(multimodal_mem_i_view, p=2, dim=1)  # 3TC, D
            print(f"[DEBUG metric_learning] batch {i}: contrast_feature has NaN: {torch.isnan(contrast_feature).any()}")

            # valid_mask = torch.norm(contrast_feature, p=1, dim=1).cuda()   # 3TC
            # valid_mask = torch.where(valid_mask > 0, torch.tensor(1).cuda() , torch.tensor(0).cuda()).contiguous().view(-1, 1)

            y_anchor = label_i.contiguous().view(-1, 1)  # HW,  1
            anchor_feature = F.normalize(anchor_fea_R_i, p=2, dim=1)  # HW,  D
            print(f"[DEBUG metric_learning] batch {i}: anchor_feature has NaN: {torch.isnan(anchor_feature).any()}")

            mask = torch.eq(y_anchor, y_contrast.T).float().cuda()             # HW, 3TC
            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                            temperature)  # HW, 3TC
            print(f"[DEBUG metric_learning] batch {i}: anchor_dot_contrast has NaN: {torch.isnan(anchor_dot_contrast).any()}")
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()  # HW, 3TC
            print(f"[DEBUG metric_learning] batch {i}: logits has NaN: {torch.isnan(logits).any()}")

            neg_mask = 1 - mask
            neg_mask = neg_mask #* valid_mask.T
            mask = mask #* valid_mask.T

            neg_logits = torch.exp(logits) * neg_mask
            neg_logits = neg_logits.sum(1, keepdim=True)

            exp_logits = torch.exp(logits)

            log_prob = logits - torch.log(exp_logits + neg_logits)
            print(f"[DEBUG metric_learning] batch {i}: log_prob has NaN: {torch.isnan(log_prob).any()}")
            if torch.isnan(log_prob).any():
                print(f"[DEBUG metric_learning] batch {i}: exp_logits has NaN: {torch.isnan(exp_logits).any()}")
                print(f"[DEBUG metric_learning] batch {i}: neg_logits has NaN: {torch.isnan(neg_logits).any()}")
                print(f"[DEBUG metric_learning] batch {i}: neg_logits min/max: {neg_logits.min()}/{neg_logits.max()}")

            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            print(f"[DEBUG metric_learning] batch {i}: mean_log_prob_pos has NaN: {torch.isnan(mean_log_prob_pos).any()}")
            print(f"[DEBUG metric_learning] batch {i}: mask.sum(1) min: {mask.sum(1).min()}")
            loss = - mean_log_prob_pos
            loss = loss.mean()
            print(f"[DEBUG metric_learning] batch {i}: loss value: {loss.item()}")
            loss_batch.append(loss)


        loss_R = sum(loss_batch) / len(loss_batch)
        print(f"[DEBUG metric_learning] final loss_R: {loss_R.item()}, has NaN: {torch.isnan(loss_R).any()}")
        return loss_R


    def contrastive_loss(self, Current_fea, Previous_mem, Label):
        [curr_R_fea, curr_T_fea, curr_F_fea] = Current_fea      # Each Dim is: [B, N, H, W]
        [mem_R_feas, mem_T_feas, mem_F_feas] = Previous_mem     # Each Dim is: [B, N, T, C]

        labels = Label.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (curr_R_fea.shape[2], curr_R_fea.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()                       # Each Dim is: [B, H, W]

        # Update Memory
        curr_R_proto, curr_T_proto, curr_F_proto = \
            self.gen_prototypes(curr_R_fea, labels, DATASETS_NUM_CLASSES["MVSeg"]).unsqueeze(2), \
            self.gen_prototypes(curr_T_fea, labels, DATASETS_NUM_CLASSES["MVSeg"]).unsqueeze(2),\
            self.gen_prototypes(curr_F_fea, labels, DATASETS_NUM_CLASSES["MVSeg"]).unsqueeze(2)
        mem_R_feas, mem_T_feas, mem_F_feas = torch.cat([mem_R_feas, curr_R_proto], dim=2), \
                                             torch.cat([mem_T_feas, curr_T_proto], dim=2), \
                                             torch.cat([mem_F_feas, curr_F_proto], dim=2)
        mem_feats_update = [mem_R_feas, mem_T_feas, mem_F_feas]

        print("[DEBUG contrastive_loss] Computing lossR...")
        lossR = self.metric_learning(curr_R_fea, mem_feats_update, labels)
        print("[DEBUG contrastive_loss] Computing lossT...")
        lossT = self.metric_learning(curr_T_fea, mem_feats_update, labels)
        print("[DEBUG contrastive_loss] Computing lossF...")
        lossF = self.metric_learning(curr_F_fea, mem_feats_update, labels)

        loss_metric = (lossR + lossT + lossF)/3.0
        print(f"[DEBUG contrastive_loss] loss_metric: {loss_metric.item()}, has NaN: {torch.isnan(loss_metric).any()}")
        return loss_metric


    def forward(self, probs, labels, probs_aux_rgb, probs_thermal, probs_fusion, total_feas):
        # Probs:    torch.Size([2, 1, 26, 480, 640])
        # Labels:   torch.Size([2, 1, 1, 480, 640])

        if self.always_decode:
            assert(probs.size(1) == self.win_size)
        else:
            assert(probs.size(1) == 1)

        losses = []
        softmax_dim = 1

        # win_size = 4, idx_start = 3.

        for t in range(self.idx_start, self.win_size, self.loss_idx_step):
            label_idx = 0 if self.last_frame_loss else t
            probs_idx = 0 if not self.always_decode else t
            
            # Check main probs
            main_probs = probs[:,probs_idx,:,:,:]
            print(f"[DEBUG forward] t={t}: main_probs has NaN: {torch.isnan(main_probs).any()}, min: {main_probs.min().item()}, max: {main_probs.max().item()}")
            main_log_softmax = F.log_softmax(main_probs, dim=softmax_dim)
            print(f"[DEBUG forward] t={t}: main_log_softmax has NaN: {torch.isnan(main_log_softmax).any()}")
            main_loss = self.losses[t - self.idx_start](main_log_softmax, labels[:,label_idx,0,:,:])
            print(f"[DEBUG forward] t={t}: main_loss: {main_loss.item()}, has NaN: {torch.isnan(main_loss).any()}")
            losses.append(main_loss)

            if probs_aux_rgb is not None:
                aux_probs = probs_aux_rgb[:,probs_idx,:,:,:]
                print(f"[DEBUG forward] t={t}: aux_rgb_probs has NaN: {torch.isnan(aux_probs).any()}, min: {aux_probs.min().item()}, max: {aux_probs.max().item()}")
                aux_log_softmax = F.log_softmax(aux_probs, dim=softmax_dim)
                print(f"[DEBUG forward] t={t}: aux_rgb_log_softmax has NaN: {torch.isnan(aux_log_softmax).any()}")
                aux_loss = self.losses[t - self.idx_start + 1](aux_log_softmax, labels[:,label_idx,0,:,:])
                print(f"[DEBUG forward] t={t}: aux_rgb_loss: {aux_loss.item()}, has NaN: {torch.isnan(aux_loss).any()}")
                losses.append(aux_loss)
                
            if probs_thermal is not None:
                thermal_probs = probs_thermal[:,probs_idx,:,:,:]
                print(f"[DEBUG forward] t={t}: thermal_probs has NaN: {torch.isnan(thermal_probs).any()}, min: {thermal_probs.min().item()}, max: {thermal_probs.max().item()}")
                thermal_log_softmax = F.log_softmax(thermal_probs, dim=softmax_dim)
                print(f"[DEBUG forward] t={t}: thermal_log_softmax has NaN: {torch.isnan(thermal_log_softmax).any()}")
                thermal_loss = self.losses[t - self.idx_start + 2](thermal_log_softmax, labels[:,label_idx,0,:,:])
                print(f"[DEBUG forward] t={t}: thermal_loss: {thermal_loss.item()}, has NaN: {torch.isnan(thermal_loss).any()}")
                losses.append(thermal_loss)
                
            if probs_fusion is not None:
                fusion_probs = probs_fusion[:,probs_idx,:,:,:]
                print(f"[DEBUG forward] t={t}: fusion_probs has NaN: {torch.isnan(fusion_probs).any()}, min: {fusion_probs.min().item()}, max: {fusion_probs.max().item()}")
                fusion_log_softmax = F.log_softmax(fusion_probs, dim=softmax_dim)
                print(f"[DEBUG forward] t={t}: fusion_log_softmax has NaN: {torch.isnan(fusion_log_softmax).any()}")
                fusion_loss = self.losses[t - self.idx_start + 3](fusion_log_softmax, labels[:,label_idx,0,:,:])
                print(f"[DEBUG forward] t={t}: fusion_loss: {fusion_loss.item()}, has NaN: {torch.isnan(fusion_loss).any()}")
                losses.append(fusion_loss)

        print(f"[DEBUG forward] Individual losses: {[l.item() for l in losses]}")
        loss_ce = sum(losses) / len(losses)
        print(f"[DEBUG forward] loss_ce: {loss_ce.item()}, has NaN: {torch.isnan(loss_ce).any()}")

        if total_feas is None:
            return loss_ce
        else:
            # Calculating the MVRegulator Loss
            [Current_fea, Previous_mem] = total_feas
            label_index = 0 if self.last_frame_loss else exit(1)
            Curr_Target = labels[:,label_index, 0,:,:]              # labels: [B, T=1, 1, H, W]

            print("[DEBUG forward] Computing contrastive loss...")
            loss_metirc = self.contrastive_loss(Current_fea, Previous_mem, Curr_Target)
            print(f"[DEBUG forward] loss_metric: {loss_metirc.item()}")
            final_loss = loss_ce + 0.001 * loss_metirc
            print(f"[DEBUG forward] final_loss: {final_loss.item()}, has NaN: {torch.isnan(final_loss).any()}")
            return final_loss
