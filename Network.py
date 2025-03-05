# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Backbones.SAM import sam_model_registry
from Backbones.DINO.vit import Block, trunc_normal_
from Backbones.DINO.utils import neq_load_external_vit
from Backbones.DINO.vit import vit_small, vit_base, vit_large, trunc_normal_
#from Backbones.dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from Backbones.DINO.align_model import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from functools import partial

def norm(t):
    return F.normalize(t, dim=-1, eps=1e-10)

class Found_Feature_Adaptor(nn.Module):
    def __init__(self, embedding_dim,feat_adaptor_depth,num_heads=6): #embedding_size:
        super(Found_Feature_Adaptor, self).__init__()

        self.blocks = nn.ModuleList([Block(dim=embedding_dim, num_heads=num_heads, qkv_bias=True,
                                           norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(feat_adaptor_depth)])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self,x, last_self_attention=False):
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x = blk(x, return_attention=last_self_attention)
        if last_self_attention:
            x, attn = x
        x = self.norm(x)
        if last_self_attention:
            return x, attn[:, :, 0, 1:]  # [B, heads, cls, cls-patch]
        return x

class Proto_Adaptor(nn.Module):
    def __init__(self, num_queries,embed_dim,proto_adaptor_depth=1,num_heads=4): #embedding_size:
        super(Proto_Adaptor, self).__init__()

        self.clsQueries = nn.Embedding(num_queries, embed_dim)

        # simple Transformer Decoder with num_decoder_layers
        self.num_decode_layers = proto_adaptor_depth
        self.decoder_cross_attention_layers = nn.ModuleList()
        self.decoder_self_attention_layers = nn.ModuleList()
        self.decoder_ffn_layers = nn.ModuleList()
        for _ in range(proto_adaptor_depth):
            self.decoder_cross_attention_layers.append(
                CrossAttentionLayer(d_model=embed_dim, nhead=num_heads)
            )
            self.decoder_self_attention_layers.append(
                SelfAttentionLayer(d_model=embed_dim, nhead=num_heads)
            )
            self.decoder_ffn_layers.append(
                FFNLayer(d_model=embed_dim, dim_feedforward=embed_dim)
            )

    def forward(self, embedding):
        B = embedding.size(0)
        outQueries = self.clsQueries.weight.unsqueeze(0).repeat(B, 1, 1)
        posQueries = pos = None

        for j in range(self.num_decode_layers):
            # attention: cross-attention first
            queries = self.decoder_cross_attention_layers[j](
                outQueries, embedding, pos=pos, query_pos=posQueries)
            # self-attention
            queries = self.decoder_self_attention_layers[j](
                queries, query_pos=posQueries)
            # FFN
            queries = self.decoder_ffn_layers[j](queries)

        return queries

class Linear_Clsfier(nn.Module):
    def __init__(self, num_cls,embedding_dim,linear_clsfier_depth=1): #embedding_size:
        super(Linear_Clsfier, self).__init__()

        if num_cls == 2:
            self.out_ch = 1
        else:
            self.out_ch = num_cls

        layers = []
        for i in range(linear_clsfier_depth):
            if i == linear_clsfier_depth - 1:
                layers += [nn.Linear(embedding_dim, self.out_ch)]
            else:
                layers += [nn.Linear(embedding_dim, embedding_dim),
                           nn.ReLU(inplace=True)
                           ]

        self.fc = nn.Sequential(*layers)


    def forward(self,embedding):

        embedding = embedding.mean(1)

        out = self.fc(embedding)

        return out



class FTFoundClsNet(nn.Module):
    def __init__(self, configs): #
        super(FTFoundClsNet, self).__init__()

        if configs.backbone == 'SAM_vit_b': # 'SAM_vit_b', 'SAM_vit_l', 'SAM_vit_h'
            checkpoint = '/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/SAM/sam_vit_b_01ec64.pth'
            self.backbone = sam_model_registry['vit_b'](checkpoint=checkpoint)
            self.preprocess = None
            self.feature_process = None

        self.feature_adaptor = None

        self.proto_generator = None



    def get_nonbackbon_params(self):
        decoder_params = []
        for name, param in self.named_parameters():
            # if name.startswith("backbone"):
            #     param.requires_grad = False
            if "backbone" not in name:
                decoder_params.append(param)


        return decoder_params

class EffFTFoundClsNet(nn.Module):
    def __init__(self, configs): #
        super(EffFTFoundClsNet, self).__init__()

        # self.feature_process = None
        # if 'SAM' in configs.backbone: # 'SAM_vit_b', 'SAM_vit_l', 'SAM_vit_h'
        #     self.feature_process =    # SAM img_embedding_size:(7007, 256, 64, 64), gt_size: (7007, 1)
        self.backbone = configs.backbone
        self.encoder = None
        self.emb_dim = 384
        if configs.backbone == 'DINO_vit_small':  # 'SAM_vit_b', 'SAM_vit_l', 'SAM_vit_h'
            checkpoint = '/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/DINO/dino_deitsmall16_pretrain.pth'  # patch size = 16
            self.encoder = vit_small()
            pretrained_model = torch.load(checkpoint, map_location=torch.device('cpu'))
            neq_load_external_vit(self.encoder, pretrained_model)
            self.emb_dim = self.encoder.embed_dim

        self.feature_adaptor = None
        if configs.feat_adaptor_depth != 0:
            self.feature_adaptor = Found_Feature_Adaptor(self.emb_dim,configs.feat_adaptor_depth,configs.num_heads)

        # self.norm = nn.LayerNorm(embedding_dim)

        self.clsfier = Linear_Clsfier(configs.num_cls,self.emb_dim,configs.linear_clsfier_depth)


        self.original_img_size = configs.img_size

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def norm(self,x):
        return F.normalize(x, dim=-1, eps=1e-10)

    def get_nonbackbon_params(self):
        decoder_params = []
        for name, param in self.named_parameters():
            # if name.startswith("backbone"):
            #     param.requires_grad = False
            if "encoder" not in name:
                # print(name)
                decoder_params.append(param)


        return decoder_params


    def forward_features(self,img):
        embedding = self.encoder.forward_backbone(img)

        if self.feature_adaptor is not None:
            embedding = self.feature_adaptor(embedding)

        embedding_mean = embedding.mean(1, keepdim=True)

        logits = self.clsfier(embedding)

        return embedding, embedding_mean,logits  # (B,N,emd_dim), (B,1,emd_dim)


    def forward(self,img):

        embedding = self.encoder.forward_backbone(img)

        if self.feature_adaptor is not None:
            embedding = self.feature_adaptor(embedding)

        logits = self.clsfier(embedding)

        # prob
        # prob = self.compute_prediction(embedding, prototypes, prob_type)

        return logits

    def compute_prediction(self,embedding, prototypes, prob_type):
        # embedding: (B,N,C), or (B,1+N,C) for DINO
        # prototypes: (B,num_cls,C)
        if prob_type == 'evi':
            evidence_logits_map = torch.einsum("bnc,bmc->bnm", embedding, prototypes)  # (B,N,Cls)
            evidence_map = F.softplus(evidence_logits_map)  # (B,N,Cls)
            evidence = torch.sum(evidence_map,1) # (B,Cls)

            prob = (evidence+1)/torch.sum(evidence+1,dim=1,keepdim=True)  # (B,cls)

            return prob
        elif prob_type == 'sw_evi': # self-weighted evidence
            evidence_logits_map = torch.einsum("bnc,bmc->bnm", embedding, prototypes)  # (B,N,Cls)
            weight_map = F.softmax(evidence_logits_map,dim=2)  # (B,N,Cls)
            evidence_map = F.softplus(evidence_logits_map)  # (B,N,Cls)

            weighted_evidence_map = weight_map * evidence_map  # (B,N,Cls)
            evidence = torch.sum(weighted_evidence_map, 1)  # (B,Cls)

            prob = (evidence + 1) / torch.sum(evidence + 1, dim=1, keepdim=True)  # (B,cls)

            return prob

        elif prob_type == 'gw_evi': # global weighted evidence, only for ''DINO'' having global feature
            evidence_logits_map = torch.einsum("bnc,bmc->bnm", embedding[:,1:,:], prototypes) # (B,N,cls)
            evidence_map = F.softplus(evidence_logits_map) # (B,N,cls)

            cossin_weight_map = torch.einsum("bnc,bmc->bnm", self.norm(embedding[:,1:,:]), self.norm(embedding[:,:1,:])) # (B,N,1)
            positive_weight_map = F.relu(cossin_weight_map) # (B,N,1)

            weighted_evidence_map = positive_weight_map * evidence_map # (B,N,cls)
            evidence = torch.sum(weighted_evidence_map, 1)  # (B,cls)

            prob = (evidence + 1) / torch.sum(evidence + 1, dim=1, keepdim=True)  # (B,cls)

            return prob

        ####    similar to above but from cosin similarity results

        elif prob_type == 'cosin':
            cos_sim_map =  torch.einsum("bnc,bmc->bnm", self.norm(embedding), self.norm(prototypes))  # (B,N,Cls)
            positive_sim_map = F.relu(cos_sim_map)  # (B,N,Cls)

            logits = torch.sum(positive_sim_map,dim=1)  # (B,cls)

            prob = F.softmax(logits,dim=1)

            return prob

        elif prob_type == 'g_cosin': # global , only for ''DINO'' having global feature
            cos_sim_global = torch.einsum("bnc,bmc->bnm", self.norm(embedding[:,:1,:]), self.norm(prototypes))  # (B,1,Cls)
            cos_sim_global = F.relu(cos_sim_global)  # (B,N,Cls) ************ ignored for ablation

            prob = F.softmax(cos_sim_global[:,0], dim=1)

            return prob

        elif prob_type == 'gwl_cosin': # global weighted local, only for ''DINO'' having global feature
            cos_sim_local_map = torch.einsum("bnc,bmc->bnm", self.norm(embedding[:, 1:, :]), self.norm(prototypes))  # (B,N,cls)
            positive_cos_sim_local_map = F.relu(cos_sim_local_map)  # (B,N,Cls) ************ ignored for ablation

            cossin_weight_map = torch.einsum("bnc,bmc->bnm", self.norm(embedding[:, 1:, :]),
                                             self.norm(embedding[:, :1, :]))  # (B,N,1)
            positive_weight_map = F.relu(cossin_weight_map)  # (B,N,1)

            weighted_cos_sim_local_map = positive_cos_sim_local_map * positive_weight_map  # (B,N,Cls)
            logits = torch.sum(weighted_cos_sim_local_map,dim=1)  # (B,Cls)

            prob = F.softmax(logits, dim=1)

            return prob

        elif prob_type == 'gwl_cosin_cons': # consistency between global and  weighted local, only for ''DINO'' having global feature, semi-supervised learning
            cos_sim_local_map = torch.einsum("bnc,bmc->bnm", self.norm(embedding[:, 1:, :]),
                                             self.norm(prototypes))  # (B,N,cls)
            positive_cos_sim_local_map = F.relu(cos_sim_local_map)  # (B,N,Cls) ************ ignored for ablation

            cossin_weight_map = torch.einsum("bnc,bmc->bnm", self.norm(embedding[:, 1:, :]),
                                             self.norm(embedding[:, :1, :]))  # (B,N,1)
            positive_weight_map = F.relu(cossin_weight_map)  # (B,N,1)

            weighted_cos_sim_local_map = positive_cos_sim_local_map * positive_weight_map  # (B,N,Cls)
            logits = torch.sum(weighted_cos_sim_local_map, dim=1)  # (B,Cls)

            prob_local = F.softmax(logits, dim=1)

            # global
            cos_sim_global = torch.einsum("bnc,bmc->bnm", self.norm(embedding[:, :1, :]),
                                          self.norm(prototypes))  # (B,1,Cls)
            cos_sim_global = F.relu(cos_sim_global)  # (B,N,Cls) ************ ignored for ablation

            prob_global = F.softmax(cos_sim_global[:, 0], dim=1)

            return prob_local, prob_global



















