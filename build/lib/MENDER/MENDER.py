import seaborn as sns
import numpy as np
import scanpy as sc
import squidpy as sq
from scipy.spatial import distance_matrix
from scipy.spatial.distance import *
import time
import anndata as ad
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from MENDER.utils import *
import os

class MENDER_single(object):
    def __init__(self,adata,ct_obs='ct',verbose=0,random_seed=666):
        # verbose:0-no; 1-scale level; 2-ME level
        if adata and isinstance(adata,ad.AnnData) and 'spatial' in adata.obsm:
            self.adata = adata
            self.set_ct_obs(ct_obs)
            # self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
            # self.ct_array = np.array(self.adata.obs[self.ct_obs])
            self.nn_mode = None
            self.nn_para = None
            self.count_rep = None
            self.include_self = None
            self.n_scales = None
            self.verbose = verbose
            self.random_seed=random_seed
        else:
            print('Please input an anndata object with spatial coordinates')
            exit(1)
            
    def dump(self,pre=''):
        self.adata.obs['MENDER'] = self.adata_MENDER.obs['MENDER'].copy()
        self.adata.write_h5ad(f'{pre}_GEX.h5ad')
        self.adata_MENDER.write_h5ad(f'{pre}_MENDER.h5ad')
        
        
    def set_ct_obs(self,new_ct):
        if new_ct not in self.adata.obs:
            print('Please input a valid cell type obs')
            return
        else:
            self.ct_obs = new_ct
            self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
            self.ct_array = np.array(self.adata.obs[self.ct_obs])
    
    def set_MENDER_para(self,
                    nn_mode='k', # how to construct the neighborhood graph
                    nn_para=1, # the parameter of nn graph, depending on nn_mode. For ring, the default val is 6 (because Visium has 6 neighbors)
                    count_rep='s', # how to count cell types within each scale
                    # n_neighbors
                    include_self=True,# include the node itself for the adjacent matrix
                    n_scales=15, # the number of scales

                   ):
        if nn_mode not in ['ring','radius','k']:
            print('nn_mode: please input in [ring,radius,k]')
            print('ring: usually used in grid-like spatial spots, such as Visium')
            print('radius: not restricted by spatial technologies, each scale is determined by the graph constructed by radius-nn')
            print('k: not restricted by spatial technologies, each scale is determined by the graph constructed by k-nn')
            
            return
        if count_rep not in ['s','a']:
            print('count_rep: please input in [s,a]')
            print('s: count the cell type frequencies for each single scale')
            print('a: accumulate the cell type frequencies for each single scale and all subordinate scales') 
            return 
        self.nn_mode = nn_mode
        self.nn_para = nn_para
        self.count_rep = count_rep
        self.include_self = include_self
        self.n_scales = n_scales
        
    def run_representation(self,group_norm=False):
        if self.nn_mode=='ring':
            self.ring_representation(group_norm)
        elif self.nn_mode=='radius':
            self.radius_representation(group_norm)
        elif self.nn_mode=='k':
            self.k_representation()
        else:
            print('Please input [ring,radius,k]')
            return
    
    def print_settings(self):
        print(f'adata: {self.adata.shape}')
        print(f'ct_obs: {self.ct_obs}')
        print(f'nn_mode: {self.nn_mode}')
        print(f'nn_para: {self.nn_para}')
        print(f'count_rep: {self.count_rep}')
        print(f'include_self: {self.include_self}')
        print(f'n_scales: {self.n_scales}')
        
    def estimate_radius(self):
        spatialmat = self.adata.obsm['spatial']
        min_dist_list = []
        cur_distmat = squareform(pdist(spatialmat))
        np.fill_diagonal(cur_distmat,np.inf)
        cur_min_dist = np.min(cur_distmat,axis=0)
        min_dist_list.append(cur_min_dist)
        min_dist_array = np.hstack(min_dist_list)
        neighbor_sz = np.median(min_dist_array)
        print(f'estimated radius: {neighbor_sz}')
        self.estimated_radius = neighbor_sz
        
    def mp_helper(self,cur_scale):
        adata_tmp = self.adata.copy()
        # cur_scale = i
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        if self.verbose==1:
            print(f'scale {cur_scale}')

  
        sq.gr.spatial_neighbors(adata_tmp,coord_type='generic',n_neighs=self.nn_para+cur_scale,set_diag=self.include_self)
        
        I = adata_tmp.obsp['spatial_connectivities']
        ME_X = np.zeros(shape=(cls_array.shape[0],ME_var_names_np_unique.shape[0]))

        time_start = time.time()

        for i in range(I.shape[0]):
            if self.verbose == 2:
                if i%1000==0:
                    time_end=time.time()

                    print('{0} MEs,time cost {1} s, {2} MEs, {3}s left'.format(i,time_end-time_start,I.shape[0]-i,(I.shape[0]-i)*(time_end-time_start)/1000))
                    time_start=time.time()

            cur_neighbors = I[i,:].nonzero()[1]

            cur_neighbors_cls = cls_array[cur_neighbors]
            cur_cls_unique,cur_cls_count = np.unique(cur_neighbors_cls,return_counts=1) #counting for each cluster
            cur_cls_idx = [np.where(ME_var_names_np_unique==c)[0][0] for c in cur_cls_unique] #c is string
            ME_X[i,cur_cls_idx] = cur_cls_count
        cur_ME_key = f'scale{cur_scale}'

        cur_X = ME_X

        return cur_ME_key,cur_X

        # self.adata.obsm[cur_ME_key] = cur_X.copy()
        
        
        
    def k_representation_mp(self,mp=200):
        from multiprocessing import Process, Pool
        
        
        
        pool = Pool(mp)
        i_list = np.arange(self.n_scales)
        ME_key_X_list = pool.map(self.mp_helper,i_list)
        self.ME_key_X_list = ME_key_X_list
        for i in range(len(ME_key_X_list)):
            self.adata.obsm[ME_key_X_list[i][0]] = ME_key_X_list[i][1]
        self.generate_ct_representation()
        
        
            
        
    def ring_representation(self,group_norm=False):
        
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        # self.exp_name = f'{nn_mode}_{count_rep}_{para}_{n_scales}_{cls_obs}'
        
        ME_X_prev = np.zeros(shape=(cls_array.shape[0],ME_var_names_np_unique.shape[0]))
        
        for cur_scale in range(self.n_scales):
            # cur_scale = i
            if self.verbose==1:
                print(f'scale {cur_scale}')
            
            sq.gr.spatial_neighbors(self.adata,coord_type='grid',n_neighs=self.nn_para,n_rings = cur_scale+1,set_diag=self.include_self)
            # sq.gr.spatial_neighbors(self.adata,coord_type='grid',n_neighs=self.nn_para,n_rings = cur_scale,set_diag=self.include_self)
            
            I = self.adata.obsp['spatial_connectivities']
            ME_X = np.zeros(shape=(cls_array.shape[0],ME_var_names_np_unique.shape[0]))

            time_start = time.time()
            
            for i in range(I.shape[0]):
                if self.verbose == 2:
                    if i%1000==0:
                        time_end=time.time()

                        print('{0} MEs,time cost {1} s, {2} MEs, {3}s left'.format(i,time_end-time_start,I.shape[0]-i,(I.shape[0]-i)*(time_end-time_start)/1000))
                        time_start=time.time()

                cur_neighbors = I[i,:].nonzero()[1]

                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique,cur_cls_count = np.unique(cur_neighbors_cls,return_counts=1) #counting for each cluster
                cur_cls_idx = [np.where(ME_var_names_np_unique==c)[0][0] for c in cur_cls_unique] #c is string
                ME_X[i,cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'

            if self.count_rep == 'a':
                cur_X = ME_X

            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                ME_X_prev = ME_X
            print(f'scale {cur_scale}, median #cells per ring (r={self.nn_para}):',np.median(np.sum(cur_X,axis=1)))
            
            self.adata.obsm[cur_ME_key] = cur_X.copy()
            if group_norm:
                self.adata.obsm[cur_ME_key] = self.adata.obsm[cur_ME_key]/np.sum(self.adata.obsm[cur_ME_key],axis=1,keepdims=True)
                self.adata.obsm[cur_ME_key] = np.nan_to_num(self.adata.obsm[cur_ME_key],0)
                
        self.generate_ct_representation()
            
            
    def radius_representation(self,group_norm=False):
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        # self.exp_name = f'{nn_mode}_{count_rep}_{para}_{n_scales}_{cls_obs}'
        
        ME_X_prev = np.zeros(shape=(cls_array.shape[0],ME_var_names_np_unique.shape[0]))
        
        for i in range(self.n_scales):
        
            cur_scale = i
            if self.verbose==1:
                print(f'scale {cur_scale}')
            
            sq.gr.spatial_neighbors(self.adata,coord_type='generic',radius=self.nn_para*(cur_scale+1),set_diag=self.include_self)
            I = self.adata.obsp['spatial_connectivities']
            ME_X = np.zeros(shape=(cls_array.shape[0],ME_var_names_np_unique.shape[0]))

            time_start = time.time()
            
            for i in range(I.shape[0]):
                if self.verbose ==2:
                    if i%1000==0:
                        time_end=time.time()

                        print('{0} MEs,time cost {1} s, {2} MEs, {3}s left'.format(i,time_end-time_start,I.shape[0]-i,(I.shape[0]-i)*(time_end-time_start)/1000))
                        time_start=time.time()

                cur_neighbors = I[i,:].nonzero()[1]

                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique,cur_cls_count = np.unique(cur_neighbors_cls,return_counts=1) #counting for each cluster
                cur_cls_idx = [np.where(ME_var_names_np_unique==c)[0][0] for c in cur_cls_unique] #c is string
                ME_X[i,cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'

            if self.count_rep == 'a':
                cur_X = ME_X

            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                ME_X_prev = ME_X
            print(f'scale {cur_scale}, median #cells per radius (r={self.nn_para}):',np.median(np.sum(cur_X,axis=1)))

            self.adata.obsm[cur_ME_key] = cur_X.copy()
            if group_norm:
                
                self.adata.obsm[cur_ME_key] = self.adata.obsm[cur_ME_key]/np.sum(self.adata.obsm[cur_ME_key],axis=1,keepdims=True)
                self.adata.obsm[cur_ME_key] = np.nan_to_num(self.adata.obsm[cur_ME_key],0)
        self.generate_ct_representation()
        
    def k_representation(self):
        cls_array = self.ct_array
        ME_var_names_np_unique = self.ct_unique
        # self.exp_name = f'{nn_mode}_{count_rep}_{para}_{n_scales}_{cls_obs}'
        
        ME_X_prev = np.zeros(shape=(cls_array.shape[0],ME_var_names_np_unique.shape[0]))
        
        for i in range(self.n_scales):
        
            cur_scale = i
            if self.verbose==1:
                print(f'scale {cur_scale}')
            
            sq.gr.spatial_neighbors(self.adata,coord_type='generic',n_neighs=self.nn_para+cur_scale,set_diag=self.include_self)
            I = self.adata.obsp['spatial_connectivities']
            ME_X = np.zeros(shape=(cls_array.shape[0],ME_var_names_np_unique.shape[0]))

            time_start = time.time()
            
            for i in range(I.shape[0]):
                if self.verbose==2:
                    if i%1000==0:
                        time_end=time.time()

                        print('{0} MEs,time cost {1} s, {2} MEs, {3}s left'.format(i,time_end-time_start,I.shape[0]-i,(I.shape[0]-i)*(time_end-time_start)/1000))
                        time_start=time.time()

                cur_neighbors = I[i,:].nonzero()[1]

                cur_neighbors_cls = cls_array[cur_neighbors]
                cur_cls_unique,cur_cls_count = np.unique(cur_neighbors_cls,return_counts=1) #counting for each cluster
                cur_cls_idx = [np.where(ME_var_names_np_unique==c)[0][0] for c in cur_cls_unique] #c is string
                ME_X[i,cur_cls_idx] = cur_cls_count
            cur_ME_key = f'scale{cur_scale}'

            if self.count_rep == 'a':
                cur_X = ME_X

            # else:
            elif self.count_rep == 's':
                cur_X = ME_X - ME_X_prev
                ME_X_prev = ME_X

            self.adata.obsm[cur_ME_key] = cur_X
        self.generate_ct_representation()
        
            
    def generate_ct_representation(self):
        
        ME_var_names_np_unique = self.ct_unique
        whole_feature_list = []
        while_feature_X = []
        for ct_idx in range(len(ME_var_names_np_unique)):
            rep_list = []
            for i in range(self.n_scales):
                rep_list.append(self.adata.obsm[f'scale{i}'][:,ct_idx])
                whole_feature_list.append(f'ct{ct_idx}scale{i}')
                while_feature_X.append(self.adata.obsm[f'scale{i}'][:,ct_idx])

            cur_ct_rep = np.array(rep_list).transpose()
            cur_obsm = f'ct{ct_idx}'
            self.adata.obsm[cur_obsm] = cur_ct_rep

        self.adata.obsm[f'whole'] = np.array(while_feature_X).transpose()

        # make another anadata for whole feature
        adata_feature = ad.AnnData(X = np.array(while_feature_X).transpose())
        adata_feature.obs_names = self.adata.obs_names
        adata_feature.var_names = whole_feature_list
        adata_feature.obsm['spatial'] = self.adata.obsm['spatial']
        # adata_obs_keys = self.adata.obs.keys()
        for k in self.adata.obs.keys():
            adata_feature.obs[k] = self.adata.obs[k]
        # for k in self.adata.uns.keys()
        if 'spatial' in self.adata.uns:
            adata_feature.uns['spatial'] = self.adata.uns['spatial']
        self.adata_MENDER = adata_feature

        
    def preprocess_adata_MENDER(self,mode=0,neighbor=True):
        if hasattr(self,'is_adata_MENDER_preprocess')==False or self.is_adata_MENDER_preprocess==False:
            if mode==0:
                sc.pp.normalize_total(self.adata_MENDER)
                sc.pp.log1p(self.adata_MENDER)
            elif mode==1:
                sc.pp.normalize_total(self.adata_MENDER)
            elif mode==2:
                sc.pp.log1p(self.adata_MENDER)
            elif mode==3:
                pass
                

            sc.pp.pca(self.adata_MENDER)
            if neighbor:
                sc.pp.neighbors(self.adata_MENDER)
            self.is_adata_MENDER_preprocess = True
        else:
            pass


    
    def run_clustering_normal(self,target_k,run_umap=False,if_reprocess=True,mode=0):
        if if_reprocess:
            self.preprocess_adata_MENDER(mode)
        if run_umap:
            sc.tl.umap(self.adata_MENDER)
            self.adata_MENDER.obsm['X_MENDERMAP2D'] = self.adata_MENDER.obsm['X_umap'].copy()
        
        if target_k>0:
            res = res_search(self.adata_MENDER,target_k=target_k,random_state=self.random_seed)
            sc.tl.leiden(self.adata_MENDER,resolution=res,key_added=f'MENDER_leiden_k{target_k}',random_state=self.random_seed)
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_k{target_k}'].copy()
        elif target_k<0:
            res = -target_k
            sc.tl.leiden(self.adata_MENDER,resolution=res,key_added=f'MENDER_leiden_res{res}',random_state=self.random_seed)
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_res{res}'].copy()
            
        else:
            print('please input a valid target_k')
            return 
    
   
    def output_cluster(self,obs):

        ax = sc.pl.embedding(self.adata_MENDER,basis='spatial',color=obs,show=False)
        ax.axis('equal')



  
        
        
        
class MENDER(object):
    def __init__(self,adata,batch_obs,ct_obs='ct',verbose=0,random_seed=666):
        self.adata = adata[:,0:0].copy()
        self.batch_obs = batch_obs
        self.ct_obs = ct_obs
        self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
        self.verbose=verbose
        self.random_seed=random_seed
    
    
    def prepare(self):
        self.adata_list = []
        self.batch_list = np.array(self.adata.obs[self.batch_obs].cat.categories)
        for b in self.batch_list:
            cur_a = self.adata[self.adata.obs[self.batch_obs]==b]
            self.adata_list.append(cur_a)
            
    def set_ct_obs(self,new_ct):
        if new_ct not in self.adata.obs:
            print('Please input a valid cell type obs')
            return
        else:
            self.ct_obs = new_ct
            self.ct_unique = np.array(self.adata.obs[self.ct_obs].cat.categories)
    
    def set_MENDER_para(self,
                    nn_mode='k', # how to construct the neighborhood graph
                    nn_para=1, # the parameter of nn graph, depending on nn_mode. For ring, the default val is 6 (because Visium has 6 neighbors)
                    count_rep='s', # how to count cell types within each scale
                    # n_neighbors
                    include_self=True,# include the node itself for the adjacent matrix
                    n_scales=15, # the number of scales

                   ):
        if nn_mode not in ['ring','radius','k']:
            print('nn_mode: please input in [ring,radius,k]')
            print('ring: usually used in grid-like spatial spots, such as Visium')
            print('radius: not restricted by spatial technologies, each scale is determined by the graph constructed by radius-nn')
            print('k: not restricted by spatial technologies, each scale is determined by the graph constructed by k-nn')
            
            return
        if count_rep not in ['s','a']:
            print('count_rep: please input in [s,a]')
            print('s: count the cell type frequencies for each single scale')
            print('a: accumulate the cell type frequencies for each single scale and all subordinate scales') 
            return 
        self.nn_mode = nn_mode
        self.nn_para = nn_para
        self.count_rep = count_rep
        self.include_self = include_self
        self.n_scales = n_scales
        
    def run_representation(self,group_norm=False):
        print('for faster version, use run_representation_mp')
        # rst_dict = {}
        adata_MENDER_list = []
        for i in range(len(self.batch_list)): 
            cur_batch_name = self.batch_list[i]
            cur_batch_adata = self.adata_list[i]
            print(f'total batch: {len(self.batch_list)}, running batch {cur_batch_name}')
            
            cur_MENDER = MENDER_single(cur_batch_adata,ct_obs=self.ct_obs,verbose=self.verbose,random_seed=self.random_seed)
            cur_MENDER.set_MENDER_para(
                nn_mode=self.nn_mode,
                nn_para=self.nn_para,
                count_rep=self.count_rep,
                include_self=self.include_self,
                n_scales=self.n_scales
            )
            cur_MENDER.ct_unique = self.ct_unique
            cur_MENDER.run_representation(group_norm)
            cur_adata_MENDER = cur_MENDER.adata_MENDER.copy()
            # rst_dict[cur_batch_name] = cur_adata_MENDER
            adata_MENDER_list.append(cur_adata_MENDER)
        self.adata_MENDER_list = adata_MENDER_list
        
        
    def mp_helper(self,i):
        cur_batch_name = self.batch_list[i]
        cur_batch_adata = self.adata_list[i]
        print(f'total batch: {len(self.batch_list)}, running batch {cur_batch_name}')
        
        cur_MENDER = MENDER_single(cur_batch_adata,ct_obs=self.ct_obs,verbose=self.verbose)
        cur_MENDER.set_MENDER_para(
            nn_mode=self.nn_mode,
            nn_para=self.nn_para,
            count_rep=self.count_rep,
            include_self=self.include_self,
            n_scales=self.n_scales
        )
        cur_MENDER.ct_unique = self.ct_unique
        cur_MENDER.run_representation(self.group_norm)
        cur_adata_MENDER = cur_MENDER.adata_MENDER.copy()
        return cur_adata_MENDER
        
        
    def estimate_radius(self):
        for i in range(len(self.batch_list)):
            cur_batch_name = self.batch_list[i]
            cur_batch_adata = self.adata_list[i]
            cur_MENDER = MENDER_single(cur_batch_adata,ct_obs=self.ct_obs,verbose=self.verbose)
 
            cur_MENDER.estimate_radius()
            
            print(f'{cur_batch_name}:estimated radius:{cur_MENDER.estimated_radius}')
        
        
    
    def run_representation_mp(self,mp=200,group_norm=False):
        print('default number of process is 200')
        from multiprocessing import Process, Pool
        self.group_norm = group_norm
        # rst_dict = {}
        adata_MENDER_list = []
        i_list = np.arange(len(self.batch_list))
        pool = Pool(mp)
        adata_MENDER_list = pool.map(self.mp_helper,i_list)
        self.adata_MENDER = adata_MENDER_list[0].concatenate(adata_MENDER_list[1:])
        self.adata_MENDER_dump = self.adata_MENDER.copy()
    
    def refresh_adata_MENDER(self):
        del self.adata_MENDER
        self.adata_MENDER = self.adata_MENDER_dump.copy()
        self.is_adata_MENDER_preprocess = False
        
#   

    def preprocess_adata_MENDER(self,neighbor=True):
        if hasattr(self,'is_adata_MENDER_preprocess')==False or self.is_adata_MENDER_preprocess==False:
            sc.pp.normalize_total(self.adata_MENDER)
            sc.pp.log1p(self.adata_MENDER)

            sc.pp.pca(self.adata_MENDER)
            if neighbor:
                sc.pp.neighbors(self.adata_MENDER)
            self.is_adata_MENDER_preprocess = True
        else:
            pass



        
        
        

        
        
    def run_clustering_normal(self,target_k,run_umap=False,if_reprocess=True):
        if if_reprocess:
            self.preprocess_adata_MENDER()
        
        if run_umap:
            sc.tl.umap(self.adata_MENDER)
            self.adata_MENDER.obsm['X_MENDERMAP2D'] = self.adata_MENDER.obsm['X_umap'].copy()
        
        if target_k>0:
            res = res_search(self.adata_MENDER,target_k=target_k,random_state=self.random_seed)

            sc.tl.leiden(self.adata_MENDER,resolution=res,key_added=f'MENDER_leiden_k{target_k}',random_state=self.random_seed)
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_k{target_k}'].copy()
            
        elif target_k<0:
            res = -target_k
            sc.tl.leiden(self.adata_MENDER,resolution=res,key_added=f'MENDER_leiden_res{res}',random_state=self.random_seed)
            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_leiden_res{res}'].copy()
            
        else:
            print('please input a valid target_k')
            return 
        
    def run_clustering_mclust(self,target_k,run_umap=False):
        
        if run_umap:
            self.preprocess_adata_MENDER(neighbor=True)
            
            sc.tl.umap(self.adata_MENDER)
            self.adata_MENDER.obsm['X_MENDERMAP2D'] = self.adata_MENDER.obsm['X_umap'].copy()
        else:
            self.preprocess_adata_MENDER(neighbor=False)
        
        if target_k>0:
            self.adata_MENDER = STAGATE.mclust_R(self.adata_MENDER,used_obsm='X_pca', num_cluster=target_k)

            self.adata_MENDER.obs['MENDER'] = self.adata_MENDER.obs[f'MENDER_mclust_k{target_k}'].copy()
            
        
            
        else:
            print('please input a valid target_k')
            return 
        
    
        
    
      

    
    

            
   



    def output_cluster(self,dirname,obs):
        # path = 'a'
        sc.pl.embedding(self.adata_MENDER,basis='X_MENDERMAP2D',color=obs,save=f'_umap_{obs}')
        path = dirname
        os.mkdir(f'figures/spatial_{path}')
        adata_feature = self.adata_MENDER
        for i in range(len(self.batch_list)):

            cur_batch = self.batch_list[i]
            cur_a = adata_feature[adata_feature.obs[self.batch_obs]==cur_batch]
            # sc.pl.embedding(cur_a,basis='spatial',color=obs,save=f'_{path}/{cur_batch}',show=False,title=cur_batch)
            ax = sc.pl.embedding(cur_a,basis='spatial',color=obs,show=False,title=cur_batch,save=None)
            ax.axis('equal')
            plt.savefig(f'figures/spatial_{path}/{cur_batch}.png',dpi=200,bbox_inches='tight',transparent=True)
            plt.close()






    def output_cluster_single(self,obs,idx=0):
        cur_batch = self.batch_list[idx]
        adata_feature = self.adata_MENDER

        cur_a = adata_feature[adata_feature.obs[self.batch_obs]==cur_batch]
        sc.pl.embedding(cur_a,basis='spatial',color=obs,show=True,title=cur_batch)




        
        
    def output_cluster_all(self,obs='MENDER',obs_gt='gt'):
        
        

        sc.pl.embedding(self.adata_MENDER,basis='spatial',color=obs)

        self.adata_MENDER.obs[self.batch_obs] = self.adata_MENDER.obs[self.batch_obs].astype('category')
        for si in self.adata_MENDER.obs[self.batch_obs].cat.categories:
            cur_a = self.adata_MENDER[self.adata_MENDER.obs[self.batch_obs]==si]
            
            if obs_gt in cur_a.obs:
                nmi = compute_NMI(cur_a,obs_gt,obs)
                ari = compute_ARI(cur_a,obs_gt,obs)
                nmi = np.round(nmi,3)
                ari = np.round(ari,3)
            
            # pas = compute_PAS(cur_a,obs)
            # chaos = compute_CHAOS(cur_a,obs)
            # pas = np.round(pas,3)
            # chaos = np.round(chaos,3)

            if obs_gt in cur_a.obs:
                # title = f'{si}\n nmi:{nmi} ari: {ari}\n pas:{pas} chaos:{chaos}'
                title = f'{si}\n nmi:{nmi} ari: {ari}'
                
            else:
                # title = f'{si}\n pas:{pas} chaos:{chaos}'
                title = si
                
            ax = sc.pl.embedding(cur_a,basis='spatial',color=obs,show=False)
            ax.axis('equal')
            ax.set_title(title)
    
    
    