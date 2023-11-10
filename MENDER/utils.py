
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial import *
from sklearn.preprocessing import *
import numpy as np

def res_search(adata,target_k = 7, res_start = 0.1, res_step = 0.05, res_epochs = 10,random_state=666):

    print(f"searching resolution to k={target_k}")
    res = res_start
    sc.tl.leiden(adata, resolution=res,random_state=random_state)

    old_k = len(adata.obs['leiden'].cat.categories)
    print("Res = ", res, "Num of clusters = ", old_k)

    run = 0
    while old_k != target_k:
        old_sign = 1 if (old_k<target_k) else -1
        sc.tl.leiden(adata, resolution=res+res_step*old_sign,random_state=random_state)
        new_k = len(adata.obs['leiden'].cat.categories)
        print("Res = ", res+res_step*old_sign, "Num of clusters = ", new_k)
        if new_k == target_k:
            res = res+res_step*old_sign
            print("recommended res = ", str(res))
            return res
        new_sign = 1 if (new_k<target_k) else -1
        if new_sign==old_sign:
            res = res+res_step*old_sign
            print("Res changed to", res)
            old_k = new_k
        else:
            res_step = res_step/2
            print("Res changed to", res)
        if run>res_epochs:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run+=1
    print("Recommended res = ", str(res))
    return res





def _compute_CHAOS(clusterlabel,location):

    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)
#     matched_location = location
    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel==k,:]
        if len(location_cluster)<=2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i,location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1


    return np.sum(dist_val)/len(clusterlabel)
def fx_1NN(i,location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    return np.min(dist_array)
    
    #     results = 
    
    
    
    
def _compute_PAS(clusterlabel,location):
    
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i,matched_location,k=10,cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results)/len(clusterlabel)
    
    
def fx_kNN(i,location_in,k,cluster_in):
#     print(i)

# def
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)


    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind]!=cluster_in[i])>(k/2):
        return 1
    else:
        return 0


    
    
    

def markerFC(adata_valid,marker_list,sdm_key):
    rst_dict = {}
    sdm_unique = adata_valid.obs[sdm_key].cat.categories
    for marker in marker_list:
        mean_exp_list = []
        for sdm in sdm_unique:
            mean_exp_list.append(np.mean(adata_valid[adata_valid.obs[sdm_key]==sdm][:,marker].X))
        max_sdm_idx = np.argmax(mean_exp_list)
#         print(sdm_unique[max_sdm_idx])

        max_sdm_value = np.max(mean_exp_list)
        other_sdm_value = np.mean(adata_valid[adata_valid.obs[sdm_key]!=sdm_unique[max_sdm_idx]][:,marker].X)
        cur_fc = max_sdm_value/other_sdm_value
        rst_dict[marker] = cur_fc
    return rst_dict

from sklearn.metrics import *
def compute_ARI(adata,gt_key,pred_key):
    return adjusted_rand_score(adata.obs[gt_key],adata.obs[pred_key])

def compute_NMI(adata,gt_key,pred_key):
    return normalized_mutual_info_score(adata.obs[gt_key],adata.obs[pred_key])

def compute_CHAOS(adata,pred_key,spatial_key='spatial'):
    return _compute_CHAOS(adata.obs[pred_key],adata.obsm[spatial_key])

def compute_PAS(adata,pred_key,spatial_key='spatial'):
    return _compute_PAS(adata.obs[pred_key],adata.obsm[spatial_key])



def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='X_pca', random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    try:
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
        mclust_res = np.array(res[-2])
    except:
        print('mclust error')
        mclust_res = np.zeros(shape=(adata.shape[0]))

    adata.obs[f'MS_mclust_k{num_cluster}'] = mclust_res
    adata.obs[f'MS_mclust_k{num_cluster}'] = adata.obs[f'MS_mclust_k{num_cluster}'].astype('int')
    adata.obs[f'MS_mclust_k{num_cluster}'] = adata.obs[f'MS_mclust_k{num_cluster}'].astype('category')
    return adata



def CNC(adata_raw,leiden_res,k,is_preprocessed=False):
    # this is the cellular neighborhood clustering methods proposed by Nolan's Cell paper:
    # Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front
    # the implementation is based on the paper's description and the github code:https://github.com/nolanlab/NeighborhoodCoordination/blob/master/Neighborhoods/Neighborhood%20Identification.ipynb
    
    # this function input 
    # (1) adata_raw: spatial omics anndata data
    # (2) leiden_res: the leiden resolution of the cell type clustering, which is used to be counted the frequency within each neighborhood
    # (3) k: the expected number of domains
    
    print('start CNC...')
    import squidpy as sq
    import scanpy as sc
    import anndata as ad
    from sklearn.cluster import MiniBatchKMeans
    
    adata = adata_raw.copy()
    
    print('preprocessing...')
    if not is_preprocessed:
#         the data has not been preprocessed
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000)
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)

    print('sc neighboring...')
    sc.pp.pca(adata)
    sc.pp.neighbors(adata,method='gauss')
    print('cell typing...')
    sc.tl.leiden(adata,resolution=leiden_res,key_added='ct',random_state=666)
    adata.obs['ct'] = adata.obs['ct'].astype('category')
    ct_obs = 'ct'
    n_cells = adata.shape[0]

    cls_array = np.array(adata.obs[ct_obs])

    ct_array = adata.obs[ct_obs].cat.categories

    ME_X = np.zeros(shape=(n_cells,len(ct_array)))


    sq.gr.spatial_neighbors(adata,coord_type='generic',n_neighs=10,set_diag=True)
    I = adata.obsp['spatial_connectivities']

    print('searching...')
    for i in range(I.shape[0]):

        cur_neighbors = I[i,:].nonzero()[1]

        cur_neighbors_cls = cls_array[cur_neighbors]
        cur_cls_unique,cur_cls_count = np.unique(cur_neighbors_cls,return_counts=1) #counting for each cluster
        cur_cls_idx = [np.where(ct_array==c)[0][0] for c in cur_cls_unique] #c is string
        ME_X[i,cur_cls_idx] = cur_cls_count

    adata_CNC = ad.AnnData(ME_X)
    adata_CNC.obs_names = adata.obs_names
    adata_CNC.var_names = ct_array
    adata_CNC.obs = adata.obs.copy()
    adata_CNC.obsm = adata.obsm.copy()
    print('clustering...')
    km = MiniBatchKMeans(n_clusters = k,random_state=0)
#     km = KMeans(n_clusters = k,random_state=0)

    pred = km.fit_predict(adata_CNC.X)
    adata_CNC.obs['CNC'] = pred.astype('str')
    adata_CNC.obs['CNC'] = adata_CNC.obs['CNC'].astype('category')
    return adata_CNC