import dgl
import os
import torch
import numpy as np
import scipy.sparse as sp

from typing import Dict, Any, Tuple

class DataHandler():
    
    def __init__(self, data,Q,S,A,ls) -> None:
        """
        Load and preprocess Graph
        data:原始图[height, width, bands]
        Q:相关系数矩阵[HW，超像素个数] Q[207400,2074]
        S:超像素 [超像素个数,光谱维度] S[2074,8]
        A:超像素邻接矩阵 A[2074,2074]
        ls:?
        """ 
        
        self.p_graph: dgl.DGLGraph = self.load_p_graph(self,data,Q,S,A,ls)
        self.sp_graph: dgl.DGLGraph = self.load_sp_graph(self,data,Q,S,A,ls)

    def load_p_graph(self,data,Q,S,A,ls) -> Dict[str, Any]:
        """
        Load data for the specified data name.

        Args:
            root_dir (str, optional): The root directory where the data is located. Defaults to DATA_ROOT_DIR.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded data. The dictionary has two keys:
                - "p_graph" (dgl.DGLGraph): The DGL graph of the pixel-level Graph.
                - "sp_graph" (dgl.DGLGraph): The DGL graph of the superpixel-level Graph.
        """
        height, width, bands = data.shape
        adj = sp.coo_matrix((height*width, height*width))
        #
        adj=np.tensor((height*width, height*width)) 

        graph = dgl.from_scipy(adj)
        graph = dgl.remove_self_loop(graph)
        # dgl.save_graphs(graph_path, [graph])
        # print(f"Saved graph to {graph_path}.")
        