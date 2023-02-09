# Simple-Multi-view-Can-Bring-Powerful-Graph-Neural-Network
<center><img width="374" alt="image" src="https://user-images.githubusercontent.com/101496242/217576624-342f8592-b588-49d6-ab08-9ccbea4da170.png"></center>

<center><img width="351" alt="image" src="https://user-images.githubusercontent.com/101496242/217576796-2ca599a5-74ac-4338-a73b-ed2578c5a898.png"></center>
### Overall Architecture

The framework of MV-GNN is illustrated in Fig.2. For the input graph, instead of running GNN on one view ( the entire graph ), we first construct $K$ different views by sampling $K$ diverse subgraphs. The adjacency matrix and feature matrix of each view are fed into one shared base GNN model over all views. Therefore, we obtain $K$ representations for each node. Then we leverage only one injective function to combine these $K$ representations for the final representation of each node.\

### Run the code
Reproduce our model with GCN backbone:\
step 1: \
 &ensp; <code>cd ./code/MV-GCN</code>\
step 2:\
 &ensp;  <code>python train.py </code> for dataset Cora, CiteSeer, PubMed\
 &ensp;  <code>python train_multilabel.py </code> for dataset Douban

Reproduce our model with GAT backbone:\
step 1: \
 &ensp;  <code>cd ./code/MV-GAT  </code>\
step 2:\
 &ensp;  <code>python execute_sdata_sparse.py </code> for dataset Cora, CiteSeer, PubMed\
 &ensp;  <code>python execute_doubanmovie_sparse.py  </code> for dataset Douban
