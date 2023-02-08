# Simple-Multi-view-Can-Bring-Powerful-Graph-Neural-Network
<img width="374" alt="image" src="https://user-images.githubusercontent.com/101496242/217576624-342f8592-b588-49d6-ab08-9ccbea4da170.png">
<img width="351" alt="image" src="https://user-images.githubusercontent.com/101496242/217576796-2ca599a5-74ac-4338-a73b-ed2578c5a898.png">
### overall architecture
The framework of MV-GNN is illustrated in Fig.2. For the input graph, instead of running GNN on one view ( the entire graph ), we first construct $K$ different views by sampling $K$ diverse subgraphs. The adjacency matrix and feature matrix of each view are fed into one shared base GNN model over all views. Therefore, we obtain $K$ representations for each node. Then we leverage only one injective function to combine these $K$ representations for the final representation of each node.
