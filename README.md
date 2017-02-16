# RecursiveHierarchicalClustering
In this project, we build an unsupervised system to capture dominating user behaviors from clickstream data (traces of users’ click events), and visualize the detected behaviors in an intuitive manner. 

Our system identifies "clusters" of similar users by partitioning a similarity graph (nodes are users; edges are weighted by clickstream similarity). 

The partitioning process leverages **iterative feature pruning** to capture the natural hierarchy within user clusters and produce intuitive features for visualizing and understanding captured user behaviors.
