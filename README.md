# Recursive Hierarchical Clustering
In this project, we build an unsupervised system to capture dominating user behaviors from clickstream data (traces of users’ click events), and visualize the detected behaviors in an intuitive manner. 

Our system identifies "clusters" of similar users by partitioning a similarity graph (nodes are users; edges are weighted by clickstream similarity). 

The partitioning process leverages **iterative feature pruning** to capture the natural hierarchy within user clusters and produce intuitive features for visualizing and understanding captured user behaviors.

This script is used to perform user behavior clustering. Users are put into a hierarchy of clusters, each identified by a list of behavior patterns.

---

## Dependency
Python library `numpy`, `scipy` is required.

## Usage
The main file of this script is `recursiveHierarchicalClustering.py`. 
There are two ways of executing the script, through the command line interface or through python import.

### Command Line Interface

```
$> python recursiveHierarchicalClustering.py input.txt output/ 0.05
```

#### Arguments
**inputPath**: specify the path to a files that contains information for the users to be clustered. Each line is represent a user.
> user_id \t A(1)G(10)

Where the `A` and `G` are actions and `1` and `10` are the frequencies of each action. The `user_id` grows from 1 to the total number of users.

**outputPath**: The directory to place all temporary files as well as the final result.

**sizeThreshold** (optional): Defines the minimum size of the cluster, that we are going to further divide.
`0.05` means clusters containing less than 5% of the total instances is not going to be further splitted.


### Python Interface
```python
import recursiveHierarchicalClustering as rhc
data = rhc.getSidNgramMap(inputPath)
matrixPath = '%smatrix.dat' % inputPath
treeData = rhc.runDiana(outputPath, data, matrixPath)
```

Here treeData is the resulting cluster tree. Same as `output/result.json` if ran through CLI.

## Result
**output/matrix.dat**: A distance matrix for the root level is stored to avoid repeated calculation. If the file is available, the scirpt will read in the matrix instead of calculating it again. The file format is a N*N distance matrix scaled to integer in the range of (0-100).

**output/result.json**: Stores the clustering result, in the form of `['t', sub-cluster list, cluster info]` or `['l', user list, cluster info]`.

* **Node type**: node type can be either `t` or `l`. `l` means leaf which means the cluster is not further split. `t` means tree meaning there are further splitting for the given cluster.
* **Sub-cluster list**: a list of clusters that is the resulting clusters derived from splitting the current cluster.
* **User list**: a list of user ids representing the users in the given cluster.
* **Cluster info**: a dictionary containing meta data for the cluster.
	* **gini**: gini-coefficient for chi-square score value distribution, measures the skewness of feature importance distribution.
	* **sweetspot**: the modularity for the best `k` we picked when further splitting this cluster.
	* **exlusions**: a list of top features (ranked) that helps to distinguish the cluster from others.
	* **exclusionsScore**: the chi-square scores correspond to the top features listed in **exclusions**.
	
## Configuration
This script is designed to be distributed on multiple machines with shared file system. However, it is also be configured to run locally.
The configuration is stored in `server.json` in the following format:
```javascript
{
	"threadNum": 5,
	"minPerSlice": 1000,
	"server":
		["server1.example.com", "server2.example.com"]
}	
```

* **threadNum** specifies how many threads can be ran on each server.
* **minPerSlice** specifies the minimum number of users each thread should handle.
* **server** specifies the server to be used for matrix computation task. If you want to run it locally, specify it as `["localhost"]`.

---
## Visulization
Along with the clustering, we also developed a visulization tool based on [D3.js](https://d3js.org/) in
order to inspect the content of the resulting clusters.

### Generating data file
To generate a json file readable by `D3.js`, you can run the following bash command:

```
$> python visulization.py result.json  input.txt vis/vis.json
```

#### Arguments
**result.json** is the output file of the recursive hierarchical clustering.

**intput.txt** is the path to a files that contains information for the users to be clustered. Each line is represent a user.

**vis.json** is the path to the final output.

### Running webserver
To properly display the visulization, you need to set up a web server under folder `vis`:
```
$> python -m SimpleHTTPServer
```

And then by visiting `http://localhost:8000/multi_color.html?json=vis.json` you will be able to look at the visulization for `vis.json`.

---
## Optimization
A faster version is also available making better use of `numpy`'s support for 
matrix computation.

This requires additional package of `sklearn`.


## Usage
The main file of the faster implementation is `recursiveHierarchicalClusteringFast.py`. 
There are two ways of executing the script, through the command line interface or through python import.

### Command Line Interface

```
$> python recursiveHierarchicalClusteringFast.py input.txt output/ 0.05
```

### Python Interface
```python
import recursiveHierarchicalClustering as rhc
import recursiveHierarchicalClusteringFast as rhcFast
data = rhc.getSidNgramMap(inputPath)
treeData = rhcFast.runDiana(inputPath, data, outPath)
```

Here treeData is the resulting cluster tree. Same as `output/result.json` if ran through CLI.

---
## Publication
Gang Wang, Xinyi Zhang, Shiliang Tang, Haitao Zheng, Ben Y. Zhao. 
[Unsupervised Clickstream Clustering for User Behavior Analysis](http://www.cs.ucsb.edu/~ravenben/publications/pdf/clickstream-chi16.pdf).
Proceedings of SIGCHI Conference on Human Factors in Computing Systems (CHI), San Jose, CA, May 2016. 
