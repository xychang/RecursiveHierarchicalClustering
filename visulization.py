#!/usr/bin/python

# the script to convert the resulting json file from RHC to one suited for visulization

# arguments

# arg1: path to the json file generated from RHC
# arg2: path to a file containing the different ngram count mapping
# arg3: output path of the json file used for visulization

import json
import sys
import numpy as np
import recursiveHierarchicalClustering as rhc

inPath = sys.argv[1] # 'outpath/result.json'
sid_ngram = sys.argv[2] # 'input.txt'
outPath = sys.argv[3] # 'outpath/vis.json'


binCount = 24 # display 24 bins in visulization

data = json.load(open(inPath))
sid_seq = rhc.getSidNgramMap(sid_ngram)

# output all users in the tree
def allUser(tree):
	users = []
	if (tree[0] == 'l'):
		return tree[1]
	for subTree in tree[1]:
		users.extend(allUser(subTree))
	return users

# get pattern distribution
def getPatternDist(pattern, sids, sid_seq):
	ratios = []
	for sid in sids:
		totalPattern = sum([x[1] for x in sid_seq[sid].items()])
		currentPattern = sid_seq[sid][pattern] if pattern in sid_seq[sid] else 0
		ratio = float(currentPattern) / totalPattern
		ratios.append(ratio)
	return ratios

# given a (not leaf) tree, return the children filed of json
def getJsonChildren(tree, sid_seq, clusterId):
    if tree[0] == 'l':
        return None, clusterId
    if 'sweetspot' in tree[2] and tree[2]['sweetspot'] < 0.01:
        return None, clusterId

    subTrees = []
    allSids = allUser(tree)
    sidsAll = [allUser(subTree) for subTree in tree[1]]

    for tidx in range(len(tree[1])):
        subTree = tree[1][tidx]
        sids = sidsAll[tidx]
        size = len(sids)
        children, clusterId = getJsonChildren(subTree, sid_seq, clusterId)
        name = ''
        info =  {}
        if 'sweetspot' in subTree[2]:
            info['sweetspot'] = subTree[2]['sweetspot']
        if 'gini' in subTree[2]:
            info['gini'] = subTree[2]['gini']
        if 'isExclude' in subTree[2]:
            info['isExclude'] = subTree[2]['isExclude']
        if 'exclusions' in subTree[2]:
            info['exclusions'] = subTree[2]['exclusions']
            # for each excluded features, calculate the average within cluster and out of cluster
            # and also the distribution bins
            for eidx in range(len(info['exclusions'])):
                # the second entry can be a longer description of the feature excluded
                info['exclusions'][eidx] = exclusions = [info['exclusions'][eidx]] * 2 
                key = exclusions[0]
                localDists = getPatternDist(key, sids, sid_seq)
                globalDists = getPatternDist(key, list(set(allSids) - set(sids)), sid_seq)
                maxRange = max(localDists + globalDists) + 0.00000001 # so that max is counted in the last bin
                # if the maxRange is too large, reduce it
                maxRange = min(maxRange, max(np.mean(localDists) + np.std(localDists) * 2, np.mean(globalDists) + np.std(globalDists) * 2))
                minRange = min(localDists + globalDists)
                binSize = float(maxRange - minRange) / binCount
                localDists = [int((ratio - minRange) / binSize) for ratio in localDists]
                globalDists = [int((ratio - minRange) / binSize) for ratio in globalDists]
                localBins = [0] * binCount
                for ratio in localDists:
                    if ratio >= binCount:
                        ratio = binCount - 1
                    localBins[ratio] += 1
                globalBins = [0] * binCount
                for ratio in globalDists:
                    if ratio >= binCount:
                        ratio = binCount - 1
                    globalBins[ratio] += 1
                exclusions.append([[float(x) / len(localDists) for x in localBins],\
                 [float(x) / len(globalDists) for x in globalBins]])
                exclusions.append(subTree[2]['exclusionsScore'][eidx])
        info['size'] = size
        info['id'] = clusterId
        clusterId += 1
        if children:
            subTrees.append({'name' : name, 'children': children, 'info': info})
        else:
            subTrees.append({'name' : name, 'size' : size, 'info': info})
    return subTrees, clusterId

# actually transfer the dumped clustering into json readable from visulization tool
def dumpLabels(treeData, sid_seq, outFile):
    json.dump({'name':'root', 'children': getJsonChildren(treeData, sid_seq, 1)[0]}, open(outFile, 'w'))

dumpLabels(data, sid_seq, outPath)