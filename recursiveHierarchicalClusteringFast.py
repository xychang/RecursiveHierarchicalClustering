#!/usr/bin/python

"""
the script is a modification of recursiveHierarchicalClustering.py
It make better use of the matrix computation in numpy
and thus has improved speed

# arguments

arg1: path to a file containing the different ngram count mapping
example:
    10 \t A5A(1)y8y(29)...
the first col is line number
the second col is a list of pattern(count)

arg2: path to a directory to put the output file

arg3: sizeThreshold, defines the minimum size of the cluster, that we
      are going to further divide
      deafult: 0.05: means the size cannot be less than 5% of the total
                     instances

# result:

resultDir/matrix.dat: store the pairwise distance matrix
 if the file exist before the scirpt is run, the script assume the matrix
 is correct and will read it in
resultDir/result.json: store the clustering result
 the result is recorded in a nested list, in the form of:
    [type, list of sub-clusters, information about the cluster]
"""

import sys
import json
import time
# import cProfile
import cPickle
import numpy as np
import recursiveHierarchicalClustering as rhc
import calculateDistanceInt as calculateDistance


class HCClustering:

    def __init__(self, matrix, sid_seq, outPath, exclusions, idxToSid,
                 sizeThreshold, idfMap=None):
        self.matrix = matrix
        self.sizeThreshold = sizeThreshold
        self.maxDistance = 100
        self.sid_seq = sid_seq
        self.outPath = outPath
        self.exclusions = exclusions
        if not idxToSid:
            idxToSid = [x+1 for x in range(len(sid_seq))]
        self.idxToSid = idxToSid
        if not idfMap:
            idfMap = rhc.excludeFeatures(rhc.getIdf(sid_seq, idxToSid), exclusions)
        self.idfMap = idfMap

    def getDia(self, cid, cluster):
        """
        find out cluster diameter
        :type cid: int
            - cluter id
        :type cluster: List[int]
            - all the stream ids in the cluster
        :rtype: int
            - the diameter of the cluster
        """
        startTime = time.time()
        sumDists = self.sumEntriesMap[cid]
        maxIdx = np.argmax(sumDists)

        maxDist = np.max(self.matrix[cluster[maxIdx]][cluster])
        total = len(cluster)
        for idx in range(len(cluster)):
            sumDist = sumDists[idx]
            # optimization: prune out rows inpossible to have diameter
            if (sumDist * 2) < (maxDist * total):
                continue
            curMax = np.max(self.matrix[cluster[idx]][cluster])
            if curMax > maxDist:
                maxDist = curMax
        return maxDist

    def splitCluster(self, (baseCluster, diameter, cid)):
        """
        :type baseCluster: List[int]
            - take in the cluster to be splited as a list of stream index
        :type diameter: int (not used)
        :type cid: int (cluster id, not used)

        :rtype: (baseCluster, newCluster, sumEntryA, sumEntryB, sumAB)
        :rtype baseCluster: List[int]
            - a list of stream index in one part of the split (clusterA)
        :rtype newCluster: List[int]
            - a list of stream index in the other part of the split (clusterB)
        :rtype sumEntryA: List[int]
            - sum of each row for clusterA
        :rtype sumEntryB: List[int]
            - sum of each row for clusterB
        :rtype sumAB: float
            - sum of all pairwise distance between clusterA and clusterB
        """
        newCluster = []
        # the number of nodes in the base cluster
        baseNodes = totalNodes = len(baseCluster)
        sumDists = self.sumEntriesMap[cid]
        newDists = np.zeros(totalNodes, dtype=np.float64)

        # all reasoning about the process below is written in
        # recursiveHierarchicalClustering.py
        while baseNodes > 1:
            # print('%d nodes left' % baseNodes)
            newNodes = totalNodes - baseNodes
            if baseNodes == totalNodes:
                difDist = sumDists
                maxIdx = np.argmax(difDist)
                maxValue = difDist[maxIdx]
            else:
                diffs = (sumDists - newDists) * newNodes - \
                        newDists * (baseNodes - 1)
                maxIdx = np.argmax(diffs)
                maxValue = diffs[maxIdx]
            if maxValue <= 0:
                break
            newCluster.append(maxIdx)

            # update the distance to new cluster for all nodes not in the cluster
            newDists = newDists + self.matrix[baseCluster[maxIdx]][baseCluster]
            # ensure that nodes in the new cluster will never be selected
            newDists[maxIdx] += self.maxDistance * totalNodes
            baseNodes -= 1

        clusterAIdx = newCluster
        clusterBIdx = np.delete(np.arange(totalNodes), clusterAIdx)
        sumEntryA = (newDists - self.maxDistance * totalNodes)[clusterAIdx]
        sumEntryB = (sumDists - newDists)[clusterBIdx]

        sumAB = np.sum(newDists[clusterBIdx], dtype=np.float64)
        baseCluster = np.array(baseCluster)

        return list(baseCluster[newCluster]), list(baseCluster[clusterBIdx]),\
            sumEntryA, sumEntryB, sumAB

    def modularityBasics(self):
        """
        compute the basic values needed during modularity calculate
        so that we don't need to calculate it again and again
        :type matrix: List[List(int)]
        :rtype sumAll: int
            - the sum of all distances in the matrix
        :rtype sumEntries: List[int]
            - the sum of each row in the matrix
        """
        sumEntries = (self.maxDistance * len(self.matrix) -
                      np.sum(self.matrix, axis=1, dtype=np.float64) -
                      self.maxDistance)
        sumAll = np.sum(sumEntries)
        self.sumAll = sumAll
        self.sumEntries = sumEntries
        return sumAll, sumEntries

    def evaluateModularity(self, clusters, sumEntries):
        """
        calculate the modularity given cluster division
        :type clusters: List[tuple]
            - a list of cluster tuples for the clusters we wan to study
        :type sumEntries: List[int]
            - the sum of each row in the matrix
        :rtype: float
        """
        m = self.sumAll
        # for each cluster, compute the modularity
        # firstEntry is the intra cluster distance
        # second entry is rowSums product for each node pair in cluster
        firstEntry = 0.0
        secondEntry = 0.0
        for cluster, sumEntry in zip(clusters, sumEntries):
            nodes = np.array(cluster)
            firstEntry += self.maxDistance * len(nodes) ** 2 - \
                np.sum(sumEntry, dtype=np.float64)

            for nodeA in nodes:
                firstEntry -= self.maxDistance - self.matrix[nodeA][nodeA]
            sumRowCur = self.sumEntries[nodes]
            secondEntry += np.sum(sumRowCur) ** 2
        return (firstEntry - secondEntry / m) / m

    def evaluateModularityShift(self, (clusterA, clusterB), sumAB):
        """
        the reasoning is detailed in recursiveHierarchicalClustering.py
        :type clusterA: List[int]
            - user ids for the first cluster resulted from split
        :type clusterB: List[int]
            - user ids for the second cluster resulted from split
        :type matrix: List[List[int]]
            - distance matrix
        :type sumAB: float
            - sum of all pairwise distance between clusterA and clusterB
        :rtype: float
        """
        m = self.sumAll
        firstEntry = (self.maxDistance * len(clusterA) * len(clusterB) - sumAB) 
        secondEntry = np.sum(self.sumEntries[clusterA]) \
            * np.sum(self.sumEntries[clusterB])
        return -2 * (firstEntry - secondEntry / m) / m

    def runDiana(self):
        """
        Perform recursive hierarchical clustering
        """
        global matrixCompTotal, splitTotal, modularityTotal, diaTotal
        global excluTotal

        M = self.matrix
        self.modularityBasics()
        print('[LOG]: finished calculating modularityBasics')

        # child Cid => parent Cid
        clusterHi = []

        # record the evaluation metrics
        evalResults = {}

        cid = 1
        clusters = [(range(len(self.matrix)), self.maxDistance, cid)]

        # get a mapping from cid => list of row sums for all ids in cluster
        self.sumEntriesMap = {}
        self.sumEntriesMap[cid] = np.sum(self.matrix, axis=1, dtype=np.float64)

        while clusters[-1][1] and len(clusters[-1][0]) > self.sizeThreshold:
            parentCid = clusters[-1][2]
            # print('splitting %s\t%s' % (clusters[-1][1],clusters[-1][2]))
            clusterHi.append((parentCid, cid + 1, cid + 2))

            curTime = time.time()
            (clusterA, clusterB, sumEntryA, sumEntryB, sumAB) = \
                (self.splitCluster(clusters.pop()))
            splitTotal += time.time() - curTime

            curTime = time.time()
            cid += 1
            self.sumEntriesMap[cid] = sumEntryA
            clusters.append((clusterA, self.getDia(cid, clusterA), cid))
            # clusters.append((clusterA, np.mean(sumEntryA) / len(clusterA), cid))
            cid += 1
            self.sumEntriesMap[cid] = sumEntryB
            clusters.append((clusterB, self.getDia(cid, clusterB), cid))
            # clusters.append((clusterB, np.mean(sumEntryB) / len(clusterB), cid))
            diaTotal += time.time() - curTime

            curTime = time.time()
            clusters = \
                sorted(clusters, key=lambda x: (x[1], len(x[0]))
                       if len(x[0]) > self.sizeThreshold else (0, 0))
            if len(clusters) == 2:
                # if it is the first time to compute modularity
                evalResult = self.evaluateModularity(
                    (clusterA, clusterB), (sumEntryA, sumEntryB))
            else:
                # if it is based on the previous scores
                evalResult = evalResults[len(clusters) - 1] + \
                    self.evaluateModularityShift((clusterA, clusterB), sumAB)
            modularityTotal += time.time() - curTime

            # print(sorted([len(x[0]) for x in clusters], reverse = True))
            # print(len(clusters[-1][0]))
            evalResults[len(clusters)] = evalResult
            # print('cluster num is %d, modularity %f' % (len(clusters), evalResult))

        # print(evalResults)
        sweetSpot = rhc.getSweetSpot(evalResults, 5)
        sweetSpot = sorted(evalResults.keys(),
                           key=lambda x: abs(x - sweetSpot))[0]
        print('[LOG]: sweetSpot is %d, modularity %f' %
              (sweetSpot, evalResults[sweetSpot]))

        # merge the clusters to the point of sweet spot
        clusterMap = dict([(row[2], row) for row in clusters])
        cids = [(row[2]) for row in clusters]
        while(len(cids) > sweetSpot):
            (parentCid, childACid, childBCid) = clusterHi.pop()
            # dismeter doesn't matter, so put zero here
            clusterMap[parentCid] = \
                (clusterMap[childACid][0] + clusterMap[childBCid][0],
                 0, parentCid)
            cids.append(parentCid)
            cids.remove(childACid)
            cids.remove(childBCid)

        # reconstruct the cluster list after merging
        clusters = [(x[0], x[1], None, x[2]) for cid, x in clusterMap.items()
                    if cid in cids]

        # get the exclusion map according to the current clustering
        startTime = time.time()
        excludeMap, exclusionScoreMap, scoreMap = \
            rhc.getExclusionMap(clusters, self.sid_seq, self.idfMap,
                                self.idxToSid, [row[3] for row in clusters],
                                self.exclusions)
        excluTotal += time.time() - startTime

        # for each cluster, we start a new clustering
        results = []
        for cidx in range(len(clusters)):
            row = clusters[cidx]
            idxs = row[0]    # get the list of all node in clusters
            sids = sorted([self.idxToSid[nidx] for nidx in idxs])
            excludedFeatures = excludeMap[row[3]]
            excludedScores = exclusionScoreMap[row[3]]

            # if we want to continue cluster this subcluster
            if len(sids) > self.sizeThreshold:
                newExclusions = self.exclusions + excludedFeatures
                # remove sids where the vector have all zeros
                newExclusionSet = set(newExclusions)
                oldLen = len(sids)
                excludedSids = [sid for sid in sids if len(
                    set(self.sid_seq[sid].keys()) - newExclusionSet) == 0]
                sids = [sid for sid in sids if len(
                    set(self.sid_seq[sid].keys()) - newExclusionSet) > 0]
                # if the cluster size is too small after feature selection,
                # don't cluster it
                # or if the cluster diameter is 0
                if not len(sids) > self.sizeThreshold:
                    result = ('l', sids + excludedSids,
                              {'exclusions': excludedFeatures,
                               'exclusionsScore': excludedScores})
                else:
                    matrixStart = time.time()
                    matrix = calculateDistance.partialMatrix(
                        sids,
                        rhc.excludeFeatures(rhc.getIdf(sid_seq, sids),
                                            newExclusions),
                        ngramPath,
                        'tmp_%d' % row[3],
                        '%st%d_' % (self.outPath, row[-1]),
                        True)
                    matrixCompTotal += time.time() - matrixStart
                    # after the matrix is calculated, we need to handle a
                    # speacial case where all entries in the matrix is zero,
                    # besically means if the first row of the
                    # matrix adds up to zero
                    # if this is the case, do not split the cluster
                    if np.sum(matrix[0]) == 0:
                        result = ('l', sids + excludedSids,
                                  {'exclusions': excludedFeatures,
                                   'exclusionsScore': excludedScores})
                    else:
                        # now that we have a new distance matrix, go and
                        # do another round of clustering
                        result = HCClustering(
                            matrix,
                            sid_seq,
                            '%sp%d_' % (self.outPath, row[-1]),
                            newExclusions,
                            sids,
                            self.sizeThreshold).runDiana()
                        if len(results) > 2:
                            info = result[2]
                        else:
                            info = {}

                        # put the excluded sids back as a cluster
                        if (len(excludedSids) > 0):
                            result[1].append(('l', excludedSids,
                                              {'isExclude': True}))

                        info['exclusions'] = excludedFeatures
                        info['exclusionsScore'] = excludedScores
                        # base on the score map, calculate the gini coefficient
                        # score map format {cid:[(feature, score)]}
                        # info['gini'] = getGini([x[1] for x in scoreMap[row[3]]])
                        result = (result[0], result[1], info)
            else:
                result = ('l', sids,
                          {'exclusions': excludedFeatures,
                           'exclusionsScore': excludedScores})
            results.append(result)

        return(('t', results, {'sweetspot': evalResults[sweetSpot]}))


def run(ngramPath, sid_seq, outPath):
    """
    this function is a simpler version of the main function
    it is a wrapper around runDiana
    intended to be used for rhcBootstrap (my testing script)
    :type ngramPath: str
          - the path to the computed pattern dataset
    :type sid_seq: Dict{int:Dict{str:int}}
          - for each user, record the pattern and corresponding occurence #
    :type outPath: str
          - the path to store the output and temporary file
    """
    global matrixCompTotal

    startTime = time.time()

    idxToSid = [x+1 for x in range(len(sid_seq))]

    idfMap = rhc.excludeFeatures(rhc.getIdf(sid_seq, idxToSid), [])

    matrix = calculateDistance.partialMatrix(
        idxToSid, idfMap, ngramPath, 'tmp_%sroot' % int(time.time()),
        outPath, True)

    print('[LOG]: first matrixTime %f' % (time.time() - startTime))
    matrixCompTotal += time.time() - startTime

    hc = HCClustering(
        matrix, sid_seq, outPath, [], idxToSid,
        sizeThreshold=0.05 * len(sid_seq), idfMap=idfMap)
    result = hc.runDiana()

    print('[STAT]: total clustering time %f' % (time.time() - startTime))
    return result


# store the total time needed to compute distance matrix
matrixCompTotal = 0
# store the total time needed to compute modularity
modularityTotal = 0
# store the total time needed to split clusters
splitTotal = 0
# store the total time needed to compute cluster diameter
diaTotal = 0
# store the total time needed to find out feature exclusion
excluTotal = 0

if __name__ == '__main__':

    ngramPath = sys.argv[1]
    outPath = sys.argv[2]
    sizeThreshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    startTime = time.time()
    sid_seq = rhc.getSidNgramMap(ngramPath)
    print('[LOG]: total users %d' % len(sid_seq))
    result = run(ngramPath, sid_seq, outPath)

    json.dump(result, open('%sresult.json' % outPath, 'w'))

    print('[STAT]: total time %f' % (time.time() - startTime))
    print(('[STAT]: maxtrix com: %f, dismeter: %f, modularity: %f, split: %f, '
           'exclusion: %f') %
          (matrixCompTotal, diaTotal, modularityTotal, splitTotal, excluTotal))
