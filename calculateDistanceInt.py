#!/usr/bin/python
"""
This is used to compute distance matrix
It makes heavy use of numpy and thus runs faster
"""

import multiprocessing
import json
import sys
import math
import subprocess
import os
import glob
import cPickle
import time
from array import *
import datetime
import gc
# import cProfile
# import pstats
# import line_profiler
from scipy import sparse
import numpy as np
from sklearn import preprocessing
from sklearn.utils import sparsefuncs


config = json.load(open('servers.json'))
# thread number
THREAD_NUM = config['threadNum']

# the minimum slice size
MIN_SLICE = config['minPerSlice']

# the minimum size per server
MIN_SERVER = THREAD_NUM * MIN_SLICE

PI_VALUE = 3.141592653589793


class myThread (multiprocessing.Process):
    def __init__(self, threadID, prefix, matrix, sfrom, sto):
        # pr = line_profiler.LineProfiler()
        # self.pr = pr
        # pr.add_function(self.run)
        # pr.enable()

        self.threadID = threadID
        self.matrix = matrix
        self.sfrom = sfrom
        self.sto = sto
        # self.fo = gzip.open(prefix+'_'+str(srange[0])+'-'+
        # str(srange[1])+'.gz', 'w')
        self.fo = prefix+'_'+str(sfrom[0])
        multiprocessing.Process.__init__(self)

    # @profile
    def run(self):
        # pr = cProfile.Profile()
        print '[LOG]: start new thread '+str(self.threadID)
        curTime = time.time()
        distM = self.matrix[self.sfrom].dot(
                    self.matrix[self.sto].T).todense()
        distM = np.maximum(
            np.arccos(np.minimum(distM, np.ones(distM.shape))) /
            (PI_VALUE/200)-0.01,
            np.zeros(distM.shape)).astype(np.int8)

        # np.savetxt(self.fo, distM, fmt = '%d')
        np.save(self.fo + '.npy', distM)
        print('[LOG]: thread %d finished after %d' %
              (self.threadID, time.time() - curTime))

        # self.pr.disable()
        # # sortby = 'cumulative'
        # # pstats.Stats(pr).strip_dirs().sort_stats(sortby).print_stats()
        # self.pr.print_stats()


def getSparseMatrix(idfMap, fromSids, toSids, inputPath):
    """
    build a sparse matrix based on stream and feature
    :type idfMap: Dict{str:float}
          - build a mapping between feature and features idf score
    :type fromSids: List[int]
    :type toSids: List[int]
    :type inputPath: str
          - the path to the computed pattern dataset
    :rtype: a numpy matrix
    """
    sidxs, fidxs, values, features = \
        ngramToMatrix(inputPath, fromSids | toSids)

    featureDict = dict([(features[idx], idx) for idx in range(len(features))])

    matrix = sparse.csr_matrix(
        (values, (sidxs, fidxs)),
        shape=(max(sidxs) + 1, len(features)), dtype=np.float64)

    featureWeight = [idfMap[feature] if feature in idfMap else 0
                     for feature in features]

    # apply idf transformation to the matrix
    sparsefuncs.inplace_csr_column_scale(matrix, np.array(featureWeight))

    # to maintain consistency with previous implementation, round the
    # result after applying idf
    matrix = np.floor(matrix)

    # apply normalization to the matrix
    matrix = preprocessing.normalize(matrix, copy=False)

    return matrix


def calDist(inputPath, sidsPath, outputPath, tmpPrefix='', idfMapPath=None):
    """
    :type intputPath: str
          - the path to the computed pattern dataset
    :type sidsPath: str
          - the path to two lists of sids that we need to calculate one as 
          from and one as to
    :type outputPath: str
          - a path to specify where the final result is placed
    :type tmpPrefix: str (default: '')
          - a predix used so that temporary files have no conflict
    :type idfMapPath: str (default: None)
          - if the idf for each pattern is already computed, provide a path
          to avoid repeated computation
    """
    print('[LOG]: %s computing matrix for %s' % (datetime.datetime.now(),
                                                 sidsPath.split('sid_')[0]))
    # read the ngram file, generate a node list
    lineNum = 1
    sids = cPickle.load(open(sidsPath))

    fromSids = set(sids[0])
    # in principle the toSids should be all sids
    toSids = set(sids[1])

    startTime = time.time()

    # get the idfMap, if provided
    if (idfMapPath):
        idfMap = cPickle.load(open(idfMapPath))
    else:
        idfMap = None

    matrix = getSparseMatrix(idfMap, fromSids, toSids, inputPath)

    # slice fromSids into THREAD_NUM pieces
    if (len(fromSids) < MIN_SLICE * THREAD_NUM):
        step = MIN_SLICE
    else:
        step = len(fromSids) / THREAD_NUM + 1

    print('[LOG]: %s preprocessing takes %.4fs' %
          (datetime.datetime.now(), time.time() - startTime))

    tid = 0
    threads = []
    start = 0
    while start < len(fromSids):
        tid += 1
        thread = myThread(tid, '%s%sdist' % (outputPath, tmpPrefix),
                          matrix, sids[0][start:start+step], sids[1])
        thread.start()
        threads.append(thread)
        start += step

    # wait unitl thread ends
    for t in threads:
        t.join()

    # print('everything takes %.4fs' % (time.time() - startTime))


def partialMatrix(sids, idfMap, ngramPath, tmpPrefix, outputPath, 
                  realSid=False):
    """
    at this point the outputPath should have been made
    calling this function returns a distance matrix
    :type sids: List[int]
    :type idfMap: Dict{str:float}
          - build a mapping between feature and features idf score
    :type ngramPath: str
          - the path to the computed pattern dataset
    :type tmpPrefix: str
          - a predix used so that temporary files have no conflict
    :type outputPath: str
          - a path to specify where the final result is placed
    :type realSid: bool
          - indicates whether the sid index is actually offset by 1,
          so that the lowest sid is 0
    """
    servers = json.load(open('servers.json'))['server']
    if not realSid:
        sids = [x + 1 for x in sids]
    total = len(sids)

    if total < MIN_SLICE:
        # if the matrix is small enough to be handle by a single thread, avoid
        # writting files comlete to reduce overhead
        matrix = getSparseMatrix(idfMap, set(sids), set(sids), ngramPath)
        distM = matrix[sids].dot(matrix[sids].T).todense()
        distM = np.maximum(
            np.arccos(np.minimum(distM, np.ones(distM.shape))) /
            (PI_VALUE/200)-0.01,
            np.zeros(distM.shape)).astype(np.int8)
        return np.array(distM)

    if (total < MIN_SERVER * len(servers)):
        step = MIN_SERVER
    else:
        step = total / len(servers) + 1
    processes = []
    start = 0
    cPickle.dump(idfMap, open('%s%sidf.pkl' % (outputPath, tmpPrefix), 'w'))

    # if number of tasks is small enough, run it locally
    if total < MIN_SERVER:
        servers = ['localhost']

    for server in servers:
        if (start >= total):
            break
        cPickle.dump([sids[start:start+step], sids], open('%s%ssid_%s.pkl' %
                     (outputPath, tmpPrefix, server), 'w'))
        print('[LOG]: starting in %s for %s' % (server, tmpPrefix))
        if server == 'localhost':
            calDist(ngramPath, '%s%ssid_%s.pkl' %
                    (outputPath, tmpPrefix, server),
                    outputPath, tmpPrefix, '%s%sidf.pkl' %
                    (outputPath, tmpPrefix))
        else:
            gc.collect()
            processes.append(subprocess.Popen(
                ['ssh', server,
                 ('cd %s\npython calculateDistanceInt.py %s %s%ssid_%s.pkl' +
                  ' %s %s %s%sidf.pkl') %
                 (os.getcwd(), ngramPath, outputPath,
                  tmpPrefix, server, outputPath, tmpPrefix, outputPath,
                  tmpPrefix)]))
        start += step

    for process in processes:
        process.wait()

    print('[LOG]: %s merge started for %s%s' %
          (datetime.datetime.now(), outputPath, tmpPrefix))

    files = sorted(glob.glob('%s%sdist_*' % (outputPath, tmpPrefix)),
                   key=lambda x: int(x.split('_')[-1][:-4]))

    matrix = np.concatenate(tuple([np.load(file) for file in files]))
    print('[LOG]: %s merge finished for %s%s' % (
        datetime.datetime.now(), outputPath, tmpPrefix))

    for fname in glob.glob('%s%s*' % (outputPath, tmpPrefix)):
        os.remove(fname)
    # print('[LOG]: all tmp files removed for %s%s' % (outputPath, tmpPrefix))
    print('[LOG]: %s matrix computation finished for %s%s' % (
        datetime.datetime.now(), outputPath, tmpPrefix))

    return matrix


def ngramToMatrix(inputPath, asids):
    """
    convert all_sid_ngram data into a sparse matrix coordinates
    :type inputPath: str
          - the path to the computed pattern dataset
    :type asids: List[int]
          - the users we want to study
    """
    fidx = []
    values = []
    sids = []
    for line in open(inputPath):
        # get the sid
        sid = int(line.split('\t')[0])
        if sid not in asids:
            continue
        # get the ngram
        line = line.strip()
        line = line.split('\t')[1]
        # remove the trailing ) so that the spliting would not have an
        # empty tail
        line = line[:-1].split(')')
        curSeq = [(item[0], int(item[1])) for item in
                  map(lambda x: x.split('('), line)]

        for feature, value in curSeq:
            # records.append((sid, feature, value))
            sids.append(sid)
            fidx.append(feature)
            values.append(value)

    features = set(fidx)
    features = list(features)
    featureDict = dict([(features[idx], idx) for idx in range(len(features))])
    fidx = [featureDict[fid] for fid in fidx]

    return sids, fidx, values, features


if __name__ == "__main__":
    if (sys.argv[1] == 'coord'):
        ngramToMatrix(*sys.argv[2:])
    else:
        if (len(sys.argv) > 5):
            calDist(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        else:
            calDist(sys.argv[1], sys.argv[2], sys.argv[3])