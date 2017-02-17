import math
import numpy as np
import scipy.stats
import cPickle
import json
import sys
import cProfile
import pstats
import StringIO


def avgStdWithZero(l, numZero):
    """
    get the std of a list with trailing zeros
    :type l: List[int]
    :type numZero: int
    """
    if len(l) == 0:
        return 0, 0
    avg = float(sum(l)) / (len(l) + numZero)
    error = 0.0
    for num in l:
        error += (num - avg) ** 2
    error += numZero * (avg ** 2)
    error = error / (len(l) + numZero)
    return avg, math.sqrt(error)


def mutual_info(patternInCluster, allPatternInCluster, totalPattern,
                totalAllPattern):
    """
    This function is not used
    get the mutual information score for a given cluster and given pattern
    :type patternInCluster: int
          - the number of occurence of this pattern in this cluster
    :type allPatternInCluster: int
          - the total number of occurences for all patterns in this cluster
    :type totalPattern: int
          - the total number of occurences of this pattern in all clusters
    :type totalAllPattern: int
          - the total number of occurences for all patterns in all clusters
    """
    N = totalAllPattern
    D = patternInCluster
    B = allPatternInCluster - D
    C = totalPattern - D
    A = totalAllPattern - C - D - B

    very_small = 1.0/N/1000000

    tmpD = 1.0*N*D/((B+D)*(C+D))+very_small
    tmpC = 1.0*N*C/((C+D)*(A+C))+very_small
    tmpB = 1.0*N*B/((A+B)*(B+D))+very_small
    tmpA = 1.0*N*A/((A+B)*(A+C))+very_small

    muinfo = (1.0*D/N) * math.log(tmpD, 2) \
           + (1.0*C/N) * math.log(tmpC, 2) \
           + (1.0*B/N) * math.log(tmpB, 2) \
           + (1.0*A/N) * math.log(tmpA, 2)

    return muinfo


# @profile
def chi_square(dist1, usersLeft1,  dist2, usersLeft2):
    """
    get the chi square score for a given cluster and given pattern
    :type dist1: List[int]
          - All the users' coutn of this pattern except those having 0
    :type usersLeft1: int
          - The number of users don't have this pattern
    :type dist2: List[int]
          - All the users' coutn of this pattern except those having 0
    :type usersLeft2: int
          - The number of users don't have this pattern
    dist1 is used to build the base bins that dist2 will be put into
    """

    avg, s = avgStdWithZero(dist1, usersLeft1)
    avg2, s2 = avgStdWithZero(dist2, usersLeft2)
    s = s / 3.0
    if s == 0:
        if s2 == 0:
            return ((avg2 - avg) / avg) * (len(dist2) + usersLeft2) \
                   if not avg == 0 else 0
        else:
            # should be based on dist1 still, need to figure out how to select bins
            return chi_square(dist2, usersLeft2, dist1, usersLeft1)

    # print(s, avg)
    bins = [(idx, (idx + 1)) for idx in range(-18, 18)]
    bins = [((x * s + avg), (y * s + avg)) for (x, y) in bins]
    bins = [(-float('inf'), bins[0][0])] + bins
    bins += [(bins[-1][1], float('inf'))]

    # distribute samples from dist1 into bins
    # binMap = dict([(idx, len([1 for x in dist1 if bins[idx][0]<=x<bins[idx][1]])) for \
    #     idx in range(len(bins))])
    binMap = dict([(idx, 0) for idx in range(38)])
    for x in dist1:
        x = int((x - avg) / s + 19)
        if x < 0:
            x = 0
        elif x > 37:
            x = 37
        binMap[x] += 1
    idxZero = int((- avg) / s + 19)
    idxZero = 0 if idxZero < 0 else 37 if idxZero > 37 else idxZero
    binMap[idxZero] += usersLeft1

    # merge bins that have less than 5 memebers
    for idx in range(1, len(bins))[::-1]:
        if (binMap[idx] < 5):
            binMap[idx - 1] += binMap[idx]
            del binMap[idx]
            bins[idx - 1] = (bins[idx - 1][0], bins[idx][1])
            del bins[idx]
            # print('delete', idx)
    if (binMap[0] < 5):
        if len(binMap) == 1:
            # too few data to get chi-square
            return 0
        nextIdx = sorted(binMap.keys())[1]
        bins[0] = (bins[0][0], bins[1][1])
        binMap[0] += binMap[nextIdx]
        del binMap[nextIdx]
        del bins[1]
    idxZero = [idx for idx in range(len(bins))
               if bins[idx][0] <= 0 < bins[idx][1]][0]
    bins1 = [binMap[idx] for idx in sorted(binMap.keys())]
    # distribute sample from dist2 into bins
    bins2 = np.array([len([1 for x in dist2
                           if bins[idx][0] <= x < bins[idx][1]])
                      for idx in range(len(bins))])
    bins2[idxZero] += usersLeft2

    bins1 = np.array([x * float(usersLeft2 + len(dist2)) /
                      (usersLeft1 + len(dist1)) for x in bins1])
    chisqValue, pvalue = (scipy.stats.chisquare(bins2, f_exp=bins1))
    return chisqValue


def chi_square_feature(cid_pattern_list, cid_user_cnt,
                       interested_cids=None, printEval=False):
    """
    get a list of feature values, compute the one that distinguishes the
    most from distuributions in other clusters
    :type cid_pattern_list: Dict{int:Dict{str:List[int]}}
          - a dictionary for user's pattern count for each cluster
          - the first dictionary key is the cluster id
          - the second dictionary key is the pattern string
          - the value is a list of users' pattern count, excluding 0
    :type cid_user_cnt: Dist{int:int}
          - total number of users in each cluster
    :type interested_cids: List{int} (default: None)
          - a list of cluster ids that we want to compute chi_square for
          - None means compute for all clusters
    :type printEval: bool
          - whether output intermedirary result to tmp.txt
    :rtype: Dict{int: List[(str, float)]}
          - a map of cid and (feature, score) list
    """
    if printEval:
        fout = open('tmp.txt', 'w')
    resultMap = {}
    for cid in cid_pattern_list:
        if interested_cids and cid not in interested_cids:
            continue
        scores = {}
        for pattern in cid_pattern_list[cid]:
            baselist = reduce(
                lambda x, y: x+y,
                [cid_pattern_list[curCid][pattern]
                 for curCid in cid_pattern_list
                 if not curCid == cid and pattern in cid_pattern_list[curCid]],
                [])
            usersLeftBase = sum([cid_user_cnt[curCid]
                                 for curCid in cid_user_cnt
                                 if not curCid == cid]) - len(baselist)
            curList = cid_pattern_list[cid][pattern]
            usersLeft = cid_user_cnt[cid] - len(curList)
            scores[pattern] = chi_square(baselist, usersLeftBase,
                                         curList, usersLeft)

        resultMap[cid] = sorted(scores.items(),
                                key=lambda x: x[1], reverse=True)
        if printEval:
            fout.write('%s\t%s\n' % (cid, cid_user_cnt[cid]))
            for (feature, score) in resultMap[cid][:50]:
                fout.write('%s\t%s\n' % (feature, score))
            fout.write('--------------------------\n')
            fout.flush()

            print('%s\t%s' % (cid, cid_user_cnt[cid]))
            for (feature, score) in resultMap[cid][:50]:
                print('%s\t%s' % (feature, score))
            print('--------------------------')

    return resultMap


def mutual_info_feature(cid_pattern_cnt, cid_user_cnt=None,
                        interested_cids=None):
    """
    this is a modified version of the original print_mutual_info,
    now each feature is associated with a score instead of a binary has/has not
    :type cid_pattern_cnt: Dict{int: Dict{str:float}}
          - reacord for each cluster the sum of each features
    :type cid_user_cnt: Dist{int:int} (default: None)
          - total number of users in each cluster
          - None means the script will compute it itself
    :type interested_cids: List{int} (default: None)
          - a list of cluster ids that we want to compute chi_square for
          - None means compute for all clusters
    :rtype: Dict{int: List[(str, float)]}
          - a map of cid and (feature, score) list
    """
    if not cid_user_cnt:
        cid_user_cnt = {}
        for cid in cid_pattern_cnt:
            # TODO: sum or max or avg pattern num * user num?
            cid_user_cnt[cid] = \
                sum([x[1] for x in cid_pattern_cnt[cid].items()])

    # compute a tmp dictionary:
    pattern_cid_cnt = {}
    for cid in cid_pattern_cnt:
        for w in cid_pattern_cnt[cid]:
            cnt = cid_pattern_cnt[cid][w]
            if w not in pattern_cid_cnt:
                pattern_cid_cnt[w] = {}
            pattern_cid_cnt[w][cid] = cnt

    total_pattern_cnt = {}
    for pattern in pattern_cid_cnt:
        total_pattern_cnt[pattern] = sum([x[1] for x in pattern_cid_cnt[pattern].items()])

    print 'how many patterns', len(pattern_cid_cnt)

    print 'mutual information...'

    """
    # I (feature word, class c):
    # A: class !=c and feature !=f  => the expected number of non-f features per user * total user not in c (*)
    # B: class =c and feature != f  => the expected number of non-f features per user * total user in c (*)
    # C: class != c and feature =f  => the expected number of feature f per user * total user not in c
    # D: class = c and feature = f  => the expected number of feature f per user * total user in c
    # N = total number of samples   => the total number of features (*)
    (*) - might need adjustment

    I =  D/N * log (N *D) / (BD CD) ) 
        + C/N * log (N *C) / (CD AC) )
        + B/N * log (N *B) / (AB BD) )
        + A/N * log (N *A) / (AB AC) )

    pre compute:
    total number of node clas: class_count
    total number of node per feature: feature_count
    total number of node per-class, per-feature: class_featurecount
    """

    N = totalFeatures = sum([x[1] for x in cid_user_cnt.items()])

    resultMap = {}
    for cid in cid_pattern_cnt:
        if interested_cids and cid not in interested_cids:
            continue
        # for each feature, compute the mutual information
        scores = {}
        for feature in cid_pattern_cnt[cid]:
            D = cid_pattern_cnt[cid][feature]
            B = cid_user_cnt[cid] - D
            C = total_pattern_cnt[feature] - D
            A = totalFeatures - C - D - B

            very_small = 1.0/N/1000000

            tmpD = 1.0*N*D/((B+D)*(C+D))+very_small
            tmpC = 1.0*N*C/((C+D)*(A+C))+very_small
            tmpB = 1.0*N*B/((A+B)*(B+D))+very_small
            tmpA = 1.0*N*A/((A+B)*(A+C))+very_small

            muinfo = (1.0*D/N) * math.log(tmpD, 2)
                   + (1.0*C/N) * math.log(tmpC, 2)
                   + (1.0*B/N) * math.log(tmpB, 2)
                   + (1.0*A/N) * math.log(tmpA, 2)

            scores[feature] = muinfo
        # fout.write('%s\t%s\n' % (cid, cid_user_cnt[cid]))
        resultMap[cid] = sorted(scores.items(),
                                key=lambda x: x[1], reverse=True)
        # fout.write('%s\t%s\n' % (feature, score))
        # fout.write('--------------------------\n')
    return resultMap

# if __name__ == '__main__':
#     # testing only
#     print(mutual_info_feature(cPickle.load(open('cid_pattern_cnt.pkl')),
#           cPickle.load(open('cid_user_cnt.pkl')),
#           cPickle.load(open('interested_cids.pkl'))))


def print_mutual_info(fname, cid_user_cnt, cid_pattern_cnt):
    """
    This is legacy code, not used. For reference only
    compute a tmp dictionary
    """
    pattern_cid_cnt = {}
    for cid in cid_pattern_cnt:
        for w in cid_pattern_cnt[cid]:
            cnt = cid_pattern_cnt[cid][w]
            if w not in pattern_cid_cnt: pattern_cid_cnt[w] = {}
            pattern_cid_cnt[w][cid] = cnt

    print 'how many patterns', len(pattern_cid_cnt)

    print 'mutual information...'
    # mutual information
    """
    # I (feature word, class c):
    # A: class !=c and feature !=f
    # B: class =c and feature != f
    # C: class != c and feature =f
    # D: class = c and feature = f
    # N = total number of samples

    I =  D/N * log (N *D) / (BD CD) ) 
        + C/N * log (N *C) / (CD AC) )
        + B/N * log (N *B) / (AB BD) )
        + A/N * log (N *A) / (AB AC) )
    
    pre compute:
    total number of node clas: class_count
    total number of node per feature: feature_count
    total number of node per-class, per-feature: class_featurecount
    """
    # total node
    N = 0
    for cid in cid_user_cnt:
        N+= cid_user_cnt[cid]

    print 'total N', N
    print 'how many class?', len(cid_user_cnt)

    fo = open(fname,'w')
    for cid in cid_pattern_cnt:
        w_mutual = {}
        w_D = {}
        w_C = {}

        for w in cid_pattern_cnt[cid]:
            cnt = cid_pattern_cnt[cid][w]
            # start to compute mutual info:
            # D: class = c and feature = f
            D = cnt
            # B: class =c and feature != f
            B = cid_user_cnt[cid] - cnt
            # C: class != c and feature =f
            C = 0
            for kid in pattern_cid_cnt[w]:
                if kid == cid: continue
                C += pattern_cid_cnt[w][kid]
            # A: class !=c and feature !=f
            A = N-(C+B+D)
            very_small = 1.0/N/1000000
            tmpD = 1.0*N*D/((B+D)*(C+D))+very_small
            tmpC = 1.0*N*C/((C+D)*(A+C))+very_small
            tmpB = 1.0*N*B/((A+B)*(B+D))+very_small
            tmpA = 1.0*N*A/((A+B)*(A+C))+very_small
            muinfo = (1.0*D/N) * math.log(tmpD,2) + (1.0*C/N) * math.log(tmpC,2) + (1.0*B/N) * math.log(tmpB,2) + (1.0*A/N) * math.log(tmpA,2)

            if (1.0*D/N) * math.log(tmpD,2) > 0:
                w_mutual[w] = muinfo
                w_D[w] = D
                w_C[w] = C

            # debug:
            """
            if str(cid)=='4' and (w=='A8A6A' or w=='A5A'):
                print '========== debug cid=4, pattern=', w, 'mutual=', muinfo
                print 'D', (1.0*D/N) * math.log(tmpD,2)
                print 'C', (1.0*C/N) * math.log(tmpC,2)
                print 'B', (1.0*B/N) * math.log(tmpB,2)
                print 'A', (1.0*A/N) * math.log(tmpA,2)
            """

        sortlist = sortlist = sorted(w_mutual.iteritems(), key=lambda (k,v): (v,k))
        sortlist.reverse()
        tmpcnt = 0
        # for understanding purpuse, let's print D and C as well
        fo.write('===== cid:%s, total users:%s ===================== \n' %(cid, cid_user_cnt[cid]))
        fo.write('format: pattern, mutual, D: class = c and feature =f, C: class != c and feature =f\n')
        for w, cnt in sortlist:
            D = w_D[w]
            C = w_C[w]
            fo.write('##%s\t%s\t%s\t%s\t%s\n' %(cid, w, cnt, D, C))
            tmpcnt += 1
            if tmpcnt>10: break
        print 'done for cluster', cid
    fo.close()

    return

