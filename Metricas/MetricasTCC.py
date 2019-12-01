import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.stats import fisher_exact as fisher
from scipy.cluster.hierarchy import dendrogram, linkage, distance, fcluster
from math import sqrt
from math import log10
from math import log
from itertools import combinations
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import scipy.cluster.hierarchy as sch

class raMetricas:
    @staticmethod
    def __getCol(db, itemset, not1=False):
        teste = np.array(itemset)
        teste = teste - 1
        mtz = db[:,  teste]
        if len(mtz.shape) == 2:
            mtz = mtz.all(axis=1)
        if not1:
            mtz = np.invert(mtz)
        return mtz

    @staticmethod
    def __intersect(db, itemset1, itemset2=None, not1=False, not2=False):
        itemsets = raMetricas.__getCol(db, itemset1, not1=not1)
        if itemset2 is not None:
            it2 = raMetricas.__getCol(db, itemset2, not1=not2)
            itemsets = np.vstack((itemsets, it2))
            itemsets = itemsets.all(axis=0)
        return itemsets


    @staticmethod
    def abSupp(db, itemset1, itemset2=None, not1=False, not2=False):
        itemsets = raMetricas.__intersect(db, itemset1, itemset2, not1, not2)
        return np.count_nonzero(itemsets)

    @staticmethod
    def relSupp(db, itemset1, itemset2=None, not1=False, not2=False):
        return raMetricas.abSupp(db, itemset1, itemset2, not1, not2) / db.shape[0]

    @staticmethod
    def subSupp(db, itemset):
        # Calcula o suporte absoluto de todos os 1-itemsets contidos em itemset.
        subSup = db[:, np.array(itemset) - 1]
        subSup = np.sum(subSup, axis=0)
        return subSup

    @staticmethod
    def conf(db, antc, conq, notA=False, notC=False):
        suporteRA = raMetricas.abSupp(db, antc, conq, not1=notA, not2=notC)
        suporteAntc = raMetricas.abSupp(db, antc, not1=notA)
        return suporteRA / suporteAntc

    @staticmethod
    def addedValue(db, antc, conq):
        return raMetricas.conf(db, antc, conq) - raMetricas.relSupp(db, conq)

    @staticmethod
    def allConf(db, itemset):
        max = raMetricas.subSupp(db, itemset)
        max = np.max(max)
        max /= db.shape[0]
        return raMetricas.relSupp(db, itemset) / max

    @staticmethod
    def causalConf(db, antc, conq):
        f1x = raMetricas.abSupp(db, antc)
        fx1 = raMetricas.abSupp(db, conq)
        f10 = raMetricas.abSupp(db, antc, conq, not2=True)
        return 1 - f10/db.shape[0] * (1/f1x + 1/fx1)

    @staticmethod
    def casualConf2(db, antc, conq):
        conf1 = raMetricas.conf(db, antc, conq)
        conf2 = raMetricas.conf(db, antc, conq, notA=True, notC=True)
        return (conf1 + conf2) / 2

    @staticmethod
    def causalSupp(db, antc, conq):
        f1x = raMetricas.abSupp(db, antc)
        fx1 = raMetricas.abSupp(db, conq)
        f10 = raMetricas.abSupp(db, antc, conq, not2=True)
        return (f1x + fx1 - 2 * f10) / db.shape[0]

    @staticmethod
    def causalSupp2(db, antc, conq):
        t1 = raMetricas.relSupp(db, antc, conq)
        t2 = raMetricas.relSupp(db, antc, conq, not1=True, not2=True)
        return t1 + t2

    @staticmethod
    def certFactor(db, antc, conq):
        cf = raMetricas.conf(db, antc, conq) - raMetricas.relSupp(db, conq)
        return cf / raMetricas.relSupp(db, conq, not1=True)

    @staticmethod
    def __tbContingencia(db, antc, conq):
        n = db.shape[0]
        c11 = raMetricas.abSupp(db, antc, conq)
        c1x = raMetricas.abSupp(db, antc)
        cx1 = raMetricas.abSupp(db, conq)
        c10 = c1x - c11
        c01 = cx1 - c11
        cx0 = n - cx1
        c0x = n - c1x
        c00 = c0x - c01

        obs = np.array([[c11, c01], [c10, c00]])
        exp = np.outer([cx1, cx0], [c1x, c0x]) / n
        return (obs, exp)

    @staticmethod
    def chiSqrd(db, antc, conq):
        obs, exp = raMetricas.__tbContingencia(db, antc, conq)

        result = obs - exp
        result *= result
        result /= exp
        result = np.sum(result)
        return result

    @staticmethod
    def crossSuppRatio(db, itemset):
        subSup = raMetricas.subSupp(db, itemset)
        return np.min(subSup) / np.max(subSup)

    @staticmethod
    def collectiveStrength(db, antc, consq):
        n = db.shape[0]
        f11 = raMetricas.abSupp(db, antc, consq)
        f1x = raMetricas.abSupp(db, antc)
        fx1 = raMetricas.abSupp(db, consq)
        fx0 = n - fx1
        f0x = n - f1x
        f00 = n - f1x - fx1 + f11
        t1 = (f1x * fx1 + f0x + fx0)
        t2 = (n-f11-f00)
        if (t1 == 0) or (t2==0):
            return float("NaN")
        return f11*f00/t1 * (n**2 -f1x*fx1-f0x*fx0)/t2

    @staticmethod
    def conviction(db, antc, cons):
        div = (1 - raMetricas.conf(db, antc, cons))
        if div == 0:
            return float("NaN")
        return (1 - raMetricas.relSupp(db, cons)) / div

    @staticmethod
    def cosine(db, antc, cons):
        return sqrt(raMetricas.conf(db, antc, cons) * raMetricas.conf(db, cons, antc))

    @staticmethod
    def coverage(db, antc, cons):
        return raMetricas.relSupp(db, antc)

    @staticmethod
    def descCfConf2(db, antc, cons):
        conf1 = raMetricas.conf(db, antc, cons)
        conf2 = raMetricas.conf(db, antc, cons, notC=True)
        return conf1 - conf2

    @staticmethod
    def descCfConf(db, antc, cons):
        c1x = raMetricas.abSupp(db, antc)
        c10 = raMetricas.abSupp(db, antc, cons, not2=True)
        return (c1x-2*c10)/db.shape[0]

    @staticmethod
    def differenceOfConfidence(db, antc, cons):
        return raMetricas.conf(db, antc, cons) - raMetricas.conf(db, antc, cons, notA=True)

    @staticmethod
    def exCounterEx(db, antc, conq):
        suporte1 = raMetricas.abSupp(db, antc, conq)
        suporte2 = raMetricas.abSupp(db, antc, conq, not2=True)
        return (suporte1 - suporte2) / suporte1

    @staticmethod
    def fischers(db, antc, cons):
        obs = raMetricas.__tbContingencia(db, antc, cons)[0]
        return fisher(obs)[1]

    @staticmethod
    def giniIndex(db, antc, conq):
        t1 = raMetricas.relSupp(db, antc)
        t1 *= raMetricas.conf(db, antc, conq)**2 + raMetricas.conf(db, antc, conq, notC=True)**2
        t1 -= raMetricas.relSupp(db, conq)**2
        t2 = raMetricas.relSupp(db, antc, not1=True)
        t2 *= raMetricas.conf(db, antc, conq, notA=True)**2 + raMetricas.conf(db, antc, conq, notA=True, notC=True)**2
        t2 -= raMetricas.relSupp(db, conq, not1=True)**2
        return t1 + t2


    @staticmethod
    def hyperConfidence(db, antc, consq):
        total = db.shape[0]
        cxy = raMetricas.abSupp(db, antc, consq)
        cx = raMetricas.abSupp(db, antc)
        cy = raMetricas.abSupp(db, consq)
        result = hypergeom.cdf(k=cxy -1, M=total, n=cy,  N=cx)
        return result

    @staticmethod
    def hyperLift(db, antc, consq):
        total = db.shape[0]
        cxy = raMetricas.abSupp(db, antc, consq)
        cx = raMetricas.abSupp(db, antc)
        cy = raMetricas.abSupp(db, consq)

        q = hypergeom.ppf(q=.99, M=total, n=cy,  N=cx)
        return cxy/q

    @staticmethod
    def imbalanceRatio(db, antc, conq):
        suporteA = raMetricas.relSupp(db, antc)
        suporteC = raMetricas.relSupp(db, conq)
        termo1 = abs(suporteA - suporteC)
        termo2 = suporteA + suporteC - raMetricas.relSupp(db, antc, conq)
        return termo1 / termo2

    @staticmethod
    def importance(db, antc, cons):
        # implementação do arules diferente da formula
        return log10(raMetricas.laplaceConf(db, antc, cons) / raMetricas.laplaceConf(db, antc, cons, not1=True))

    @staticmethod
    def improvement(db, antc, cons):
        confRule = raMetricas.conf(db, antc, cons)
        if len(antc) <= 1:
            return 0
        improvement = lambda base, cr, a, c: (cr - raMetricas.conf(base, a, c))
        subAntcs = [list(j) for i in range(1, len(antc)) for j in combinations(antc, i)]
        return min([improvement(db, confRule, subAntc, cons) for subAntc in subAntcs])

    @staticmethod
    def jaccardCoefficient(db, antc, cons):
        termo1 = raMetricas.relSupp(db, antc, cons)
        termo2 = raMetricas.relSupp(db, antc) + raMetricas.relSupp(db, cons) - raMetricas.relSupp(db, antc, cons)

        return termo1 / termo2

    @staticmethod
    def Jmeasure(db, antc, cons):
        conf = raMetricas.conf(db, antc, cons) #0.875
        fx1 = raMetricas.relSupp(db, cons)     # 0.4
        f1x = raMetricas.relSupp(db, antc)     # 0.45

        if 0 in [conf, f1x, fx1]:
            return float("NaN")

        t1 = 1 - conf
        t2 = conf/fx1
        t3 = 1 - fx1

        if 0 in [t1, t2, t3]:
            return float("NaN")
        return f1x * (conf * log(t2) + (t1) * log( (t1) / (t3) ))


    @staticmethod
    def kappa(db, antc, cons):
        n = db.shape[0]
        f11 = raMetricas.relSupp(db, antc, cons)
        f00 = raMetricas.relSupp(db, antc, cons, not1=True, not2=True)
        f1x = raMetricas.relSupp(db, antc)
        f0x = raMetricas.relSupp(db, antc, not1=True)
        fx1 = raMetricas.relSupp(db, cons)
        fx0 = raMetricas.relSupp(db, cons, not1=True, not2=True)
        return (f11 + f00 - f1x * fx1 - f0x * fx0) / (1 - f1x * fx1 - f0x * fx0)


    @staticmethod
    def klosgen(db, antc, conq):
        t1 = sqrt(raMetricas.relSupp(db, antc, conq))
        t2 = (raMetricas.conf(db, antc, conq) - raMetricas.relSupp(db, conq))
        return t1 * t2

    @staticmethod
    def kulczynski(db, antc, conq):
        return (raMetricas.conf(db, antc, conq) + raMetricas.conf(db, conq, antc)) / 2

    @staticmethod
    def predictiveAssociation(db, antc, consq):
        fx1 = raMetricas.abSupp(db, consq)
        fx0 = raMetricas.abSupp(db, consq, not1=True)
        f11 = raMetricas.abSupp(db, antc, consq)
        f10 = raMetricas.abSupp(db, antc, consq, not2=True)
        f01 = raMetricas.abSupp(db, antc, consq, not1=True)
        f00 = raMetricas.abSupp(db, antc, consq, not1=True, not2=True)
        x0x1 = max(fx1, fx0)

        return (max(f11, f10) + max(f01, f00) - x0x1) / (db.shape[0] - x0x1)

    @staticmethod
    def laplaceConf(db, antc, cons, not1=False, not2=False):
        return (raMetricas.abSupp(db, antc, cons, not1=not1, not2=not2) + 1) / (raMetricas.abSupp(db, antc, not1=not1) + 2)

    @staticmethod
    def lermanSimilarity(db, antc, cons):
        t1 = sqrt(db.shape[0])
        t2 = raMetricas.relSupp(db, antc) * raMetricas.relSupp(db, cons)
        return (raMetricas.relSupp(db, antc, cons) - t2 / sqrt(t2)) * t1

    @staticmethod
    def leastContradiction(db, antc, conq):
        return raMetricas.conf(db, conq, antc)

    @staticmethod
    def leastContradiction2(db, antc, conq):
        termo1 = raMetricas.relSupp(db, antc, conq) - raMetricas.relSupp(db, antc, conq, not2=True)
        termo2 = raMetricas.relSupp(db, conq)
        return termo1 / termo2

    @staticmethod
    def leverage(db, antc, conq):
        t1 = raMetricas.relSupp(db, antc, conq)
        t2 = raMetricas.relSupp(db, antc) * raMetricas.relSupp(db, conq)
        return t1 - t2

    @staticmethod
    def lift(db, antc, conq):
        t1 = raMetricas.conf(db, antc, conq)
        t2 = raMetricas.relSupp(db, conq)
        return t1 / t2

    @staticmethod
    def maxConf(db, antc, conq):
        return max(raMetricas.conf(db, antc, conq), raMetricas.conf(db, conq, antc))

    @staticmethod
    def mutualInformation(db, antc, consq):
        n = db.shape[0]
        f11 = raMetricas.abSupp(db, antc, consq)
        f1x = raMetricas.abSupp(db, antc)
        fx1 = raMetricas.abSupp(db, consq)
        f10 = f1x - f11
        f01 = fx1 - f11
        f00 = n - f11 - f10 - f01
        fx0 = n - fx1
        f0x = n - f1x

        I = lambda n, i, j, k: i * (float("NaN") if 0 in [i, n, j, k] else log(i*n / (j*k)))

        a = I(n, f00, f0x, fx0) + I(n, f01, f0x, fx1)
        a += I(n, f10, f1x, fx0) + I(n, f11, f1x, fx1)
        b = -1 * (f0x / n * log(f0x / n) + f1x / n * log(f1x / n))
        b = min(b, -1 * (fx0 / n * log(fx0 / n) + fx1 / n * log(fx1 / n)) )
        return a / b

    @staticmethod
    def mutualInformation3(db, antc, cons):
        I = lambda i: i * log10(i) + (1-i) * log10(1-i)
        return I(raMetricas.relSupp(db, cons)) - I(raMetricas.relSupp(db, antc, cons))

    @staticmethod
    def CorrelationCoeficient(db, antc, conq):
        a = raMetricas.relSupp(db, antc)
        c = raMetricas.relSupp(db, conq)
        return (raMetricas.relSupp(db, antc, conq) - a * c) / sqrt(a * c * (1-a) * (1-c))

    @staticmethod
    def oddsRatio(db, antc, cons):
        c11 = raMetricas.abSupp(db, antc, cons) # 2
        c00 = raMetricas.abSupp(db, antc, cons, not1=True, not2=True) # 11
        c10 = raMetricas.abSupp(db, antc, cons, not2=True) # 1
        c01 = raMetricas.abSupp(db, antc, cons, not1=True) # 1

        t2 = c10 * c01
        if t2 == 0:
            return float("NaN")
        return (c11*c00)/t2

    @staticmethod
    def phi(db, antc, cons):
        f11 = raMetricas.relSupp(db, antc, cons)
        f1x = raMetricas.relSupp(db, antc)
        fx1 = raMetricas.relSupp(db, cons)
        f0x = 1 - f1x
        fx0 = 1 - fx1

        return (f11-f1x*fx1) / sqrt(f1x*fx1*f0x*fx0)

    @staticmethod
    def ralambondrainyMeasure(db, antc, conq):
        return raMetricas.relSupp(db, antc, conq, not2=True)

    @staticmethod
    def RLD(db, antc, conq):

        n = db.shape[0]
        c11 = raMetricas.abSupp(db, antc, conq)
        c1x = raMetricas.abSupp(db, antc)
        cx1 = raMetricas.abSupp(db, conq)
        c10 = c1x - c11
        c01 = cx1 - c11
        c00 = n - c11 - c10 - c01

        d = (c11 * c00 - c10 * c01) / n

        if d > 0:
            if c01 < c10:
                return d / (d + c01)
            else:
                return d / (d + c10)
        else:
            if c11 < c00:
                return d / (d - c11)
            else:
                return d / (d - c00)



    @staticmethod
    def rulePF(db, antc, conq):
        return raMetricas.relSupp(db, antc, conq) * raMetricas.conf(db, antc, conq)

    @staticmethod
    def sebagSchoenauerMeasure(db, antc, conq):
        t1 = raMetricas.relSupp(db, antc, conq)
        t2 = raMetricas.relSupp(db, antc, conq, not2=True)
        if t2 == 0.0: return float("NaN")
        return t1 / t2

    @staticmethod
    def varyingRatesLiaison(db, antc, cons):
        return raMetricas.lift(db, antc, cons) - 1

    @staticmethod
    def yulesQ(db, antc, cons):
        t1 = raMetricas.oddsRatio(db, antc, cons) - 1
        t2 = raMetricas.oddsRatio(db, antc, cons) + 1
        return t1 / t2

    @staticmethod
    def yulesY(db, antc, cons):
        t1 = sqrt(raMetricas.oddsRatio(db, antc, cons)) - 1
        t2 = sqrt(raMetricas.oddsRatio(db, antc, cons)) + 1
        return t1 / t2



def tabelaMetricas():

    metricas = "metricasRegras.dat"
    baseDados = "BPressureNishiBook.dat"
    baseRegras = "BPressureNishiBook.txt"

    rules = pd.read_csv(metricas, sep=" ")
    rules["antc"], rules["consq"] = rules["regras"].str.replace("{", "").str.replace("}", "").str.split("=>").str
    rules["antc"] = rules["antc"].str.split(",").apply(lambda x: np.array(list(map(int, x))))
    rules['consq'] = rules['consq'].str.replace(' ', '').str.split()
    rules['consq'] = rules['consq'].apply(lambda x: list(map(int, x)))
    del rules["regras"]

    mt = pd.read_csv(baseDados, sep=" ", dtype="str", header=None)

    mtBinaria = pd.get_dummies(mt).astype('bool')
    dados = mtBinaria.to_numpy()

    df = pd.DataFrame()
    df['antc'] = rules["antc"]
    df['consq'] = rules['consq']
    df['Support  1'] = rules.apply(lambda x: raMetricas.relSupp(dados, x['antc'], x['consq']), axis=1)
    df['count  1'] = rules.apply(lambda x: raMetricas.abSupp(dados, x['antc'], x['consq']), axis=1)
    df['Confidence  2'] = rules.apply(lambda x: raMetricas.conf(dados, x['antc'], x['consq']), axis=1)
    df['Added Value  3'] = rules.apply(lambda x: raMetricas.addedValue(dados, x['antc'], x['consq']), axis=1)
    #All-Confidence
    df['Causal Support  4'] = rules.apply(lambda x: raMetricas.causalSupp(dados, x['antc'], x['consq']), axis=1)
    df['Causal Confidence  5'] = rules.apply(lambda x: raMetricas.causalConf(dados, x['antc'], x['consq']), axis=1)
    df['Certainty Factor  6'] = rules.apply(lambda x: raMetricas.certFactor(dados, x['antc'], x['consq']), axis=1)
    df['Chi-Squared  7'] = rules.apply(lambda x: raMetricas.chiSqrd(dados, x['antc'], x['consq']), axis=1)
    #Cross Support Ratio
    df['Collective Strength  8'] = rules.apply(lambda x: raMetricas.collectiveStrength(dados, x['antc'], x['consq']), axis=1)
    df['Conviction  9'] = rules.apply(lambda x: raMetricas.conviction(dados, x['antc'], x['consq']), axis=1)
    df['Cosine 10'] = rules.apply(lambda x: raMetricas.cosine(dados, x['antc'], x['consq']), axis=1)
    df['Coverage 11'] = rules.apply(lambda x: raMetricas.coverage(dados, x['antc'], x['consq']), axis=1)
    df['Confirmed Confidence 12'] = rules.apply(lambda x: raMetricas.descCfConf(dados, x['antc'], x['consq']), axis=1)
    df['Difference of Confidence 13'] = rules.apply(lambda x: raMetricas.differenceOfConfidence(dados, x['antc'], x['consq']), axis=1)
    df['Counter Example 14'] = rules.apply(lambda x: raMetricas.exCounterEx(dados, x['antc'], x['consq']), axis=1)
    df['Fisher\'s Exact Test 15'] = rules.apply(lambda x: raMetricas.fischers(dados, x['antc'], x['consq']), axis=1)
    df['Gini Index 16'] = rules.apply(lambda x: raMetricas.giniIndex(dados, x['antc'], x['consq']), axis=1)
    df['Hyper Confidence 17'] = rules.apply(lambda x: raMetricas.hyperConfidence(dados, x['antc'], x['consq']), axis=1)
    df['Hyper Lift 18'] = rules.apply(lambda x: raMetricas.hyperLift(dados, x['antc'], x['consq']), axis=1)
    df['Imbalance Ratio 19'] = rules.apply(lambda x: raMetricas.imbalanceRatio(dados, x['antc'], x['consq']), axis=1)
    df['Importance 20'] = rules.apply(lambda x: raMetricas.importance(dados, x['antc'], x['consq']), axis=1)
    df['Improvement 21'] = rules.apply(lambda x: raMetricas.improvement(dados, x['antc'], x['consq']), axis=1)
    df['Jaccard Coefficient 22'] = rules.apply(lambda x: raMetricas.jaccardCoefficient(dados, x['antc'], x['consq']), axis=1)
    df['J-Measure 23'] = rules.apply(lambda x: raMetricas.Jmeasure(dados, x['antc'], x['consq']), axis=1)
    df['Kappa 24'] = rules.apply(lambda x: raMetricas.kappa(dados, x['antc'], x['consq']), axis=1)
    df['Klosgen 25'] = rules.apply(lambda x: raMetricas.klosgen(dados, x['antc'], x['consq']), axis=1)
    df['Kulczynski 26'] = rules.apply(lambda x: raMetricas.kulczynski(dados, x['antc'], x['consq']), axis=1)
    df['Goodman-Kruskal 27'] = rules.apply(lambda x: raMetricas.predictiveAssociation(dados, x['antc'], x['consq']), axis=1)
    df['Laplace Confidence 28'] = rules.apply(lambda x: raMetricas.laplaceConf(dados, x['antc'], x['consq']), axis=1)
    df['Least Contradiction 29'] = rules.apply(lambda x: raMetricas.leastContradiction(dados, x['antc'], x['consq']), axis=1)
    df['Lerman Similarity 30'] = rules.apply(lambda x: raMetricas.lermanSimilarity(dados, x['antc'], x['consq']), axis=1)
    df['Leverage 31'] = rules.apply(lambda x: raMetricas.leverage(dados, x['antc'], x['consq']), axis=1)
    df['Lift 32'] = rules.apply(lambda x: raMetricas.lift(dados, x['antc'], x['consq']), axis=1)
    df['Max Confidence 33'] = rules.apply(lambda x: raMetricas.maxConf(dados, x['antc'], x['consq']), axis=1)
    df['Mutual Information 34'] = rules.apply(lambda x: raMetricas.mutualInformation(dados, x['antc'], x['consq']), axis=1)
    df['Odds Ratio 35'] = rules.apply(lambda x: raMetricas.oddsRatio(dados, x['antc'], x['consq']), axis=1)
    df['Phi Correlation Coeficient 36'] = rules.apply(lambda x: raMetricas.phi(dados, x['antc'], x['consq']), axis=1)
    df['Ralambondrainy Measure 37'] = rules.apply(lambda x: raMetricas.ralambondrainyMeasure(dados, x['antc'], x['consq']), axis=1)
    df['Relative Linkage Disequilibrium 38'] = rules.apply(lambda x: raMetricas.RLD(dados, x['antc'], x['consq']), axis=1)
    df['Rule Power Factor 39'] = rules.apply(lambda x: raMetricas.rulePF(dados, x['antc'], x['consq']), axis=1)
    df['Sebag-Schoenauer Measure 40'] = rules.apply(lambda x: raMetricas.sebagSchoenauerMeasure(dados, x['antc'], x['consq']), axis=1)
    df['Varying Rates Liaison 41'] = rules.apply(lambda x: raMetricas.varyingRatesLiaison(dados, x['antc'], x['consq']), axis=1)
    df['Yule\'s Q 42'] = rules.apply(lambda x: raMetricas.yulesQ(dados, x['antc'], x['consq']), axis=1)
    df['Yule\'s Y 43'] = rules.apply(lambda x: raMetricas.yulesY(dados, x['antc'], x['consq']), axis=1)

    df = df.round(decimals=5)
    df.to_csv('MetricasTCC.csv')
    return df


def PlotarGraficos(df):
    # matriz de correlacao
    del df['count  1']
    df = df.loc[:, 'Support  1':].corr(method='pearson')

    # mascara para heatmap
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # cor
    cmap = sns.diverging_palette(10, 220, as_cmap=True)


    sns.heatmap(df, xticklabels=True, yticklabels=True, cmap=cmap, mask=mask, square=True)


    y = df.values
    L = sch.complete(y)
    ind = sch.fcluster(L, 0.5 * y.max(), criterion='distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=0).reindex(columns, axis=1)

    unique, counts = np.unique(ind, return_counts=True)
    counts = dict(zip(unique, counts))

    sns.heatmap(df, xticklabels=True, yticklabels=True, cmap=cmap, mask=mask, square=True)
    plt.tick_params(labelsize=7)
    plt.show()
    plt.clf()

    i = 0
    j = 0
    columns = []

    cluster_th = 4

    for cluster_l1 in set(sorted(ind)):
        j += counts[cluster_l1]
        sub = df[df.columns.values[i:j]]
        if counts[cluster_l1] > cluster_th:
            y = sub.corr().values
            L = sch.complete(y)
            ind = sch.fcluster(L, 0.5 * y.max(), 'distance')
            col = [sub.columns.tolist()[i] for i in list((np.argsort(ind)))]
            sub = sub.reindex(col, axis=1)
        cols = sub.columns.tolist()
        columns.extend(cols)
        i = j
    df2 = df.reindex(columns, axis=0).reindex(columns, axis=1)

    sns.heatmap(df2, xticklabels=True, yticklabels=True, cmap=cmap, mask=mask, square=True)

    print('q para sair, qualquer tecla para continuar')
    for i in df.loc[:, 'Support  1':].corr(method='pearson'):
        data = pd.DataFrame(data=df[i].abs())
        data['Correlação Positiva'] = np.where(df[i] >= 0, df[i], 0)
        data['Correlação Negativa'] = np.where(df[i] < 0, np.abs(df[i]), 0)

        data = data.sort_values(by=[i], ascending=False)
        data = data.drop(columns=i, index=i)

        vermelho = cmap(-0.2)
        azul = cmap(0.8)
        data = data.head(30)
        ax = data.plot(kind='bar', title=i, color=[azul, vermelho], stacked=True)
        ax.set_ylim(0, 1)

        plt.show()
        plt.close()
        del data
        if input() == 'q':
            break


if __name__ == "__main__":
    df = tabelaMetricas()
    PlotarGraficos(df)