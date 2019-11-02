import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.stats import chi2_contingency
from mpl_toolkits import mplot3d
from scipy.stats import fisher_exact as fisher
from math import sqrt
from math import log10
from math import log
from itertools import combinations
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



metricas = "metricasRegras.dat"
baseDados = "BPressureNishiBook.dat"
baseRegras = "BPressureNishiBook.txt"


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
    def casualConf(db, antc, conq):
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
    def casualSupp(db, antc, conq):
        f1x = raMetricas.abSupp(db, antc)
        fx1 = raMetricas.abSupp(db, conq)
        f10 = raMetricas.abSupp(db, antc, conq, not2=True)
        return (f1x + fx1 - 2 * f10) / db.shape[0]

    @staticmethod
    def casualSupp2(db, antc, conq):
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
        if (len(antc) == 2) and (antc[0] == 8) and (antc[1] == 13) and (cons[0] == 5):
            print()
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



rules = pd.read_csv(metricas, sep=" ")
rules["antc"], rules["consq"] = rules["regras"].str.replace("{", "").str.replace("}", "").str.split("=>").str
rules["antc"] = rules["antc"].str.split(",").apply(lambda x: np.array(list(map(int, x))))

rules['consq'] = rules['consq'].str.replace(' ', '').str.split()
rules['consq'] = rules['consq'].apply(lambda x: list(map(int, x)))
del rules["regras"]

mt = pd.read_csv(baseDados, sep=" ", dtype="str", header=None)

mtBinaria = pd.get_dummies(mt).astype('bool')
#mtBinaria.transpose
dados = mtBinaria.to_numpy()

rules2 = pd.DataFrame()
rules2['antc'] = rules["antc"]
rules2['consq'] = rules['consq']

rules2['support'] = rules.apply(lambda x: raMetricas.relSupp(dados, x['antc'], x['consq']), axis=1)
rules2['count'] = rules.apply(lambda x: raMetricas.abSupp(dados, x['antc'], x['consq']), axis=1)
rules2['coverage'] = rules.apply(lambda x: raMetricas.coverage(dados, x['antc'], x['consq']), axis=1)
rules2['confidence'] = rules.apply(lambda x: raMetricas.conf(dados, x['antc'], x['consq']), axis=1)
rules2['lift'] = rules.apply(lambda x: raMetricas.lift(dados, x['antc'], x['consq']), axis=1)
rules2['leverage'] = rules.apply(lambda x: raMetricas.leverage(dados, x['antc'], x['consq']), axis=1)
rules2['fishersExactTest'] = rules.apply(lambda x: raMetricas.fischers(dados, x['antc'], x['consq']), axis=1)
rules2['improvement'] = rules.apply(lambda x: raMetricas.improvement(dados, x['antc'], x['consq']), axis=1)
rules2['chiSquared'] = rules.apply(lambda x: raMetricas.chiSqrd(dados, x['antc'], x['consq']), axis=1)
rules2['cosine'] = rules.apply(lambda x: raMetricas.cosine(dados, x['antc'], x['consq']), axis=1)
rules2['conviction'] = rules.apply(lambda x: raMetricas.conviction(dados, x['antc'], x['consq']), axis=1)
rules2['gini'] = rules.apply(lambda x: raMetricas.giniIndex(dados, x['antc'], x['consq']), axis=1)
rules2['hyperConfidence'] = rules.apply(lambda x: raMetricas.hyperConfidence(dados, x['antc'], x['consq']), axis=1)
rules2['hyperLift'] = rules.apply(lambda x: raMetricas.hyperLift(dados, x['antc'], x['consq']), axis=1)
rules2['oddsRatio'] = rules.apply(lambda x: raMetricas.oddsRatio(dados, x['antc'], x['consq']), axis=1)
rules2['phi'] = rules.apply(lambda x: raMetricas.phi(dados, x['antc'], x['consq']), axis=1)
rules2['doc'] = rules.apply(lambda x: raMetricas.differenceOfConfidence(dados, x['antc'], x['consq']), axis=1)
rules2['RLD'] = rules.apply(lambda x: raMetricas.RLD(dados, x['antc'], x['consq']), axis=1)
rules2['imbalance'] = rules.apply(lambda x: raMetricas.imbalanceRatio(dados, x['antc'], x['consq']), axis=1)
rules2['kulczynski'] = rules.apply(lambda x: raMetricas.kulczynski(dados, x['antc'], x['consq']), axis=1)
rules2['lambda'] = rules.apply(lambda x: raMetricas.predictiveAssociation(dados, x['antc'], x['consq']), axis=1)
rules2['collectiveStrength'] = rules.apply(lambda x: raMetricas.collectiveStrength(dados, x['antc'], x['consq']), axis=1)
rules2['jaccard'] = rules.apply(lambda x: raMetricas.jaccardCoefficient(dados, x['antc'], x['consq']), axis=1)
rules2['kappa'] = rules.apply(lambda x: raMetricas.kappa(dados, x['antc'], x['consq']), axis=1)
rules2['mutualInformation'] = rules.apply(lambda x: raMetricas.mutualInformation(dados, x['antc'], x['consq']), axis=1)
rules2['jMeasure'] = rules.apply(lambda x: raMetricas.Jmeasure(dados, x['antc'], x['consq']), axis=1)
rules2['laplace'] = rules.apply(lambda x: raMetricas.laplaceConf(dados, x['antc'], x['consq']), axis=1)
rules2['certainty'] = rules.apply(lambda x: raMetricas.certFactor(dados, x['antc'], x['consq']), axis=1)
rules2['addedValue'] = rules.apply(lambda x: raMetricas.addedValue(dados, x['antc'], x['consq']), axis=1)
rules2['maxconfidence'] = rules.apply(lambda x: raMetricas.maxConf(dados, x['antc'], x['consq']), axis=1)
rules2['rulePowerFactor'] = rules.apply(lambda x: raMetricas.rulePF(dados, x['antc'], x['consq']), axis=1)
rules2['ralambondrainy'] = rules.apply(lambda x: raMetricas.ralambondrainyMeasure(dados, x['antc'], x['consq']), axis=1)
rules2['descriptiveConfirm'] = rules.apply(lambda x: raMetricas.descCfConf(dados, x['antc'], x['consq']), axis=1)
rules2['sebag'] = rules.apply(lambda x: raMetricas.sebagSchoenauerMeasure(dados, x['antc'], x['consq']), axis=1)
rules2['counterexample'] = rules.apply(lambda x: raMetricas.exCounterEx(dados, x['antc'], x['consq']), axis=1)
rules2['casualSupport'] = rules.apply(lambda x: raMetricas.casualSupp(dados, x['antc'], x['consq']), axis=1)
rules2['casualConfidence'] = rules.apply(lambda x: raMetricas.casualConf(dados, x['antc'], x['consq']), axis=1)
rules2['leastContradiction'] = rules.apply(lambda x: raMetricas.leastContradiction(dados, x['antc'], x['consq']), axis=1)
rules2['varyingLiaison'] = rules.apply(lambda x: raMetricas.varyingRatesLiaison(dados, x['antc'], x['consq']), axis=1)
rules2['yuleQ'] = rules.apply(lambda x: raMetricas.yulesQ(dados, x['antc'], x['consq']), axis=1)
rules2['yuleY'] = rules.apply(lambda x: raMetricas.yulesY(dados, x['antc'], x['consq']), axis=1)
rules2['importance'] = rules.apply(lambda x: raMetricas.importance(dados, x['antc'], x['consq']), axis=1)


rules2 = rules2.round(decimals=5)
rules = rules.round(decimals=5)


def metricas_inspect(discrepancias, df1, df2):
    hashMap = {}
    for i in discrepancias:
        hashMap[i] = pd.DataFrame()
        #hashMap[i]['antc'] = df1['antc']
        #hashMap[i]['consq'] = df1['consq']
        hashMap[i]['Sup Regra'] = df1['support']
        hashMap[i]['hahsler'] = df1[i].round(decimals=5)
        hashMap[i]['TCC'] = df2[i].round(decimals=5)
        hashMap[i]['Difference'] = hashMap[i]['TCC'] - hashMap[i]['hahsler']
        hashMap[i]['Proportion'] = hashMap[i]['TCC'] / hashMap[i]['hahsler'] - 1
    return hashMap


def comparaMetricas(arules, tcc):
    return [i for i in tcc.columns.values if not (tcc[i].equals(arules[i]))]


discrepancias = metricas_inspect(comparaMetricas(rules, rules2), rules, rules2)


def scatterplot3D(df, nameX, nameY, nameZ):
    x = df[nameX]
    y = df[nameY]
    z = df[nameZ]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    ax.set_xlabel(nameX)
    ax.set_ylabel(nameY)
    ax.set_zlabel(nameZ)
    return plt.show()

def areaplot3D(df, nameX, nameY, nameZ):
    x = pd.DataFrame(df[metrica])
    max = x.max()
    min = x.min()
    inc = (max-min)*.1
    x['grupo'] = pd.cut(x.value, range(min, max, inc ), right=False)

    y = df[nameY]
    z = df[nameZ]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    ax.set_xlabel(nameX)
    ax.set_ylabel(nameY)
    ax.set_zlabel(nameZ)
    return plt.show()

def scatterplot2D(df, nameX, nameY):
    x = df[nameX]
    y = df[nameY]
    plt.scatter(x, y, alpha=0.2)
    plt.xlabel(nameX)
    plt.ylabel(nameY)
    return plt.show()


def lineplot2DMetricas(df, *args, sort=False):
    if sort:
        df = df.sort_values(by='addedValue')
    plt.plot(df['support'])
    for i in args:
        vals = df[i].values
        vals = (vals - vals.min()) / vals.ptp()
        plt.plot(vals)
    plt.xlabel('Regras')
    plt.ylabel('Metricas')
    return plt.show()


def lineplot2D(df, nameX, nameY):
    df2 = df.sort_values(by=nameX)
    x = df2[nameX]
    y = df2[nameY]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlabel(nameX)
    plt.ylabel(nameY)
    return plt.show()

#def lineCategorical2D(df, nameX):

def Categorical3D(df, nameX, nameY, kind='surface'):
    z = df.assign(c1=pd.cut(df[nameX], bins=10))
    z = z.assign(c2=pd.cut(df[nameY], bins=10))
    zz = pd.crosstab(z.c1, columns=z.c2, dropna=False, normalize=True)
    Z = zz.to_numpy()
    x = np.array([(i + 1) * 0.1 for i in range(10)])
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')

    if kind == 'surface':
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    else:
        ax.contour3D(X, Y, Z, 50, cmap='viridis')

    #indexesX = [i.right for i in zz.index]
    #indexesY = [i.right for i in zz.columns]

    #ax.set_xticks(indexesX)
    #ax.set_yticks(indexesY)
    ax.set(xlim=())
    ax.set_xlabel(nameX)
    ax.set_ylabel(nameY)
    ax.set_zlabel('intersection')
    ax.set_title('3D Categorical')
    return plt.show()



def plotagem(df):
    exit = False
    while not(exit):
        print('Selecione uma metrica (ou "sair"): \n')
        opt = list(df.columns)
        del opt[0:4]
        del opt[opt.index('confidence')]

        print('[ 0 ] Plotar tudo' + (' ' * 23) + '[ ' + str(len(opt) + 1) + ' ] Sair\n')
        print('='*60 + '\n\n')
        for i in range(len(opt)//2):
            n = i * 2
            lin = ''
            for j in range(2):
                k = n+j+1
                if k <= len(opt):
                    lin += '[ ' + str(k) + ' ] ' + opt[k]
                    lin += (' ' * (30 - len(lin)))
            print(lin)

        sel = input()

        try:
            sel = int(sel)
        except:
            print("Opcao Invalida")

        if sel == len(opt)+1:
            exit = True

        elif sel == 0:
            for i in opt:
                scatterplot3D(df, i, 'support', 'confidence')
        elif (sel > len(opt)+1) or (sel < 0):
                print("Opcao invalida.")
        else:
            sel = int(sel) - 1
            scatterplot3D(df, opt[sel], 'support', 'confidence')
            #scatterplot2D(df, opt[sel], 'support')




#lineplot2D(rules2, 'support', 'addedValue')
#Categorical3D(rules2, 'support', 'lift')
#lineplot2DMetricas(rules2, 'lift')
#plotagem(rules2)


'''
x = 'support'       #
y = 'confidence'    #   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Setar eixos aqui
z = 'addedValue'    #


scatterplot3D(rules2, x, y, z)
scatterplot2D(rules2, x, z)
lineplot2D(rules2, z, x)

'''
# chi quadrado / fisher / gini / oddsRatio / kappa >> retorna resultados estranhos (apenas as regras com antecedentes 1-itemset coincidem
# improvement >> arules retorna infinito (range da métrica tem teto de 1)
# cosine >> OK (NaN falso negativo)





#rules['antcBin'] = rules["antc"].apply(lambda x: dados[:, x-1])
#rules['consBin'] = rules["consQ"].apply(lambda x: dados[:, x-1])

#print(dados)
#print("\n\n")
#print(np.invert(dados))

#print(dados[:,[[1, 2], [3, 4]]])
'''regras = pd.read_table(baseRegras, sep="#", names=("AR", "sup", "cnf"))
regras["antc"], regras["cons"] =  regras["AR"].str.split("==>").str
del regras["AR"]
regras["sup"] = regras["sup"].str.replace("SUP:", "").astype('float')
regras["cnf"] = regras["cnf"].str.replace("CONF:", "").astype('float')
regras["cons"] = regras["cons"].str.split()
regras["antc"] = regras["antc"].str.split()
'''

'''
#print(raMetricas.conf(dados, [1], [6, 18]))

print(raMetricas.descCfConf(dados, [1], [6]))


print(raMetricas.relSupp(dados, [6], [1]))


a = raMetricas.getCol(dados, [1])
c = raMetricas.getCol(dados, [6])

print(raMetricas.conf(dados, [1], [6]))

#print(collectiveStrength([6,1]))

print(raMetricas.abSupp(dados, [1], [6]))
'''


#print(raMetricas.relSupp(dados, [1], [6], not1=True, not2=True))
#print(1 + raMetricas.relSupp(dados, [1], [6]) - raMetricas.relSupp(dados, [1]) - raMetricas.relSupp(dados, [6]))

