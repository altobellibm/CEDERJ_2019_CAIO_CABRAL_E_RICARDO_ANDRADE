import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.stats import fisher_exact as fisher
from math import sqrt
from math import log10
from itertools import combinations


baseDados = "BPressureNishiBook.dat"
baseRegras = "BPressureNishiBook.txt"


class raMetricas:
    @staticmethod
    def __getCol(db, itemset, not1=False):
        mtz = db[:, np.array(itemset) - 1]
        if not1:
            mtz = np.invert(mtz)
        return mtz

    @staticmethod
    def __intersect(db, itemset1, itemset2=None, not1=False, not2=False):
        itemsets = raMetricas.__getCol(db, itemset1, not1=not1)
        if itemset2 is not None:
            itemsets = np.hstack((itemsets, raMetricas.__getCol(db, itemset2, not1=not2)))
        return itemsets.all(axis=1)


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
        conf1 = raMetricas.conf(db, antc, conq)
        conf2 = raMetricas.conf(db, antc, conq, notA=True, notC=True)
        return (conf1 + conf2) / 2

    @staticmethod
    def casualSupp(db, antc, conq):
        t1 = raMetricas.relSupp(db, antc, conq)
        t2 = raMetricas.relSupp(db, antc, conq, not1=True, not2=True)
        return t1 + t2

    @staticmethod
    def certFactor(db, antc, conq):
        cf = raMetricas.conf(db, antc, conq) - raMetricas.relSupp(db, conq)
        return cf / raMetricas.relSupp(db, conq, not1=True)

    @staticmethod
    def __tbContingencia(db, antc, conq):
        bdInv = abs(db-1)
        lin = [raMetricas.__intersect(db, antc), raMetricas.__intersect(bdInv, antc)]
        col = [raMetricas.__intersect(db, conq), raMetricas.__intersect(bdInv, conq)]
        return np.matrix([[np.count_nonzero(i * j) for i in lin] for j in col])

    @staticmethod
    def chiSqrd(db, antc, conq):
        #    l1 = [5,15,5]
        #    l2 = [50,10,15]
        #    l3 = [55,25,20]
        #    a= np.matrix([l1, l2, l3])

        tb = raMetricas.__tbContingencia(db, antc, conq)
        df = tb.shape[1] - 1

        calcEsp = lambda i, j: np.sum(tb[i, :]) * np.sum(tb[:, j]) / np.sum(tb)
        esperado = [calcEsp(i, j) for i in range(tb.shape[0]) for j in range(tb.shape[0])]
        observado = tb.getA1()

        chi, p = chisquare(f_obs=observado, f_exp=esperado, ddof=df)
        return chi

    @staticmethod
    def crossSuppRatio(db, itemset):
        subSup = raMetricas.subSupp(db, itemset)
        return np.min(subSup) / np.max(subSup)

    @staticmethod
    def collectiveStrength(db, itemset):
        violation = db.shape[0] - raMetricas.abSupp(db, itemset) - raMetricas.abSupp(db, itemset, not1=True)
        violation /= dados.shape[0]
        sup = raMetricas.relSupp(db, itemset)

        return ((1 - violation) / 1 - sup) * ((sup) / violation)

    @staticmethod
    def conviction(db, antc, cons):
        return (1 - raMetricas.relSupp(db, cons)) / 1 - raMetricas.conf(db, antc, cons)

    @staticmethod
    def cosine(db, antc, cons):
        t1 = raMetricas.relSupp(db, antc, cons)
        t2 = raMetricas.relSupp(db, antc) * raMetricas.relSupp(db, cons)
        return t1 * sqrt(t2)

    @staticmethod
    def coverage(db, antc, cons):
        return raMetricas.relSupp(db, antc)

    @staticmethod
    def descCfConf(db, antc, cons):
        conf1 = raMetricas.conf(db, antc, cons)
        conf2 = raMetricas.conf(db, antc, cons, notC=True)
        return conf1 - conf2


    @staticmethod
    def differenceOfConfidence(db, antc, cons):
        conf1 = raMetricas.conf(db, antc, cons)
        conf2 = raMetricas.conf(db, antc, cons, notA=True)
        return conf1 - conf2

    @staticmethod
    def exCounterEx(db, antc, conq):
        suporte1 = raMetricas.abSupp(db, antc, conq)
        suporte2 = raMetricas.abSupp(db, antc, conq, not2=True)
        return (suporte1 - suporte2) / suporte1

    @staticmethod
    def fischers(db, antc, cons):
        tb = raMetricas.__tbContingencia(db, antc, cons)
        return fisher(tb)[1]

    @staticmethod
    def giniIndex(db, antc, conq):
        t1 = (raMetricas.conf(db, antc, conq)**2 + raMetricas.conf(db, antc, conq, notA=True)**2)
        t1 *= raMetricas.relSupp(db, antc)
        t2 = (raMetricas.conf(db, antc, conq, notA=True)**2 + raMetricas.conf(db, antc, conq, notA=True, notC=True)**2)
        t2 *= raMetricas.relSupp(db, antc, not1=True)
        return t1 + t2 - raMetricas.relSupp(db, conq)**2 - raMetricas.relSupp(db, conq, not1=True)

    @staticmethod
    def imbalanceRatio(db, antc, conq):
        suporteA = raMetricas.relSupp(db, antc)
        suporteC = raMetricas.relSupp(db, conq)
        termo1 = abs(suporteA - suporteC)
        termo2 = suporteA + suporteC - raMetricas.relSupp(db, antc, conq)
        return termo1 / termo2

    @staticmethod
    def importance(db, antc, cons):
        return log10(raMetricas.laplaceConf(db, antc, cons) / raMetricas.laplaceConf(db, antc, cons, not2=True))

    @staticmethod
    def improvement(db, antc, cons):
        confRule = raMetricas.conf(db, antc, cons)
        improvement = lambda base, cr, a, c: (cr - raMetricas.conf(base, a, c))
        subRules = [combinations(antc, i) for i in range(len(antc+1))]
        return min([improvement(db, confRule, subRule, cons) for subRule in subRules])

    @staticmethod
    def jaccardCoefficient(db, antc, cons):
        termo1 = raMetricas.relSupp(db, antc, cons)
        termo2 = raMetricas.relSupp(db, antc) + raMetricas.relSupp(db, cons) - raMetricas.relSupp(db, antc, cons)

        return termo1 / termo2

    @staticmethod
    def Jmeasure(db, antc, cons):
        t1 = raMetricas.relSupp(db, antc, cons)
        t2 = raMetricas.conf(db, antc, cons)
        t3 = raMetricas.relSupp(db, cons)
        eq1 = t1 * log10(t2 / t3)
        eq2 = (1-t1) * log10((1-t2) / (1-t3))
        return eq1 + eq2

    @staticmethod
    def kappa(db, antc, cons):
        ac = raMetricas.relSupp(db, antc, cons)
        acN = raMetricas.relSupp(db, antc, cons, not1=True, not2=True)
        a = raMetricas.relSupp(db, antc)
        aN = raMetricas.relSupp(db, antc, not1=True, not2=True)
        c = raMetricas.relSupp(db, cons)
        cN = raMetricas.relSupp(db, cons, not1=True, not2=True)

        t1 = ac + acN - a * c - aN * cN
        t2 = 1 - a * c - aN * cN
        return t1 / t2

    @staticmethod
    def klosgen(db, antc, conq):
        t1 = sqrt(raMetricas.relSupp(db, antc, conq))
        t2 = (raMetricas.conf(db, antc, conq) - raMetricas.relSupp(db, conq))
        return t1 * t2

    @staticmethod
    def kulczynski(db, antc, conq):
        return (raMetricas.conf(db, antc, conq) + raMetricas.conf(db, conq, antc)) / 2

    @staticmethod
    def predictiveAssociation(db, antc, cons):
        pass

    @staticmethod
    def laplaceConf(db, antc, cons, not1=False, not2=False):
        return (raMetricas.abSupp(db, antc, not1) + 1) / (raMetricas.abSupp(db, cons, not2) + 2)

    @staticmethod
    def lermanSimilarity(db, antc, cons):
        t1 = sqrt(db.shape[0])
        t2 = raMetricas.relSupp(db, antc) * raMetricas.relSupp(db, cons)
        return (raMetricas.relSupp(db, antc, cons) - t2 / sqrt(t2)) * t1

    @staticmethod
    def leastContradiction(db, antc, conq):
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
    def mutualInformation(db, antc, cons):
        pass

    @staticmethod
    def CorrelationCoeficient(db, antc, conq):
        a = raMetricas.relSupp(db, antc)
        c = raMetricas.relSupp(db, conq)
        return (raMetricas.relSupp(db, antc, conq) - a * c) / sqrt(a * c * (1-a) * (1-c))

    @staticmethod
    def oddsRatio(db, antc, cons):
        a = raMetricas.__getCol(db, antc)
        c = raMetricas.__getCol(db, cons)

        t1 = raMetricas.relSupp(db, antc, cons) * raMetricas.relSupp(db, antc, cons, not1=True)
        t2 = raMetricas.relSupp(db, antc, cons, not2=True) * raMetricas.relSupp(db, antc, cons, not1=True)

        return t1 / t2

    @staticmethod
    def ralambondrainyMeasure(db, antc, conq):
        return raMetricas.relSupp(db, antc, conq, not2=True)

    @staticmethod
    def RLD(db, antc, conq):
        tb = raMetricas.__tbContingencia(db, antc, conq)
        total = np.sum(tb)
        return tb[1, 1] * tb[2, 2] - tb[2, 1] * tb[1, 2] / (total ** 2)

    @staticmethod
    def rulePF(db, antc, conq):
        return raMetricas.relSupp(db, antc, conq) * raMetricas.conf(db, antc, conq)

    @staticmethod
    def sebagSchoenauerMeasure(db, antc, conq):
        t1 = raMetricas.relSupp(db, antc, conq)
        t2 = raMetricas.relSupp(db, antc, conq, not2=True)
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


mt = pd.read_table(baseDados, delim_whitespace=True, dtype="str", header=None)
mtBinaria = pd.get_dummies(mt)
dados = mtBinaria.astype('bool').to_numpy()
print(dados)
print("\n\n")
print(np.invert(dados))


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


print(raMetricas.relSupp(dados, [1], [6], not1=True, not2=True))
print(1 + raMetricas.relSupp(dados, [1], [6]) - raMetricas.relSupp(dados, [1]) - raMetricas.relSupp(dados, [6]))

