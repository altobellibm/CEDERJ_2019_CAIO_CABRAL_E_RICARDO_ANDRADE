import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.stats import chi2_contingency as chi2
from scipy.stats import fisher_exact as fisher
from math import sqrt
from math import log10
from itertools import combinations

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
        return np.array([[np.count_nonzero(i * j) for i in lin] for j in col])

    @staticmethod
    def __tbContingencia2(db, antc, conq):
        c11 = raMetricas.abSupp(db, antc, conq)
        c10 = raMetricas.abSupp(db, antc, conq, not2=True)
        c01 = raMetricas.abSupp(db, antc, conq, not1=True)
        c00 = raMetricas.abSupp(db, antc, conq, not1=True, not2=True)
        return np.array([[c11, c10],
                         [c01, c00]])

    @staticmethod
    def chiSqrd2(db, antc, conq):
        n = db.shape[0]
        c11 = raMetricas.abSupp(db, antc, conq)
        c10 = raMetricas.abSupp(db, antc, conq, not2=True)
        c01 = raMetricas.abSupp(db, antc, conq, not1=True)
        c00 = raMetricas.abSupp(db, antc, conq, not1=True, not2=True)
        c1x = raMetricas.relSupp(db, antc)
        cx1 = raMetricas.relSupp(db, conq)

        obs = np.array([c11, c10, c01, c00])
        exp = np.array([n*c1x*cx1, n*c1x*(1-cx1), n*(1-c1x)*cx1, n*(1-c1x)*(1-cx1)])


        return chisquare(f_obs=obs, f_exp=exp, ddof=1)[0]
        #return sqrt(sum(((obs[i] - exp[i]) / exp[i])**2 for i in range(len(obs))))

    @staticmethod
    def chiSqrd3(db, antc, conq):
        n = db.shape[0]
        c11 = raMetricas.abSupp(db, antc, conq)
        c10 = raMetricas.abSupp(db, antc, conq, not2=True)
        c01 = raMetricas.abSupp(db, antc, conq, not1=True)
        c00 = raMetricas.abSupp(db, antc, conq, not1=True, not2=True)
        c1x = raMetricas.relSupp(db, antc)
        cx1 = raMetricas.relSupp(db, conq)

        obs = np.array([[c00, c01], [c10, c11]])
        exp = np.outer([n*c1x*cx1, n*c1x*(1-cx1)], [n*(1-c1x)*cx1, n*(1-c1x)*(1-cx1)])

        calc = lambda i, j: (obs[i, j] - exp[i, j])**2 / exp[i,j]
        return sum(calc(i, j) for i in range(obs.shape[0]) for j in range(obs.shape[0]))





    @staticmethod
    def chiSqrd(db, antc, conq):
        #    l1 = [5,15,5]
        #    l2 = [50,10,15]
        #    l3 = [55,25,20]
        #    a= np.matrix([l1, l2, l3])

        tb = raMetricas.__tbContingencia2(db, antc, conq)
        df = tb.shape[1] - 1

        calcEsp = lambda i, j: np.sum(tb[i, :]) * np.sum(tb[:, j]) / np.sum(tb)
        esperado = [calcEsp(i, j) for i in range(tb.shape[0]) for j in range(tb.shape[0])]
        observado = tb.flatten()

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
    def descCfConf(db, antc, cons):
        conf1 = raMetricas.conf(db, antc, cons)
        conf2 = raMetricas.conf(db, antc, cons, notC=True)
        return conf1 - conf2


    @staticmethod
    def differenceOfConfidence2(db, antc, cons):
        conf1 = raMetricas.conf(db, antc, cons)
        conf2 = raMetricas.conf(db, antc, cons, notA=True)

        return conf1 - conf2

    @staticmethod
    def differenceOfConfidence(db, antc, cons):
        c1 = ((raMetricas.relSupp(db, antc) - raMetricas.relSupp(db, antc, cons)) / raMetricas.relSupp(db, antc))
        c2 = (raMetricas.relSupp(db, antc, cons) / raMetricas.relSupp(db, antc))
        return c2 - c1

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
        t1 = raMetricas.relSupp(db, antc)
        t1 *= raMetricas.conf(db, antc, conq)**2 + raMetricas.conf(db, antc, conq, notC=True)**2
        t1 -= raMetricas.relSupp(db, conq)**2
        t2 = raMetricas.relSupp(db, antc, not1=True)
        t2 *= raMetricas.conf(db, antc, conq, notA=True)**2 + raMetricas.conf(db, antc, conq, notA=True, notC=True)**2
        t2 -= raMetricas.relSupp(db, conq, not1=True)**2
        return t1 + t2

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
        t1 = raMetricas.relSupp(db, antc, cons)
        t2 = raMetricas.conf(db, antc, cons)
        t3 = raMetricas.relSupp(db, cons)
        eq1 = t1 * log10(t2 / t3)
        sep = (1-t2)/(1-t3)
        if sep == 0.0:
            return float("NaN")
        sep = log10(sep)
        eq2 = (1-t1) * sep
        return eq1 + eq2

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
    def predictiveAssociation2(db, antc, consq):
        fx1 = raMetricas.abSupp(db, antc)
        fx0 = raMetricas.abSupp(db, consq, not1=True)
        f11 = raMetricas.abSupp(db, antc, consq)
        f10 = raMetricas.abSupp(db, antc, consq, not2=True)
        f01 = raMetricas.abSupp(db, antc, consq, not1=True)
        f00 = raMetricas.abSupp(db, antc, consq, not1=True, not2=True)
        x0x1 = max(fx1, fx0)

        return (max(f11, f10) + max(f01, f00) - x0x1) / (db.shape[0] - x0x1)

    @staticmethod
    def laplaceConf(db, antc, cons, not1=False, not2=False):
        return (raMetricas.abSupp(db, antc, cons, not1=not1, not2=not2) + 1) / (raMetricas.abSupp(db, antc, not1=not1, not2=not2) + 2)

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
        I = lambda i: i * log10(i) + (1-i) * log10(1-i)
        return I(raMetricas.relSupp(db,cons) - I(raMetricas.relSupp(db, antc, cons)))


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

        tb = raMetricas.__tbContingencia2(db, antc, conq)
        total = np.sum(tb)
        x1 = tb[0, 0]
        x2 = tb[0, 1]
        x3 = tb[1, 0]
        x4 = tb[1, 1]
        d = (x1*x4 - x2*x3) / (total**2)

        if (d > 0):
            if x2 < x3:
                return d / (d + x3)
            else:
                return d / (d + x2)
        else:
            if x1 < x4:
                return d / (d - x1)
            else:
                return d / (d - x4)

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
rules['consq'] = rules['consq'].apply(lambda x: list( map(int, x) ))
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
rules2['chiSquared'] = rules.apply(lambda x: raMetricas.chiSqrd3(dados, x['antc'], x['consq']), axis=1)
rules2['cosine'] = rules.apply(lambda x: raMetricas.cosine(dados, x['antc'], x['consq']), axis=1)
rules2['conviction'] = rules.apply(lambda x: raMetricas.conviction(dados, x['antc'], x['consq']), axis=1)
rules2['gini'] = rules.apply(lambda x: raMetricas.giniIndex(dados, x['antc'], x['consq']), axis=1)
rules2['oddsRatio'] = rules.apply(lambda x: raMetricas.oddsRatio(dados, x['antc'], x['consq']), axis=1)
rules2['phi'] = rules.apply(lambda x: raMetricas.phi(dados, x['antc'], x['consq']), axis=1)
rules2['doc'] = rules.apply(lambda x: raMetricas.differenceOfConfidence(dados, x['antc'], x['consq']), axis=1)
rules2['RLD'] = rules.apply(lambda x: raMetricas.RLD(dados, x['antc'], x['consq']), axis=1)
rules2['imbalance'] = rules.apply(lambda x: raMetricas.imbalanceRatio(dados, x['antc'], x['consq']), axis=1)
rules2['kulczynski'] = rules.apply(lambda x: raMetricas.kulczynski(dados, x['antc'], x['consq']), axis=1)
rules2['lambda'] = rules.apply(lambda x: raMetricas.predictiveAssociation2(dados, x['antc'], x['consq']), axis=1)
#rules2['collectiveStrength'] = rules.apply(lambda x: raMetricas.collectiveStrength(dados, x['antc']), axis=1)
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
teste = pd.DataFrame()
metrica = 'oddsRatio' # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Teste
teste['antc'] = rules['antc']
teste['consq'] = rules['consq']
teste['hahsler'] = rules[metrica].round(decimals=5)
teste['TCC'] = rules2[metrica].round(decimals=5)
teste['Equals'] = teste['TCC'].eq(teste['hahsler'])


#teste.to_csv("teste.csv")


print([i for i in rules2.columns.values if not (rules2[i].equals(rules[i]))])

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

