import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.stats import fisher_exact as fisher
from math import sqrt

baseDados = "BPressureNishiBook.dat"
baseRegras = "BPressureNishiBook.txt"


class raMetricas:
    @staticmethod
    def intersect(db, itemset):
        # Cria um vetor de colunas (matriz) de bd, usando a lista itemset como referencia de indice.
        base = db[:, np.array(itemset) - 1]
        # Metodo all do np achata a matriz usando and logico no eixo passado como parametro.
        return base.all(axis=1)

    @staticmethod
    def abSupp(db, itemset1, itemset2=None, negativo=False):
        itemset = itemset1 + (itemset2 if itemset2 is not None else [])
        # Suporte negativo verifica a existência de transações que nao contem nenhum item do itemset
        # Chama a msma funcao de intersecao, mas com a base de dados invertida.
        if negativo:
            return np.sum(raMetricas.intersect(abs(db - 1), itemset), axis=0)
        # np.sum achata o vetor usando soma no eixo passado como parametro.
        return np.sum(raMetricas.intersect(db, itemset), axis=0)

    @staticmethod
    def relSupp(bd, itemset, itemset2=None, negativo=False):
        return raMetricas.abSupp(bd, itemset, itemset2, negativo) / bd.shape[0]

    @staticmethod
    def subSupp(bd, itemset):
        # Calcula o suporte absoluto de todos os 1-itemsets contidos em itemset.
        subSup = bd[:, np.array(itemset) - 1]
        subSup = np.sum(subSup, axis=0)
        return subSup

    @staticmethod
    def conf(bd, antc, consq):
        raSup = raMetricas.abSupp(bd, antc, consq)
        antSup = raMetricas.abSupp(bd, antc)
        return raSup / antSup

    @staticmethod
    def confDB(a, c):
        base = np.hstack((a, c))
        supRegra = np.sum(base.all(axis=0))
        supAnt = np.sum(a(axis=0))
        return supRegra / supAnt

    @staticmethod
    def addedValue(db, antc, consq):
        return raMetricas.conf(db, antc, consq) - raMetricas.relSupp(db, consq)

    @staticmethod
    def allConf(db, itemset):
        max = raMetricas.subSupp(db, itemset)
        max = np.max(max)
        max /= db.shape[0]
        return raMetricas.relSupp(db, itemset) / max

    @staticmethod
    def casualSupp(db, ant, cons):
        return raMetricas.relSupp(db, ant, cons) + raMetricas.relSupp(db, ant, cons, negativo=True)

    @staticmethod
    def casualConf(db, ant, cons):
        conf1 = raMetricas.conf(db, ant, cons)
        conf2 = raMetricas.abSupp(db, ant, cons, negativo=True) / raMetricas.abSupp(db, ant, negativo=True)
        return (conf1 + conf2) / 2

    @staticmethod
    def certFactor(db, ant, cons):
        cf = raMetricas.conf(db, ant, cons) - raMetricas.relSupp(db, cons)
        return cf / raMetricas.relSupp(db, cons, negativo=True)

    @staticmethod
    def tbContingencia(db, antc, cons):
        bdInv = abs(db-1)
        lin = [raMetricas.intersect(db, antc), raMetricas.intersect(bdInv, antc)]
        col = [raMetricas.intersect(db, cons), raMetricas.intersect(bdInv, cons)]
        return np.matrix([[np.sum(i * j) for i in lin] for j in col])

    @staticmethod
    def chiSqrd(db, antc, cons):
        #    l1 = [5,15,5]
        #    l2 = [50,10,15]
        #    l3 = [55,25,20]
        #    a= np.matrix([l1, l2, l3])

        tb = raMetricas.tbContingencia(db, antc, cons)
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
        violation = db.shape[0]-raMetricas.abSupp(db, itemset) - raMetricas.abSupp(db, itemset, negativo=True)
        violation /= dados.shape[0]
        sup = raMetricas.relSupp(db, itemset)

        return ((1 - violation) / 1 - sup) * ((sup) / violation)

    @staticmethod
    def conviction(db, ant, cons):
        return (1 - raMetricas.relSupp(db, cons)) / 1 - raMetricas.conf(db, ant, cons)

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
        baseCons = db[:, np.array(cons) - 1]
        baseAnt = db[:, np.array(antc) - 1]
        baseConsInv = abs(baseCons - 1)

        conf1 = raMetricas.conf(db, antc, cons)
        conf2 = raMetricas.confDB(baseAnt, baseConsInv)

        return conf1 - conf2


    @staticmethod
    def diffOfConf(db, antc, cons):
        baseCons = db[:, np.array(cons) - 1]
        baseAnt = db[:, np.array(antc) - 1]
        baseAntInv = abs(baseAnt - 1)

        confRA = raMetricas.conf(db, antc, cons)
        conf2 = raMetricas.confDB(baseAntInv, baseCons)

        return confRA - conf2

    @staticmethod
    def exCounterEx(db, antc, cons):
        supRA = raMetricas.relSupp(db, antc, cons)

        baseAnt = db[:, np.array(antc) - 1]
        baseInvCons = abs(db[:, np.array(cons) - 1]-1)

        base = np.hstack((baseAnt, baseInvCons))
        supRegra = np.sum(base.all(axis=0))

        return (supRA-supRegra) / supRA

    @staticmethod
    def fischers(db, antc, cons):
        tb = raMetricas.tbContingencia(db, antc, cons)
        return fisher(tb)[1]





mt = pd.read_table(baseDados, delim_whitespace=True, dtype="str", header=None)
mtBinaria = pd.get_dummies(mt)
dados = mtBinaria.astype('int').to_numpy()

'''regras = pd.read_table(baseRegras, sep="#", names=("AR", "sup", "cnf"))
regras["antc"], regras["cons"] =  regras["AR"].str.split("==>").str
del regras["AR"]
regras["sup"] = regras["sup"].str.replace("SUP:", "").astype('float')
regras["cnf"] = regras["cnf"].str.replace("CONF:", "").astype('float')
regras["cons"] = regras["cons"].str.split()
regras["antc"] = regras["antc"].str.split()
'''

#print(raMetricas.conf(dados, [1], [6, 18]))

raMetricas.descCfConf(dados, [1], [6])

#print(collectiveStrength([6,1]))
