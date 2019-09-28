import numpy as np
import pandas as pd
from scipy.stats import chisquare

baseDados = "BPressureNishiBook.dat"
baseRegras = "BPressureNishiBook.txt"


class raMetricas:
    @staticmethod
    def intersect(bd, itemset):
        # Cria um vetor de colunas (matriz) de bd, usando a lista itemset como referencia de indice.
        base = bd[:, np.array(itemset) - 1]
        # Metodo all do np achata a matriz usando and logico no eixo passado como parametro.
        return base.all(axis=1)

    @staticmethod
    def abSupp(bd, itemset1, itemset2=None, negativo=False):
        itemset = itemset1 + (itemset2 if itemset2 is not None else [])
        # Suporte negativo verifica a existência de transações que nao contem nenhum item do itemset
        # Chama a msma funcao de intersecao, mas com a base de dados invertida.
        if negativo:
            return np.sum(raMetricas.intersect(abs(bd - 1), itemset), axis=0)
        # np.sum achata o vetor usando soma no eixo passado como parametro.
        return np.sum(raMetricas.intersect(bd, itemset), axis=0)

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
    def addedValue(bd, antc, consq):
        return raMetricas.conf(bd, antc, consq) - raMetricas.relSupp(bd, consq)

    @staticmethod
    def allConf(bd, itemset):
        max = raMetricas.subSupp(bd, itemset)
        max = np.max(max)
        max /= bd.shape[0]
        return raMetricas.relSupp(bd, itemset) / max

    @staticmethod
    def casualSupp(bd, ant, cons):
        return raMetricas.relSupp(bd, ant, cons) + raMetricas.relSupp(bd, ant, cons, negativo=True)

    @staticmethod
    def casualConf(bd, ant, cons):
        conf1 = raMetricas.conf(bd, ant, cons)
        conf2 = raMetricas.abSupp(bd, ant, cons, negativo=True) / raMetricas.abSupp(bd, ant, negativo=True)
        return (conf1 + conf2) / 2

    @staticmethod
    def certFactor(bd, ant, cons):
        cf = raMetricas.conf(bd, ant, cons) - raMetricas.relSupp(bd, cons)
        return cf / raMetricas.relSupp(bd, cons, negativo=True)

    @staticmethod
    def chiSqrd(bd, antc, cons):
        #    l1 = [5,15,5]
        #    l2 = [50,10,15]
        #    l3 = [55,25,20]
        #    a= np.matrix([l1, l2, l3])

        bdInv = abs(bd - 1)

        # Montando tabela de contingencia
        lin = [raMetricas.intersect(bd, antc), raMetricas.intersect(bdInv, antc)]
        col = [raMetricas.intersect(bd, cons), raMetricas.intersect(bdInv, cons)]

        tbCtg = np.matrix([[np.sum(i * j) for i in lin] for j in col])

        df = tbCtg.shape[1] - 1

        calcEsp = lambda i, j: np.sum(tbCtg[i, :]) * np.sum(tbCtg[:, j]) / np.sum(tbCtg)

        esperado = [calcEsp(i, j) for i in range(tbCtg.shape[0]) for j in range(tbCtg.shape[0])]
        observado = tbCtg.getA1()

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
        return t1 * np.sqrt(t2)

    @staticmethod
    def coverage(db, antc, cons):
        return raMetricas.relSupp(db, antc)


print(collectiveStrength([6,1]))



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

print(raMetricas.conf(dados, [1], [6, 18]))
