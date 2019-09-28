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
        # Funcao all do np achata a matriz usando and logico no eixo passado como parametro.
        return base.all(axis=1)

    @staticmethod
    def abSupp(bd, itemset, itemset2=None, negativo=False):
        if itemset2 is not None:
            itemset.append(itemset2)
        # Suporte negativo verifica a existência de transações que nao contem nenhum item do itemset
        # Chama a msma funcao de intersecao, mas com a base de dados invertida.
        if negativo:
            return np.sum(raMetricas.intersect(itemset, abs(bd-1)), axis=0)
        return np.sum(raMetricas.intersect(itemset, bd), axis=0)

    @staticmethod
    def relSupp(bd, itemset, itemset2=None, negativo=False):
        return raMetricas.abSupp(bd, itemset, itemset2, negativo) / bd.shape[0]

    @staticmethod
    def subSupp(bd, itemset):
        # Calcula o suporte de todos os 1-itemsets contidos em itemset.
        subSup = bd[:, np.array(itemset) - 1]
        subSup = np.sum(subSup, axis=0)
        return subSup

    @staticmethod
    def conf(bd, antc, consq):
        return raMetricas.abSupp(bd, antc, consq) / raMetricas.abSupp(bd, antc)

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

print(allConf([1, 6]))
print(allConf2([1, 6]))