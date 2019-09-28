import numpy as np
import pandas as pd
from scipy.stats import chisquare


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
        # Chamei a msma funcao de intersecao com a base de dados invertida.
        if negativo:
            return np.sum(raMetricas.intersect(itemset, abs(bd-1)), axis=0)
        return np.sum(raMetricas.intersect(itemset, bd), axis=0)

    @staticmethod
    def relSupp(bd, itemset, itemset2=None, negativo=False):
        if itemset2 is not None:
            itemset.append(itemset2)
        return raMetricas.abSupp(bd, itemset, itemset2, negativo) / bd.shape[0]

    @staticmethod
    def conf(bd, antc, consq):
        return raMetricas.abSupp(bd, antc, consq) / raMetricas.abSupp(bd, antc)

    @staticmethod
    def addedValue(bd, antc, consq):
        return raMetricas.conf(bd, antc, consq) - raMetricas.relSupp(bd, consq)

    @staticmethod
    def allConf(bd, itemset):
        vetorSups = np.fromiter((raMetricas.relSupp(bd, i) for i in np.array(itemset)), itemset.dtype)
        maxSup = np.max(vetorSups)
        return raMetricas.relSupp(bd, itemset) / maxSup

    @staticmethod
    def casualSupp(bd, ant, cons):
        return raMetricas.relSupp(bd, ant, cons) + raMetricas.relSupp(bd, ant, cons, negativo=True)
