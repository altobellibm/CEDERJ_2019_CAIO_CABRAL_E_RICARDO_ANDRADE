library(arules)
library(ade4)
library(reshape2)


arq <- "~/TCC/repo/Metricas/BPressureNishiBook.dat"




db <- read.table(arq , row.names=NULL, quote="\"", comment.char="")

db <- acm.disjonctif(db)
colnames(db) <- sub('\\w.\\.', "", colnames(db))
db <- as.matrix(db)

dbTransacional <- as(db, "transactions")

itemsetsEclat <- eclat(dbTransacional, parameter = list(supp = 0.1))
rulesEclat <- ruleInduction(itemsetsEclat, dbTransacional, confidence=.5)

rulesApriori <- apriori(dbTransacional, parameter = list(supp = 0.1, conf = 0.5, target="rules"))


#summary(rulesEclat)
#inspect(rulesEclat)

#summary(rulesApriori)
#inspect(rulesApriori)

metricasRegras <- interestMeasure(rulesEclat, transactions = dbTransacional)
metricasItemsets <- interestMeasure(itemsetsEclat, transactions = dbTransacional)

regras <- as(rulesEclat, "data.frame")
metricasRegras$regras <- paste(regras$rules)
write.table(metricasRegras, "metricasRegras.dat", sep=" ", row.names=FALSE)

itemsets <- as(itemsetsEclat, "data.frame")
metricasItemsets$itens <- paste(itemsets$items)
write.table(metricasRegras, "metricasItemsets.dat", sep=" ", row.names=FALSE)


metricasApriori <- interestMeasure(rulesApriori, transactions = dbTransacional)
regras <- as(rulesApriori, "data.frame")
metricasApriori$regras <- paste(regras$rules)
write.table(metricasApriori, "metricasRegrasApriori.dat", sep=" ", row.names=FALSE)

