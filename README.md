# BELearningParameters
Questo progetto implementa l'apprendimento di parametri attraverso l'approccio bayesiano di una rete Bayesiana con struttura e distribuzione di probabilità nota, quindi misura la distanza tra la distribuzione trovata e la distribuzione nota.

#### Creazione della rete

La rete è inserita attraverso il metodo __readNetwork__ della classe _BayesianNetwork_, che prende in input il percorso del file da utilizzare e crea una rete Bayesiana da esso.
<br>In _resources/datasets_ sono contenuti due esempi che rappresentano la rete Bayesiana _cancer_neapolitan_ contenuta negli esempi di Hugin Lite.
* __V__: contiene la lista dei nodi della rete.
* __E__: dizionario avente come chiave ognuno dei nodi e come valore un'array con la lista dei figli.
* __F__: dizionario avente come chiave ognuno dei nodi e come valore un dizionario, contenente
  * __values__: un array indicante il dominio
  * __parents__: un array indicante i padri 
  * __cpt__: un array monodimensionale indicante la CPT del nodo
  * __alphas__: un array contenente gli pseudocounts di ogni valore del nodo.

#### Apprendimento

L'apprendimento è effettuato attraverso la funzione __bayes_estimator__ che prende in ingresso un oggetto _BayesianNetwork_
e il percorso di un file csv ed effettua l'apprendimento dei parametri.  
Un esempio di file csv è contenuto in _resources/datasets/Example.csv_

L'apprendimento è effettuato nel seguente modo:
1. Per ogni nodo vengono riportati gli pseudocounts se diversi da zero, altrimenti viene supposta una distribuzione uniforme sull'intera cpt.
2. Il dataset in input viene analizzato aggiornando gli pseudocounts e la cpt

#### Misura della qualità della soluzione

La distanza tra la distribuzione appresa e la distribuzione p nota viene calcolata attraverso la funzione __calculateDivergency__, che prende in input un oggetto _BayesianNetwork_ e restituisce la divergenza tra le due.

#### Esempio

Di seguito un esempio di lettura della rete,apprendimento e calcolo della divergenza. 
La classe Tester.py presenta in maniera più estesa il codice utilizzato per la generazione dei risultati descritti nella relazione.
    
    from BayesianNetwork import BayesianNet
    from KLDivergenceCalculation import calculateKLDivergency
    from BayesianEstimate import bayesEstimate
    
    #Bayesian Network reading
    bn = BayesianNet()
    bn.readNetwork("resources/cancer3uniform.bn")
    
    #Learning
    csvfile10 = 'resources/datasets/10Cases.csv'
    bayesEstimate(csvfile10, bn)
    
    #Kullack-Leibler divergency
    print('KL divergence for n = 10', calculateKLDivergency(bn))
    
