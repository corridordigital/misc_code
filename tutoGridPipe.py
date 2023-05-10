# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 21:01:55 2017

"""


#%%

# j'importe les bibliotheques
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import GridSearchCV


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier

from sklearn import datasets
#%%

"""
importation des donnees
"""

iris = datasets.load_iris()
X = iris.data
y = iris.target
#%%

"""
CAS SIMPLE grid search pour trouver via la cross validation, la meilleure architecture (config d'hyperparametres) pour un algorithme precis (pour le GBM par exemple, on veut savoir
la combinaison optimale de learning_rate, n_estimators et max_depth)

"""
# on initialise un estimateur/algorithme de classification (ici GBM)
clf_GBM = GradientBoostingClassifier() # a ce stade nous n'avons pas encore definit les parametres du GBM

# notre grille de configurations pour le GBM
GBM_learning_rate = [0.1, 0.2] # 2 choix possibles
GBM_n_estimators = [20, 50, 100] # 3 choix possibles
GBM_max_depth = [3, 7, 15, 30] # 4 choix possibles
# la grille a donc 2 * 3 *4 combinaison possible de parametres
# ici on definit la grille en respectant la nomenclature des parametres de GBM dans la doc dde scikit learn
grid_GBM = {'learning_rate': GBM_learning_rate, 'n_estimators': GBM_n_estimators, 'max_depth': GBM_max_depth, }




# maintenant qu'on a la grille, nous allons entrainer notre estimateur (ici GBM) sur chacun des 24 configs et trouver celle qui donne la meilleure performance grace a la technique de cross validation a k folds (prenons k = 5)
# il suffit d'utiliser l'objet GridSearchCV() qui te retourne le meilleur estimateur des qu'il sera entraine

# initialisation de l'objet GridSearchCV() qui prend en parametre un classifier et une grille
clf_grid_GBM = GridSearchCV(estimator=clf_GBM, param_grid=grid_GBM, cv=5) # cv = k = 5
# on entraine le GBM pour 24 configs. la cross validation cree a chaque fois, 5 iterations (il y a au total 24 *5 iterations !) commencant chacune par un split de notre set X
# en un set de train et de test suivant les proportion 4/5 et 1/5 (parametre de cross validation cv=5) a partir du jeu de donnees de TRAIN initial (oui ca cree un mini set de train et test a partir du set de train initial X !)
# La performance de CHACUNE des 24 configs est la moyenne des performance des 5 modeles ayant cette meme config mais entraines sur des jeux de donnees differents (mais qui respectent tous la proportion 4/5 en train - 1/5 en test du jeu de donnees X)
clf_grid_GBM.fit(X,y) # on entraine le modele avec X la matrice numpy narray des donnees sans la colonne cible et y la colonne cible (un vecteur numpy)

# a la fin clf_grid_GBM choisit LA meilleure config parmi les 24 configs et LE meilleur modele obtenu parmi les 5 modeles partageant cette config (rappelons nous que la cross validation a fait 5 modeles par config)
# ce meilleur modele est clf_grid_GBM.best_estimator_,
# on reutilise le modele pour predire
clf_grid_GBM.predict(X) # qui revient a faire clf_grid_GBM.best_estimator_.predict(X)
# il faut comprendre que clf_grid_GBM n'est pas intrinsequement un classifier/estimateur, mais les developpers ont simplifie la syntaxe
# ce qui nous permet d'eviter d'ecrire clf_grid_GBM.best_estimator_ a chaque fois

# Si tu veux recuperer les hyperparametres learning_rate, n_estimators
clf_grid_GBM.best_estimator_.learning_rate

#%%

"""
CAS en pipeline (suite de transformations suivie ou non d'un classifier/estimateur) pour effectuer une grid search sur differentes config de PCA suivi de GBM  
rappel: dans un pipeline, les etapes intermediaires doivent avoir la methode transform() sinon ca ne marche pas, en effet "on transforme les donnees pour preparer l'entree de l'algo qui intervient a la derniere etape"
Dans notre cas, PCA() possede une methode transform(), elle peut donc intervenir dans un pipeline (en effet, c'est une transformation qui va projeter les donnees X dans un autre espace, c'est un changement de base)
"""

# on definit notre pipeline en donnant un nom a chaque etape du pipeline (un nom pour l'etape PCA et un nom pour l'etape GBM)

# notre objet pipeline sera lui aussi un estimateur/classifier car l'etape finale ici GBM est un classifier
reduce_dim = PCA() # initialisation d'un algo PCA
clf_GBM = GradientBoostingClassifier() # initialisation d'un algo GBM
pipe_GBM_wPCA = Pipeline([('reduce_dim', reduce_dim), ('clf_GBM', clf_GBM)]) # initialisation du pipeline, c'est ici qu'on definit un nom pour chaque etape


# on definit la gridSearch
# notre grille de configurations pour le PCA
PCA_n_components = [2,3,4]

# notre grille de configurations pour le GBM
GBM_learning_rate = [0.1, 0.2]
GBM_n_estimators = [20, 50, 100]
GBM_max_depth = [3, 7, 15, 30]

# la grille du pipeline: ATTENTION la syntaxe pour chaque hyperparametre est 'nomEtape__nomHyperParametre': liste_valeurs (attention au double underscore)
# par exemple PCA dont le nom de l'etape est reduce_dim, a un parametre n_components (voir la doc de scikit learn pour la liste des parametres) qu'on veut faire varier sur la liste PCA_n_components definie precedemment
# on ecrit donc 'reduce_dim__n_components': PCA_n_components
grid_GBM_wPCA = {'reduce_dim__n_components': PCA_n_components, \
          'clf_GBM__learning_rate': GBM_learning_rate, 'clf_GBM__n_estimators': GBM_n_estimators, \
          'clf_GBM__max_depth': GBM_max_depth}


# l'objet GridSearchCV()
clf_grid_GBM_wPCA = GridSearchCV(estimator=pipe_GBM_wPCA, param_grid=grid_GBM_wPCA, cv=5)

# on fait de meme que le cas simple
clf_grid_GBM_wPCA.fit(X, y)
clf_grid_GBM_wPCA.predict(X)


# Si tu veux recuperer les hyperparametres d'une etape particuliere
clf_grid_GBM.best_estimator_.named_steps['clf_GBM'].learning_rate
clf_grid_GBM.best_estimator_.named_steps['reduce_dim'].n_components

#%%
"""
CAS le plus complet: on veut tester un pipeline en 2 etapes: 
    - etape 1: avec OU sans PCA (avec une grille pour PCA)
    - etape 2 : GBM OU RegressionLog (avec une grille pour chacun des 2)
    
"""

no_reduce_dim = FunctionTransformer() # un transformer qui ne fait rien, fonction identite
reduce_dim = PCA() # initialisation d'un algo PCA
clf_GBM = GradientBoostingClassifier() # initialisation d'un algo GBM, on aurait pu choisir LogisticRegression()
clf_logReg = LogisticRegression(max_iter=150)
pipe_bourrinage = Pipeline([('reduce_dim_or_not', reduce_dim), ('clf_GBM_or_logReg', clf_GBM)]) # initialisation du pipeline, on met n'importe quel objet pour chaque etape




# c'est dans la grille qu'on va pouvoir changer completement la nature des etapes

# notre grille de configurations pour le PCA
PCA_n_components = [2,3,4]

# grille pour la regression logistique
# RQ: C est un parametre lie a la regularisation pour empecher le surapprentissage, plus C est petit (inf a 1), plus la regularisation est stricte et moins il y a de variables importantes
logReg_C = np.logspace(-4,1,6) # l'hyperParametre de la regression log variant parmi 0.0001, 0.001, 0.01 , 0.1 , 1, 10

# notre grille de configurations pour le GBM
GBM_learning_rate = [0.1, 0.2]
GBM_n_estimators = [20, 50, 100]
GBM_max_depth = [3, 7, 15, 30]

# la grille du pipeline, la syntaxe la plus stable est malheureusement tres lourde (les developpers travaillent pour rendre moins lourd le tuning d'etapes)
# on remarque que la grille n'est plus un dictionnaire mais une liste de dictionnaires
# d'autres packages sont proposes sur le net pour avoir une syntaxe moins verbose mais ce n'est pas stable suivant les version de python

grid_bourrinage = [
    {
        'reduce_dim_or_not': [no_reduce_dim],
        'clf_GBM_or_logReg': [clf_GBM],
        'clf_GBM_or_logReg__learning_rate': GBM_learning_rate,
        'clf_GBM_or_logReg__n_estimators': GBM_n_estimators,
        'clf_GBM_or_logReg__max_depth': GBM_max_depth,
    },
     {
        'reduce_dim_or_not': [no_reduce_dim],
        'clf_GBM_or_logReg': [clf_logReg],
        'clf_GBM_or_logReg__C': logReg_C
    },
     {
        'reduce_dim_or_not': [reduce_dim],
        'reduce_dim_or_not__n_components': PCA_n_components,
        'clf_GBM_or_logReg': [clf_GBM],
        'clf_GBM_or_logReg__learning_rate': GBM_learning_rate,
        'clf_GBM_or_logReg__n_estimators': GBM_n_estimators,
        'clf_GBM_or_logReg__max_depth': GBM_max_depth,
    },
      {
    'reduce_dim_or_not': [reduce_dim],
    'reduce_dim_or_not__n_components': PCA_n_components,
        'clf_GBM_or_logReg': [clf_logReg],
        'clf_GBM_or_logReg__C': logReg_C
    }
]


"""
on peut se demander quel interet d'ecrire des listes comme 'clf_GBM_or_logReg': [clf_GBM] au lieu de 'clf_GBM_or_logReg': clf_GBM

cela permettrait dans le cas exceptionnel ou nos choix pour la meme etape ont les memes noms de parametres (par exemple regression logistique et SVC ont 'hyperparametre C en commun), on peut ecrire
grid_bourrinage = [{
        'reduce_dim_or_not': [no_reduce_dim],
        'clf_logReg_or_SVC': [LogisticRegression(max_iter=150), SVC()],
        'clf_logReg_or_SVC__C': [0.0001, 0.001, 0.01 , 0.1 , 1, 10]
    }]

au lieu de
grid_bourrinage = [{
        'reduce_dim_or_not': [no_reduce_dim],
        'clf_logReg_or_SVC': [LogisticRegression(max_iter=150)],
        'clf_logReg_or_SVC__C': [0.0001, 0.001, 0.01 , 0.1 , 1, 10]
    },
{
        'reduce_dim_or_not': [no_reduce_dim],
        'clf_logReg_or_SVC': [SVC()],
        'clf_logReg_or_SVC__C': [0.0001, 0.001, 0.01 , 0.1 , 1, 10]
    }

]

Ce qui est en general plus utilise pour tester rapidement differentes techniques
de reduction de dimensions comme PCA qui ont 
tous le parametre n_components
"""

# l'objet GridSearchCV()
clf_grid_bourrinage = GridSearchCV(estimator=pipe_bourrinage, param_grid=grid_bourrinage, cv=5)

# on fait de meme que le cas simple
clf_grid_bourrinage.fit(X, y)
clf_grid_bourrinage.predict(X)






#%%

'''
UTILISATION DE LA METHODE D'ENSEMBLE BAGGING POUR REDUIRE LA VARIANCE DES PERFORMANCES D'UN ALGORITHME DUE A L'INITIALISATION

RAPPEL: 
Un algorithme initialise des coefficients d'un modele au hasard et les ajuste pendant son parcours du set de train. Meme avec la MEME configuration
d'hyperparametres (on appelle ca une architecture) et avec le MEME jeu de donnees de train, on peut trouver des resultats differents a cause
de cette initialisation aleatoire. Encore pire, si un jour on entraine sur un jeu de donnees puis un autre jour on entraine avec un autre jeu de donnees de train. 
Pour augmenter la robustesse, la stabilite d'une architecture, on va creer plusieurs modeles avec la possibilite de changer le jeu de donnees et le set de variables (on s'inspire de randomForest dont chaque arbre
a ete entraine par exemple sur un echantillon de 70% pris au hasard pour chaque arbre et non entraine sur la totalite du set de train pour simuler des jeux de donnees differents. De plus chaque arbre se restreint a un jeu de variables differents)
Le modele final sera alors la moyenne des probabilites des differents modeles ayant la meme architecture.


A NE PAS CONFONDRE

- la CrossValidation qui est une methode pour evaluer une architecture !
Elle decoupe le jeu de train en k parties appelees folds
Elle cree k modeles ayant la meme architecture en les entrainant sur des echantillons differents (les k-1 folds
pris differemment pour chaque modele) puis les teste sur le dernier fold (par exemple le premier modele est trained sur fold 1, 2, 3, 4 et tested sur le 5e
le deuxieme modele est trained sur folds 2, 3, 4, 5 et tested sur le 1er fold).
Le critere d'evaluation de l'architecture est la moyenne de performance des k modeles.

- le Bagging-ensemble est une methode consistant a creer plusieurs modeles de meme architecture mais entraines sur differents echantillons et de considerer l'ensemble comme un seul modele ! (la sortie est une proba moyenne)


Pour resumer la crossValidation permet de trouver la meilleure architecture, alors que le bagging est lui meme un modele (ensemble de modeles pour etre precis).
On peut tres bien faire un bagging de modeles ayant une architecture commune pourrie.

LA BONNE PRATIQUE est donc d'utiliser la crossValidation dans la gridSearch pour detecter la meilleure architecture(ce qui a ete fait precedemment),
puis de stabiliser cette architecture par le Bagging !


'''
# exemple bagging de 10 regression logistiques
# attention au parametrage: ecrire max_samples|max_features=1. signifie qu'on utilise 100% des samples|features
# alors que max_samples|max_features=1. signifie qu'on utilise 1 des sample|feature !!!
ensemble_logReg = BaggingClassifier(LogisticRegression(C=0.01), n_estimators=10, max_samples=0.7, max_features=1.) # ici chaque modele est trained sur 70% du set de train X et utilise toutes les variables
ensemble_logReg.fit(X, y)
ensemble_logReg.predict(X, y)

# exemple bagging de 10 modeles ayant la meme architecture que le meilleur modele de la gridSearch effectuee precedemment
ensemble_bourrinage = BaggingClassifier(clf_grid_bourrinage.best_estimator_, n_estimators=10, max_samples=0.7, max_features=1.)
ensemble_bourrinage.fit(X, y)
ensemble_bourrinage.predict(X, y)



#%%

"""
POUR FINIR:
le plus dur finalement en machine learning est la partie data management souvent avec la bibliotheque Pandas pour preparer les donnees X et y
alors que scikit learn n'utilise pas le format pandas (il ne connait pas les pandas dataframe ni pandas series et NE GERE PAS les String) ce qui necessite de se plonger dans les docs officielles pour bien verifier les formats d'entrees et sorties
pour assurer les conversions necessaires a la compatibilite

"""
