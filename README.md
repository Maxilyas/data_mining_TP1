# data_mining_TP1

**************
COMMENT LANCER
**************

Choisissez un dataset (chess.dat,mushroom.dat,connect.dat ...etc) et mettez le dans votre dossier avec search.py  
Vous pouvez en trouver à l'addresse suivante : http://fimi.ua.ac.be/data/  
Dans format_data changer le nom de l'ouverture du fichier pour qu'il corresponde au votre.  
Exemple :  with open('connect.dat') as f: ->  with open('chess.dat') as f:  
lancer search.py

**************
*****MAIN*****
**************

Vous avez plusieurs options que vous pouvez choisir dans le main :  

	- checkTimeInFunction : Si vous passez ce booléen à True, vous pourrez alors voir le temps passé dans chaques fonctions à la fin de l'exécution (dev option)  
	- saveGraph : Si vous passez ce booléen à True, alors lors de l'exécution du programme, la figure de la question7 sera automatiquement sauvegardé dans votre dossier.  
	- algo : Vous permet de choisir l'algo que vous voulez utiliser. Si vous mettez 1 alors l'algo de fréquence sera utilisé. Si vous mettez 2 alors l'algo d'aire sera utilisé.  
	- multiProcessing : Vous permet d'activer le multiprocessing.   
Vous pouvez faire varier certains paramètres :  

	- n : c'est le nombre de transaction qui va être choisie.  

**************
***QUESTION***
**************

Question 1 : Implémenter l’algorithme d’échantillonnage des motifs fréquents.  
Fonctions utilisées :  

	- format_data : Ouvre un fichier.dat et met nos données dans data. Création des poids pour les deux algorithmes.  
	- frequencyBasedSampling : Implémentation de l'algorithme 1.  
	- getTransactionData : Renvoi k transactions.  

Question 2 : Implémenter l’algorithme d’échantillonnage basé sur l’aire.  
Fonctions utilisées :  
 
	- format_data : Ouvre un fichier.dat et met nos données dans data. Création des poids pour les deux algorithmes.  
	- areaBasedSampling : Implémentation de l'algorithme 2.
	- getTransactionData : Renvoi k transactions.  

Question 3 : Ecrire une fonction qui étant données k réalisations, retourne les valeurs réelles de la
fréquence et/ou l’aire en une seule passe sur les données.   
Fonctions utilisées :  

	- frequencyMotifsInAllDB : Retourne la valeur de la fréquence (en %) dans la BD(data) des motifs choisis. (/!\ multiprocessing)  
	- init_pool_data : Permet de passer la variable global à tous les agents  
	- checkAllDB : Fonction qui récupère le nombre de fois qu'un motif est présent dans la base de données (pour chaques motifs)  
	- parallelizeCode : Récupère le nombre de cpus que vous avez.

Si vous n'avez pas activé le multiprocessing :

 	- frequencyMotifsWithoutMPInDB : Retourne la valeur de la fréquence (en %) dans la BD(data) des motifs choisis.
	- frequencyMotifsWithoutMPInTransac : Retourne la valeur de la fréquence (en %) dans les transactions des motifs choisis.
    - checkMotifs : Fonction qui prend en argument votre ensemble et récupère le nombre de fois que le motifs apparait

Question 5 : Pour 5 jeux de données différents, afficher la distribution de 1000 réalisations. Attention,
l’approche s’appuie sur un tirage avec remise, il est donc possible d’avoir des doublons qu’il
faudra veiller à supprimer.   
Fonctions utilisées :  

	- isInAllMotifs : retire les doublons.  
	- distribFile : /!\ Cette fonction est commenté dans le code. Nous avons déjà crée les 5 distributions sur des jeux de données différents.  
			Si vous voulez utilisez cette fonction vous même, il faut alors faire attention de changer :  
			**** inputdir = "/home/antoine/Documents/data_mining/data_mining_TP1" -> inputdir = "/your/location/file" ****  
			Cependant faites attention, la fonction fait la distribution de tous les fichiers .dat présent dans votre dossier et les sauvegardes dans ce même dossier.  
			( Ne marche pas avec des sets de données ou il y a une transaction beaucoup plus grande !)  
	- ShowGraphDistrib : Affiche la distribution en fonctions de la fréquence et de la longueur des motifs. (appelle à showGraphScatter)  
	

Question 6 : Mettre en place une expérience pour évaluer la diversité de k tirages.   
Fonctions utilisées :  

	- evalDiversity : Evalue la diversité sur l'algorithme. Regarde si les motifs sont équivalent. Valeur retourner est entre 0 et 1. Plus vous vous rapprochez de 0,
			  meilleur sera votre diversité. (le fait que nous faisons des choix aléatoires assure une très bonne diversité dans l'échantillon)

Question 7 : Mettre en place une expérience pour tester l’échantillonnage par rapport à une approche
complète (tous les motifs de minsupp >= 1).   
Fonctions utilisées : 
 
	- frequencyMotifs : Retourne la valeur de la fréquence dans l'échantillon des motifs choisis. (/!\ multiprocessing)  
	- frequencyMotifsInAllDB : Retourne la valeur de la fréquence (en %) dans la BD(data) des motifs choisis. (/!\ multiprocessing)   
	- init_pool_data,init_pool : Permet de passer la variable global à tous les agents  
	- checkAllDb,contains : récupère respectivement le nombre de fois qu'un motifs apparait dans la BD et dans l'échantillon.  
	- parallelizeCode : Récupère le nombre de cpus que vous avez.  
	
Si vous n'avez pas choisis le multiprocessing :

	- frequencyMotifsWithoutMPInDB : Retourne la valeur de la fréquence (en %) dans la BD(data) des motifs choisis.
	- frequencyMotifsWithoutMPInTransac : Retourne la valeur de la fréquence (en %) dans les transactions des motifs choisis. 
    - checkMotifs : Fonction qui prend en argument votre ensemble et récupère le nombre de fois que le motifs apparait  

Création du Graph :

    - showGraphFrequency : Création du graphe en fonction de la fréquence des motifs dans l'échantillon et de la fréquence dans la base de données.
                           Appel à showGraphScatterAndRL pour l'affichage de tous les points et régression linéaire.
                           
Question 8 : Comment se comporte l’algorithme sur des jeux de données contenant au moins une transactions beaucoup plus grande que les autres ? (e.g., Kosarak). 
Proposer et implémenter une solution.   

L'algorithme sur des jeux de données contenant au moins une transaction beaucoup plus grande que les autres ne se comporte pas très bien.  
En effet, la longueur de la transaction est déjà en elle même un problème pour calculer les poids lors de l'algorithme d'échantillonnage (overflow si la longueur de la transaction est énorme).  
De plus, lorsqu'on a une transaction beaucoup trop grande, celle ci va être tiré constamment à cause de son poids qui sera très élevé, et donc la diversité ne va pas être bonne.  

Pour la solution, nous pouvons changer la formule de calcul des poids pour pas que notre algorithme explose et que les transactions les plus grosses ne soit tiré.  
Ainsi tous ceux dépassant "Average + 2*Ecart-Type" serait supprimé de nos données. Lorsque cela arrive, il se peut qu'il y ait des transactions juste en dessous de ce nombre. Nous avons donc décider
de donner un poids équivalent à toutes les transactions pour pouvoir avoir une chance de toutes les tirer.

Cette solution a été implémenté dans la fonction : format_data

Question 11 : Cette approche est elle adaptée pour échantillonner des motifs fermés ? Justifier et discuter une solution en fonction.

Pas vraiment. En général les ensembles fermés sont mal adaptés à la découverte de connaissance dans des relations bruitées.
En effet la contrainte de connexion est en pratique trop forte. On pourra avoir comme idée d'affaiblir cette contrainte.
















 
