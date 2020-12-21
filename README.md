			Interface d'analyse de données par apprentissage supervisé

Avant de lancer l'application, veuillez installer les dépendances suivantes avec conda :
	conda install selenium geckodriver firefox -c conda-forge
N.B. : vous ne pourrez pas sauvegarder les résultats de vos analyses sans cette installation.

Une fois installées, changez le répertoire courant pour celui de l'application et rentrez la commande :
	bokeh serve --show Application_Python.py

L'application permet de charger un jeu de données au format .csv avec comme séparateur une ",".
A titre d'essai, l'application utilise le fichier FRvideos.csv.
Un fichier gm_2008_region.csv est également fourni pour vous permettre de changer de jeu de données.