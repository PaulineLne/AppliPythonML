################## Auteur : Decoster Rémi, Iborra Alexandre, Lainé Pauline
################## Projet Python - M2 SISE 2020/2021
################## Sous la supervision de M. Sawadogo Nicolas

import numpy as np
import pandas as pd
import time
from bokeh.io import curdoc, export_png
from bokeh.layouts import row, column
from bokeh.models.widgets import TextInput, Select,DataTable, TableColumn
from bokeh.plotting import figure
from bokeh.models import Div, MultiChoice, PreText, Button, Paragraph, ColumnDataSource
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from os.path import dirname, join


######################## Chargement des données de base ########################
data = pd.read_csv("FRvideos.csv", sep=",",encoding='utf-8')
data["heure_publication"] = data["heure_publication"].str.replace(":","").astype(int)
data["date_tendance"] = data["date_tendance"].str.replace("/","").astype(int) #pd.to_datetime(X.date_tendance) #X["date_tendance"].str.replace("/","")
data["date_publication"] = data["date_publication"].str.replace("/","").astype(int) #pd.to_datetime(X.date_publication) #X["date_publication"].str.replace("/","")
var = list(data)

######################## Fonctions de l'interface ########################
#Chargement des doonées
def importation(attr, old, new):
    fichier = text_input.value
    data = pd.read_csv(fichier, sep=",")
    var = list(data)
    select_cible.options = var
    predictive.options = var


#Selectionne les algos possible en fonction du type de la variable cible
def callback(attr, old, new):
    cible = select_cible.value #Target value
    fichier = text_input.value #Name data
    data = pd.read_csv(fichier, sep=",") #Import data
    type_cible = np.dtype(data[cible]).name 
    # Si le type de la variable cible est numerique ou non numérique
    if ('int' in type_cible) or ('float' in type_cible):
        select_algo.options = ['Ridge','Lasso','SGD']
    else:
        select_algo.options = ['DecisionTree','KNN','SGD']

#Fonction ridge
def fridge(X,Y):
    start_time = time.time()
    name1 = PreText(text=" Ridge selected")
    target= select_cible.value 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8) 
    #define the model and parameters
    ridge = Ridge(normalize=True)   
    params = {'alpha':[.01,.1,1.,10,100]}
    #Fit the model
    regR = GridSearchCV(ridge, param_grid=params, cv=5, n_jobs=6)
    regR.fit(X_train, y_train) #On applique sur les données d'apprentissage
    #predictions on train data
    y_train_predict=regR.predict(X_train)
    #predictions on test data
    y_pred = regR.predict(X_test)
    # Évaluation de l'algorithme
    print(regR.best_params_)#Meilleur paramètrage
    #Graphique
    p1 = figure( title="Prediction de la variable : %s " % target)
    p1.circle(range(len(y_pred)), y_pred[np.argsort(y_test)] , fill_alpha=0.8 , color = 'red', legend_label = "Prédiction")#,source=source)  
    p1.line(range(len(y_test)), np.sort(y_test) , color = 'blue', legend_label = "Echantillon test") #données réelles
    p1.plot_width = 900
 
    exec_time = round((time.time() - start_time),2)
    
    resu1 = row(p1, Regression_metrics(regR, X_train,y_train,X_test,y_test, exec_time, name1)) #column(blabla5,model_info,pre5,learn))
    return resu1

#Fonction Lasso    
def flasso(X,Y):
    start_time = time.time()
    name2 = PreText(text=" Lasso selected")
    target= select_cible.value
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8) 
    #define the model and parameters
    lasso = Lasso(normalize=True)
    params = {'alpha':[1,.5,.1,.01]}
    #Fit the model
    regL = GridSearchCV(lasso, param_grid=params, cv=5, n_jobs=6)
    regL.fit(X_train, y_train) #On applique sur les données d'apprentissage
    #predictions on train data
    y_train_predict=regL.predict(X_train)
    #predictions on test data
    y_pred = regL.predict(X_test)
    # Évaluation de l'algorithme
    print(regL.best_params_)#Meilleur paramètrage
    #Graphique
    p2 = figure( title="Prediction de la variable : %s " % target)   
    p2.circle(range(len(y_pred)), y_pred[np.argsort(y_test)] , fill_alpha=0.8 , color = 'red', legend_label = "Prédiction")#,source=source)  
    p2.line(range(len(y_test)), np.sort(y_test) , color = 'blue', legend_label = "Echantillon test") #données réelles 
    p2.plot_width = 900
 
    exec_time = round((time.time() - start_time),2)
    
    resu2 = row(p2, Regression_metrics(regL, X_train,y_train,X_test,y_test, exec_time, name2))
    return resu2

#Fonction KNN
def fknn(X,Y):
    
    nomal = PreText(text=" KNN selected")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

    #List Hyperparameters that we want to tune.
    leaf_size = list(range(1,10))
    n_neighbors = list(range(1,10,2))
    p=[1,2]
    #Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    
    #Create new KNN object
    knn_2 = KNeighborsClassifier()
    
    #Fit the model
    clf = GridSearchCV(knn_2, hyperparameters,cv=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end_time = time.time()
    exec_time = round(end_time - start_time, 2)

    #Graphique
    categ = y_test.astype("category")
    dico_tmp = dict(enumerate(categ.cat.categories))
    dico = dict((v, k) for k, v in dico_tmp.items())
    #on converti les categories en entier
    y_test_int = y_test.map(dico)
    y_pred_int = pd.Series(y_pred).map(dico)
    dico_bokeh = Paragraph(text=" Dictionnaire des correspondances (clef, valeur) : \n   "+str(dico),  width= 1000)
 
    p7 = figure(title="Prediction de la variable : %s " % select_cible.value, x_axis_label = "x" , y_axis_label = "y")
    p7.circle(range(len(y_pred)), y_pred_int , fill_alpha=0.8 , color = 'red', legend_label = "Prédiction")#,source=source)  
    p7.circle(range(len(y_test)), y_test_int , fill_alpha=0.4 , color = 'blue', legend_label = "Echantillon test") #données réelles
    p7.plot_width = 900 
    
    resu3 = row(column(p7,dico_bokeh),Classification_metrics(clf,X_train,y_train,X_test,y_test, exec_time, nomal))
    return resu3

#Fonction Tree    
def fdectree(X,Y):
    
    name = Paragraph(text=" DecisionTree method applied")
    start_time = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
    clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    end_time = time.time()
    exec_time = round(end_time - start_time, 2)
    
    #Graphique
    categ = y_test.astype("category")
    dico_tmp = dict(enumerate(categ.cat.categories))
    dico = dict((v, k) for k, v in dico_tmp.items())
    #on converti les categories en entier
    y_test_int = y_test.map(dico)
    y_pred_int = pd.Series(y_pred).map(dico)
    dico_bokeh = Paragraph(text=" Dictionnaire des correspondances (clef, valeur) : \n   "+str(dico),  width= 1000)

    p7 = figure(title="Prediction de la variable : %s " % select_cible.value, x_axis_label = "x" , y_axis_label = "y")
    p7.circle(range(len(y_pred)), y_pred_int , fill_alpha=0.8 , color = 'red', legend_label = "Prédiction")#,source=source)  
    p7.circle(range(len(y_test)), y_test_int , fill_alpha=0.4 , color = 'blue', legend_label = "Echantillon test") #données réelles
    p7.plot_width = 900    
    
    resu4 = row(column(p7, dico_bokeh),Classification_metrics(clf,X_train,y_train,X_test,y_test, exec_time, name))
    return resu4

#Fonction SGD
def fsgd(X,Y):
    start_time = time.time()
    blabla5 = PreText(text=" SGD selected")
    scaler = StandardScaler(with_mean=False, copy=True)
    Feature_norm = scaler.fit_transform(X) #in case different scaled data
    type_feature = np.dtype(Y).name
    if ('int' in type_feature) or ('float' in type_feature):
        #Regression
        X_train, X_test, y_train, y_test = train_test_split(Feature_norm, Y, train_size=0.8)  #or Feature_norm instead of X
        Regressor = SGDRegressor(max_iter=10**7//len(X_train)+1) #Model instance, empirically result for max_iter with 10**6
        params2 = {'alpha':[ 1e-3, 1e-4, 1e-5, 1e-6 ], 'penalty':['l1', 'l2'], 'loss':['squared_loss','huber','epsilon_insensitive']} #Paramètres à tester
        clf = GridSearchCV(Regressor, param_grid=params2, cv=5, n_jobs=6) #,scoring=scorer)
        clf.fit(X_train, y_train) #On applique sur les données d'apprentissage
        #predictions on train data
        y_train_predict=clf.predict(X_train)
        #predictions on test data
        y_pred = clf.predict(X_test)
        print(clf.best_params_)#Meilleur paramètrage
        end_time = time.time()
        exec_time = round(end_time - start_time, 2)
        p5 = figure(title="Prediction de la variable : %s " % select_cible.value)
        p5.circle(range(len(y_pred)), y_pred[np.argsort(y_test)] , fill_alpha=0.8 , color = 'red', legend_label = "Prédiction")#,source=source)  
        p5.line(range(len(y_test)), np.sort(y_test) , color = 'blue', legend_label = "Echantillon test") #données réelles
        p5.plot_width = 900
        resu5 = row(p5, Regression_metrics(clf, X_train,y_train,X_test,y_test, exec_time, blabla5)) #column(blabla5,model_info,pre5,learn))
        return resu5
    else :
        #Classification
        X_train, X_test, y_train, y_test = train_test_split(Feature_norm, Y, train_size=0.8, stratify = Y) #stratify to avoid missing classes
        Classifier = SGDClassifier(max_iter=10**7//len(X_train)+1) #Model instance, empirically result for max_iter with 10**6
        params = {'alpha':[ 1e-3, 1e-4, 1e-5, 1e-6 ], 'penalty':['l1', 'l2'], 'loss':['hinge','log']} #Paramètres à tester
        clf = GridSearchCV(Classifier, param_grid=params, cv=5, n_jobs=6)#scoring=scorer)
        clf.fit(X_train, y_train) #On applique sur les données d'apprentissage
        #predictions on train data
        y_train_predict=clf.predict(X_train)
        #predictions on test data
        y_pred=clf.predict(X_test)
        # Évaluation de l'algorithme
        print(clf.best_params_)
        end_time = time.time()
        exec_time = round(end_time - start_time, 2)
        #Graphique
        categ = y_test.astype("category")
        dico_tmp = dict(enumerate(categ.cat.categories))
        dico = dict((v, k) for k, v in dico_tmp.items())
        #on converti les categories en entier
        y_test_int = y_test.map(dico)
        y_pred_int = pd.Series(y_pred).map(dico)
        dico_bokeh = Paragraph(text=" Dictionnaire des correspondances (clef, valeur) : \n   "+str(dico),  width= 1000)
        #Graphiique
        p5 = figure( title="Prediction de la variable : %s " % select_cible.value, x_axis_label = "x" , y_axis_label = "y")
        p5.circle(range(len(y_test)), y_test_int, fill_alpha=0.8 , color = 'blue', legend_label = "Echantillon test") #données réelles
        p5.circle(range(len(y_pred)), y_pred_int, fill_alpha=0.4 , color = 'red', legend_label = "Prédiction")#,source=source)   
        p5.plot_width = 900
        resu5 = row(column(p5,dico_bokeh), Classification_metrics(clf, X_train,y_train,X_test,y_test, exec_time, blabla5)) #column(blabla5,model_info,pre5,learn))
        return resu5


#Fonction qui renvoie le resultat de l'algo choisie par l'utilisateur
def choix_algo():
    algori = select_algo.value
    fichier = text_input.value #Name data
    data = pd.read_csv(fichier, sep=",") #Import data
    global res
    target= select_cible.value #target label
    features = predictive.value #features labels
    X = data[features] #features
    Y = data[target] #target
    for i in features: 
        type_feature = np.dtype(data[i]).name
        if ('int' not in type_feature) or ('float' not in type_feature):
            X.loc[:,i] = X[i].astype("category").cat.codes    #DONE only if selected & not target
    ### Selection de l'ago
    if algori == 'Ridge' : 
        res = fridge(X,Y)
    elif algori == 'Lasso' : 
        res = flasso(X,Y)
    elif algori == 'KNN' : 
        res = fknn(X,Y)
    elif algori == 'DecisionTree' :
        res = fdectree(X,Y)
    elif algori == 'SGD' :
        res = fsgd(X,Y)    
    layout.children[1] = res    
    return res

file_number=0  #var global
#Fonction qui sauvergarde les figures issues de la fonction choix_algo()
def save_as():
    global file_number
    figure = layout.children[1]
    algo_name = select_algo.value
    export_png(figure, filename=algo_name+"_figure{}.png".format(file_number))
    file_number = file_number + 1

#Fonction qui renvoie le graphique des courbes d'apprentissage du modèle en fonction de la quantité de données
def learning_graph(model, X_train, y_train):
    N, train_score, val_score = learning_curve(model, X_train,y_train, train_sizes = np.linspace(0.1,1.0,10),cv=5)
    learn = figure( title="Learning curves ", x_axis_label = "data quantity" , y_axis_label = "learning rate")
    learn.line(N, train_score.mean(axis=1) , color = 'blue', legend_label = "Train") #training curve
    learn.line(N, val_score.mean(axis=1) , color = 'orange', legend_label = "Validation") #Validation curve
    return learn

#Fonction qui renvoie les métriques d'évaluation dans le cadre d'une classification 
def Classification_metrics(clf,X_train,y_train,X_test,y_test, exec_time, algo_name=PreText(text="")):
    
    y_pred = clf.predict(X_test)
    rep = (classification_report(y_test, y_pred,output_dict=True))
    dfrep1 = pd.DataFrame(rep).transpose()
    dfrep1['index'] = dfrep1.index  
    
    source = ColumnDataSource(dfrep1)
    columns = [TableColumn(field='index'),
               TableColumn(field='precision'),
               TableColumn(field='recall'),
               TableColumn(field='f1-score'),
               TableColumn(field='support')]
    
    tet = DataTable(source=source,columns=columns, index_position=None)
    
    conf = "Confusion matrix : \n"+str(np.array(confusion_matrix(y_test, y_pred)))+" \n Classification report : "
    conf2 = PreText(text=conf, width=550)
    model_info = PreText(text=" Paramètres : "+str(clf.best_params_) + "\n Temps d'execution de : {} sec".format(exec_time))
    learn = learning_graph(clf.best_estimator_,X_train,y_train)
    return column(pre5, algo_name, learn, model_info,conf2,tet)

#Fonction qui renvoie les métriques d'évaluation dans le cadre d'une regression 
def Regression_metrics(clf,X_train,y_train,X_test,y_test, exec_time, algo_name=PreText(text="")):
    y_train_predict = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    model_info= PreText(text=" Paramètres : "+str(clf.best_params_) +" \n Le score R2 sur la base d'apprentissage est : {}".format(r2_score(y_train, y_train_predict)) +" \n Le score R2 sur la base du test est : {}".format(r2_score(y_test, y_pred)) +" \n Temps d'execution de : {} sec".format(exec_time))
    learn = learning_graph(clf.best_estimator_,X_train,y_train)
    return column(pre5, algo_name, learn, model_info)

######################## Création des élements de l'interface ########################
#Nom de l'application
curdoc().title = 'Interface Machine Learning supervisé'


## Elements de gauche | paramétrage
#Choix du fichier
pre1 = PreText(text="""Choisisez un fichier .csv :""")
text_input = TextInput(value="FRvideos.csv")


#Choix de la variable cible
pre2 = PreText(text="""Choisisez la variable cible :""")
select_cible = Select(
    options=var,
    width = 300
)

#Choix des varialbes preds (choix multiple)
pre3 = PreText(text="""Choisisez les variables prédictives :""")
predictive = MultiChoice(options = var, width= 280 , height = 75 )

#Choix de l'algoritme à appliquer
pre4 = PreText(text="""Choisisez l'algorithme à appliquer :""")
select_algo = Select(
    options=['Please select target variable'],
    width = 300
)

# BOUTON VALIDATION
button_valide = Button(label='VALIDER', button_type='success', width = 300)

# BOUTON SAUVEGARDER
button_save = Button(label='SAUVEGARDER', button_type='success', width = 300)
button_save.on_click(save_as)

#Rassemble les élements pour le parametrage
pga = column(pre1, text_input, pre2, select_cible, pre3,predictive, pre4, select_algo, button_valide, button_save, width= 320)


## Elements de droite | résultats des algorithmes
#On récupère le résultats de la fonction
res = column()
pre5 = PreText(text=""" Métriques d'évaluation :""")
pdr = res


######################## Gestion des intéraction entre les élements de l'interface ########################
#Si le texte change on importe les nouvelles données
text_input.on_change("value", importation)

#appelle la fonction callback lorsque la valeur du menu change
select_cible.on_change('value', callback)

#Lance l'algo selectionné par l'utilisateur
button_valide.on_click(choix_algo)


######################## Affichage ########################
desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")

l = row(desc, sizing_mode="scale_both")
layout = row(pga,pdr)
# Affichage du layout
curdoc().add_root(l)
curdoc().add_root(layout)
