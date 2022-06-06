import json
import zipfile
import pickle
import pandas as pd
import os

def load_files(path):
    path_extracted = os.getcwd()+'/extracted'
    os.makedirs(path_extracted, exist_ok=True)
    # Extraemos los ficheros del zip
    with zipfile.ZipFile(path) as zip_ref:
        zip_ref.extractall(path_extracted)
    try:
        # Leemos el modelo
        with open(path_extracted+'/model_v0', 'rb') as f:
            my_model0 = pickle.load(f)

        # Leemos el dataset de test
        df_test = pd.read_csv(path_extracted+'/test.csv')
        if df_test.columns[0]=='Unnamed: 0':
            df_test.drop(columns='Unnamed: 0', inplace=True)

        # Leemos el json con la info
        with open(path_extracted+'/info.json') as json_file:
            info = json.load(json_file)
    except:
        print("error de lectura de archivos")

    for archivo in os.listdir(path_extracted):
        os.remove(path_extracted+'/'+archivo)

    return my_model0, df_test, info

def test_results(my_model0, df_test, info):
    try:
        if info['tipo_ml'] in ('C','R'):
            # Dividimos por la target
            X_test=df_test.drop(columns=[info['target']])
            y_test=df_test[[info['target']]]
            # Calculamos el score del modelo
            score = my_model0.score(X_test,y_test)
        elif info['tipo_ml'] in ('ST'):
            score = my_model0.score(len(df_test))
        elif info['tipo_ml'] in ('O'):
            score = "Otro score"
        else:
            score = "No se puede calcular el score"

        # Guardamos los resultados
        df_results = pd.json_normalize(info)
        df_results['score'] = score

        df_results_tot = pd.read_csv('resultados_modelos.csv')
        df_results_tot.append(df_results)
        df_results_tot.to_csv('resultados_modelos.csv')
    except:
        print("Error en la evaluación de los archivos")
 