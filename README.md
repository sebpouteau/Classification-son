# Classification-son
Étude de classification sonore


## Utile
- python3.36
- Tensorflow lien ici : https://www.tensorflow.org/install/
- Librosa lien ici : https://librosa.github.io/


## Contenu

Dans SRC, les différents script python utilisé

- function.py : fichier avec les fonctions utiles
- comp.py : compare la ressemblance entre deux fichier csv résultat
```
python comp.py <file1> <file2>
```
- mix.py : script permettant de mélanger 5 fichiers résultats
```
python mix.py <file1> <file2> <file3> <file4> <file5> <out>
```
- train.py : script permettant d'entrainer un réseau de neurone
```
 python3 knn.py  <train_features> <train_label> <eval_feature> <start_features> <end_features> <nbTrain> <nbEpoch> <nbClasse> <save_dest>
```
- eval.py : script permettant d'évaluer notre réseau de neurone et générer le fichier de soumission
```
python3 knn.py <model> <eval_feature> <eval_id_music> <start_features> <end_features> <save_dest>
```
- knn.py : script permettant soit d'observer les performance du knn, soit de générer les fichier de soumission
```
python3 knn.py <iftrain> <train_features> <train_label> <eval_feature> <eval_id_music> <start_features> <end_features> <nbTrain> <nbVoisinage> <save_dest>
```
- trainBinary.py : script en cours de développement pour essayer l'approche 1 contre tous
```
python3 knn.py  <train_features> <train_label> <eval_feature> <start_features> <end_features> <nbTrain> <nbEpoch> <nbClasse> <save_dest>
```
- evalBinary.py : script en cours de développement permettant d'évaluer notre réseau de neurone et générer le fichier de soumission
```
python3 knn.py <model> <eval_feature> <eval_id_music> <start_features> <end_features> <save_dest>
```

