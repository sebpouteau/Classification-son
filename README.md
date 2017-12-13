# Classification-son
Étude de classification sonore

Rapport ici : https://github.com/sebpouteau/Classification-son/wiki/Rapport
## Installation
- python3.36
- Tensorflow lien ici : https://www.tensorflow.org/install/
- Librosa lien ici : https://librosa.github.io/


## Contenu

Dans SRC, les différents scripts python utilisé


- function.py : fichier avec les fonctions utiles
- comp.py : compare la ressemblance entre deux fichiers csv résultants
```
python comp.py <file1> <file2>
```
- mix.py : script permettant de mélanger 5 fichiers résultats
```
python mix.py <file1> <file2> <file3> <file4> <file5> <out>
```
- train.py : script permettant d'entraîner un réseau de neurones
```
 python3 knn.py  <train_features> <train_label> <eval_feature> <start_features> <end_features> <nbTrain> <nbEpoch> <nbClasse> <save_dest>
```
- eval.py : script permettant d'évaluer notre réseau de neurones et générer le fichier de soumission
```
python3 knn.py <model> <eval_feature> <eval_id_music> <start_features> <end_features> <save_dest>
```
- knn.py : script permettant soit d'observer les performances du KNN, soit de générer les fichiers de soumission
```
python3 knn.py <iftrain> <train_features> <train_label> <eval_feature> <eval_id_music> <start_features> <end_features> <nbTrain> <nbVoisinage> <save_dest>
```
- trainBinary.py : script en cours de développement pour essayer l'approche 1 contre tous
```
python3 knn.py  <train_features> <train_label> <eval_feature> <start_features> <end_features> <nbTrain> <nbEpoch> <nbClasse> <save_dest>
```
- evalBinary.py : script en cours de développement permettant d'évaluer notre réseau de neurones et générer le fichier de soumission
```
python3 knn.py <model> <eval_feature> <eval_id_music> <start_features> <end_features> <save_dest>
```

