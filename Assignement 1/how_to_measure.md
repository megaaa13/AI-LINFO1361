# Comment mesurer la performance avec *timeout* et imprimer les informations utiles ?

## Automatiser les mesures
Le script shell suivant permet d'automatiser la prise de mesure. Pour ce faire, le script utilise 4 versions modifiées du fichier `tower_sorting.py`, chacun utilisiant une version différente de l'algorithme de recherche.
```bash
#!/bin/bash

MAX_TIME=3m
MEASURE_FILE=measure.txt

# Reset the measure file
echo "Starting measurement" > ${MEASURE_FILE}
echo "Starting measurement"
echo "Starting measurement. This may take a while (max 120m)"

# Measure function
measure() {
    echo "" >> ${MEASURE_FILE}
    echo "Measuring $2 with $3" >> ${MEASURE_FILE}
    timeout -s 10 ${MAX_TIME} python3 $1 $2 >> ${MEASURE_FILE}
}

# Measure all the files
export PYTHONPATH=aima-python3/
for file in instances/* ; do
    echo "Measuring $file"
    measure tower_sorting_bfsg.py $file bfsg
    measure tower_sorting_bfst.py $file bfst
    measure tower_sorting_dfsg.py $file dfsg
    measure tower_sorting_dfst.py $file dfst
done
```
Ce script prend en entrée un fichier contenant les instances de problème, et pour chaque instance, il lance 4 versions différentes de l'algorithme de recherche. Pour chaque version, le script utilise la commande `timeout` pour limiter le temps d'exécution à 3 minutes. Si l'algorithme n'a pas terminé, la commande `timeout` envoie le signal `SIGUSR1` au processus python. Le script redirige la sortie standard et la sortie d'erreur vers un fichier `measure.txt` qui contient les informations utiles pour la suite.

## Terminer le processus python
Afin de récolter les informations utiles, on ne peut pas simplement tuer le processus python avec `SIGKILL` ou `SIGTERM`. Il va donc falloir définir le comportement du processus python lorsqu'il reçoit le signal `SIGUSR1`.

### 1. Créer un fichier de gestion du signal
Créer le fichier `breaking.py` dans le dossier `aima-pyhton3` qui contient le code suivant :
```python
import signal

class SIGINT_handler():
    def __init__(self):
        self.SIGUSR1 = False

    def signal_handler(self, signal, frame):
        print("CAUTION: Timout received !")
        self.SIGUSR1 = True

handler = SIGINT_handler()
signal.signal(signal.SIGUSR1, handler.signal_handler)
```
Ce fichier permet de définir un gestionnaire de signal qui va permettre de gérer le signal `SIGUSR1` et de définir une variable `SIGUSR1` qui sera utilisée pour arrêter l'algorithme.

###  2. Modifier le fichier `search.py`
Importer le fichier `breaking.py` dans le fichier `search.py` et modifier les fonctions utillisée par les algorithmes de recherche pour qu'elle arrête l'algorithme lorsqu'elle reçoit le signal `SIGUSR1`.
```python
from breaking import *
# ...
# Dans les boucles while des algorithmes de recherche
if handler.SIGUSR1:
    break
```
### 3. Modifier les fichiers `tower_sorting.py`
Il reste a modifier les fichiers `tower_sorting_XXXX.py` pour qu'ils ne plantent pas lorsque l'algorithme ne trouve pas de solution. Pour ce faire, il faut modifier la fonction `main` de chaque fichier en chaneant les dernières lignes de code. par:
```python
# Example of print
# path = node.path()

# for n in path:
    # print(n.state)
    # pass

print("* Execution time:\t", str(end_timer - start_timer))
try:
    print("* Path cost to goal:\t", node.depth, "moves")
except:
    print("* Path cost to goal:\t", "None")
print("* #Nodes explored:\t", nb_explored)
print("* Queue size at goal:\t",  remaining_nodes)
```
