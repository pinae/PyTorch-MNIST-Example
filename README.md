# PyTorch-MNIST-Example
Der MNIST-Datensatz besteht aus 60.000 Bildchen von handgeschriebenen Ziffern zum Trainieren und 10.000 zum Testen. 
Die Bilder sind 28×28 Pixel groß und haben nur einen Farbkanal. Der Hintergrund ist komplett schwarz, die Ziffer weiß 
mit Graustufen an den Rändern.

Zu jedem Bild die korrekte Ziffer zuzuordnen ist ein typisches Klassifikationsproblem für Modelle des maschinellen 
Lernens. Fürs Training neuronaler Netze hat sich der Datensatz als Beispiel für Einsteiger etabliert. Die Daten sind 
überschaubar, sodass das Training auch ohne Hardwarebeschleunigung in Minuten statt Stunden fertig wird und der 
Speicherbedarf überfordert günstige Grafikkarten nicht.

Voraussetzung für alle Beispiele in diesem Repository ist eine funktionierende Installation von PyTorch. Für die 
Nutzung mit `pipenv` steht ein [`Pipfile`](https://github.com/pinae/PyTorch-MNIST-Example/blob/main/Pipfile) bereit:

```bash
echo "Repository laden..."
git clone https://github.com/pinae/PyTorch-MNIST-Example.git
cd PyTorch-MNIST-Example
echo "Voraussetzungen installieren..."
pipenv install
```

> [!NOTE]  
> Das `Pipfile` zeigt exemplarisch, wie Sie eine `[[source]]` namens `pytorch` anlegen. Im Beispiel verweist die auf
> PyTorch für CUDA 11.8. Um diese Quelle zu nutzen, fügen Sie `index="pytorch"` hinter `version` bei den Modulen ein. 
> [Weitere Details erklärt die Doku](https://pipenv.pypa.io/en/latest/indexes.html). 

Der Code funktioniert aber auch mit dem `venv`-Modul und `pip`:

```bash
echo "Repository laden..."
git clone https://github.com/pinae/PyTorch-MNIST-Example.git
cd PyTorch-MNIST-Example
echo "virtuelles Environment anlegen..."
python3 -m venv env
source env/bin/activate
echo "Voraussetzungen installieren..."
pip3 install numpy torch torchvision
```

Wer Anaconda bevorzugt, kann `pip3` durch `conda` ersetzen:

```bash
conda install numpy torch torchvision
```

## Loslegen mit wenig Code

In [`train-MNIST.py`](https://github.com/pinae/PyTorch-MNIST-Example/blob/main/train-MNIST.py) finden Sie ein 
Beispiel mit weniger als 100 Zeilen code, das den Datensatz lädt, ein neuronales 
Netz definiert, trainiert und testet. Der folgende Befehl führt es mit `pipenv` aus:

```bash
pipenv run python train-MNIST.py
```

In einem bereits mit `source env/bin/activate` aktivierten Virtualenv reicht der folgende Befehl:

```bash
python train-MNIST.py
```

## Experimentieren mit Übersicht

Wir empfehlen mit dem Code herumzuspielen und dabei auch zu versuchen Hyperparameter und Netzwerkstruktur so zu 
verändern, dass das Training nicht mehr klappt. Damit Sie dabei die nicht die Übersicht verlieren, können sie sich bei 
[Weights&Biases](https://wandb.ai) einen kostenlosen Account erzeugen. Ausgestattet mit dem Token dieses Accounts 
können Sie `train.py` statt `train-MNIST.py` aufrufen. Im Prinzip macht der Code das Gleiche. Die Hyperparameter und
Metriken des Trainings werden aber automatisch mit Weights&Biases synchronisiert, sodass Sie in diesem Webdienst 
leicht vergleichen können, was die änderungen bewirkt haben.

Zusätzlich haben wir in `MultiLayerPerceptron.py` und in `ConvolutionalNetwork.py` je zwei Netzwerke vorbereitet, die
Sie schnell ausprobieren können. Die größeren Netze aus diesen Dateien erreichen nach einer etwas längeren 
Trainingszeit auch eine bessere Erkennungsrate.

## Experimentieren mit Datasets

Um den Umgang mit `Dataset`-Klassen zu erklären, haben wir im Ordner `MNIST_mods` zusätzlichen Code platziert. 
Damit das Beispiel in `MNIST_mods/file_MNIST.py` funktioniert, muss es die MNIST-Daten als einzelne Bilder im Ordner 
`data/MNIST_image_files` geben. Dafür führen Sie mindestens einmal den Code zum Trainieren eines Netzes aus, 
beispielsweise `python train-MNIST.py`. Danach gibt es im Ordner `data` den Unterordner `MNIST`. Um die Daten in 
einzelne Bilder zu konvertieren führen Sie einmal das folgende Skript aus:

```bash
cd MNIST_mods
python write_image_files.py
```

Der Befehl braucht relativ lang, weil er 70.000 Bilddateien auf die Festplatte schreiben muss. Meist müssen Sie ein 
paar Minuten warten.

Anschließend können Sie `MNIST` aus `MNIST_mods.file_MNIST` statt aus `torchvision.datasets` importieren und im 
Prinzip genauso Netze trainieren. Der Code läuft allerdings etwas langsamer...
