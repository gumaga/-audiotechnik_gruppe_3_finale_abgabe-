Spracherkenner mit 3-Band-Kompressor
=================================
*Daniel-José Alcala Padilla, Tobias Danneleit und Leon Hochberger (2021) @ TGM, Jade Hochschule Wilhelmshaven Oldenburg Elsfleth*

---
## Was war das Ziel?
Die Ziel war einen Klassifikator für Sprache zu entwickeln, der mit einem Multiband-Kompressor gekoppelt werden sollte. Die Aufgabe war Teil der Prüfungsleistung im SoSe2021 für das Modul *'Audiotechnik'*. Der Klassifikator sollte in der Lage sein Sprache von Nicht-Sprache zu unterscheiden. Das Signal sollte danach mit einer Filterbank bearbeitet werden und wenn Sprache vorhanden war, komprimiert werden. Die Filterbank und der Kompressor sollten nach  eigenem Ermessen eingestellt werden. 

## Wie wurden diese umgesetzt?
#### *Programmiersprache und benötigte Packages*
Die gewählte Programmiersprache ist Python. Verwendet wurde Python 3.8.3 oder höher. Die zusätzlich benötigten Packages sind (**Fett** markierte packages sind nicht in der Standardbibliothek von Python enthalten):
- os
- pathlib 
- pickle
- **scikit-learn**
- **numpy**
- **scipy**
- **matplotlib**
- **soundfile**
- **python_speech_features**

#### *Kompressor*

Der Kompressor befindet sich in der Datei *compressor.py*. Er basiert auf Vorlesungsunterlagen und dem Paper *'Digital Dynamic Range Compressor Design - A Tutorial and Analysis'* von *Giannoulis et al*. (für weitere Info s. Datei-Header). Weiterhin kann ihm ein Index-Vektor übergeben werden, sodass er ein Signal nur zu bestimmten Zeitpunkten komprimiert. Er wurde hier mit einem Treshold von -20 dB, einer Ratio von 2:1, einer Attack-Zeit von 10 ms, eine Release-Zeit von 50 ms, Softkneeeinstellung und keinem Make-Up Gain eingestellt. Diese Einstellung wurden subjektiv am angenehmsten für Sprache wahrgenommen.   


#### *Filterbank*

Die Filterbank befindet sich in der Datei *filterbank.py*. Sie basiert auf Vorlesungsunterlagen und ist ein 3-Band-Equalizer mit jeweils aneinander angrenzendem Tief-, Band- und Hochpass. Die Grenzfrequenzen des Bandpasses können gewählt werden. In dieser Anwendung wurden sie auf 250 Hz und 4000 Hz gesetzt. Diese Grenzen sind so gelegt, dass der informationshaltige Teil der Sprache durch Absenken der Frequenzbereiche darüber und darunter hervorgehoben werden kann, um evtl. Sprachverständlichkeit zu erhöhen. Auf eine Erhöhung des mittleren Frequenzbereichs wurde verzichtet um clipping zu vermeiden. HIer wurden der Tiefpass und der Hochpass mit einem gain von -12 dB versehen. Auch hier galt wieder, dass diese Einstellung subjektiv am angehmsten wahrgenommen wurde.



... 

#### *Spracherkennung*
Die Spracherkennung wurde u.A. mittels Scikit-learn implementiert. Aus den Daten wurden MFCCs als Features extrahiert. Weiterhein wurde der Crest Faktor also Feature implementiert. Diese beiden Features schienen in bezug auf Sprache eine gute Wahl zu sein. Nach der Extraktion der Features wurde der Klassifikator trainiert. Mit einem kleinen Datenset wurden verschiedene Klassifikatoren auf ihre Fähigkeit das Sprachmaterial zu unterscheiden getestet. Die Wahrscheinlichkeit für richtige Antworten bei einem kleinen Testset, MFCCs als Features und randomisierter Train-Test-Aufteilung des Sets für mehrere Durchläufe sind im Folgenden dargestellt:
- Support Vector Machine:   94.65%, 94,78%, 94,71%, 94,75%
- Gaussian Mixture Models:  56,63%, 55,99%, 56,00%, 56,03%
- Logistic Regression:      81,30%, 81,55%, 81,72%, 81,27%

Es stellte sich heraus, dass die Support Vector Machine (SVM) in Kombination mit den MFCCs am Besten in der Lage war Sprache von nicht Sprache zu unterscheiden. Der Crest-Faktor führte in allen Fällen zu schlechteren Ergebnissen. Für das große Datenset wurde deshalb die SVM mit MFCCs als Feature trainiert. 

warum schlechter?

#### *Verwendete Daten*
Für die Spracherkennung wurde der Klassifikator mit Audiodateien aus der Vorlesung trainiert.
Darin sind Aufnahmen von Sprache, Noise und Musik enthalten.
Aus diesem Datenset wurde ein Datenset zusammengestellt, welches auf eine hohe Diversität des Materials abzielt. Das Sprachmaterial wurde von männlichen und weiblichen Sprechern in unterschiedlichen Sprachen eingesprochen. Musik ist in verschiedensten Genres vertreten. Rauschen ist ebenfalls vorhanden.

***Zubereitung der Daten***:
~~Die Daten aus dem MUSAN-Korpus wurden in 3 s lange Blöcke aufgeteilt und abgespeichert. Die Blöcke von vom Ende einer Datei waren oftmals weniger als 3 s lang. Das verwendete Skript ist *cut_my_file_into_pieces.py*. Es wird nicht zur Verwendung der Klassifikation benötigt, ist aber zu Vollständigkeit im Repository enthalten.~~

~~Das Datenset war folgendermaßen aufgebaut (nur zusammenaddierte Dauer der Audiodateien)~~:

~~Sprache: 9:21 h, Noise: 3:40 h, Musik: 3:53 h~~
Aufgrund der Verwendung eines vorgegebenen, kleineren Datensets musste keine Datenzubereitung stattfinden. Ein großes Datenset konnte leider nicht verwendet werden, da keiner der verwendeten Rechner stark genug gewesen ist. 


## Wie werden die Skripte ausgeführt?
Die Skripte können auf zwei unterschiedliche Arten ausgeführt werden.
In beiden Fällen muss die zu komprimierenden Audiodatei, das *Testfile*, im .wav-Format vorliegen.
Um die Skripte zu nutzen, muss und darf im Ordner  *test_files* **nur ein** *Testfile* abgelegt werden. Falls mehrere Testfiles komprimiert werden sollen, muss dies nacheinander geschehen, sodass immer nur ein File im Ordner ist. Der Ordner *test_files* und die vier Python-Dateien müssen im selben Directory liegen.
In diesem Directory wird nach der Nutzung die komprimierte Version des Testfiles unter dem Namen *test_signal_compressed.wav* abgespeichert.

#### 1. Nutzung mit dem bereits trainierten Klassifikator:
Mit den Skripten wird die Datei *classifier_file.obj* bereitgestellt.
Diese Datei enthält den Klassifikator, wie er mit dem oben genannten Datenset trainiert wurde.
Sie muss im selben Directory wie die vier Python-Skripten und der Ordner *test_files* liegen.
Für die Nutzung wird das Skript *classifier_speech_features_application.py* ausgeführt. Hierzu einfach in einer Python-Konsole diese Datei in dem Ordner in dem sich dieses Repository befindet ausführen.
Um den trainierten Klassifikator nutzen zu können, muss in dieser Datei  *desired_feature = 'mfcc'* stehen (Zeile 33). Bei einem anderen Feature können falsche bzw. unerwartete Ergebnisse entstehen. 

#### 2. Klassifikator wird vom Nutzer selbst trainiert:
Um einen Klassifikator trainieren zu können, muss im Directory eine bestimmte Ordnerstruktur angelegt werden. Sie besteht aus dem Ordner *Dataset*, in welchem die Unterordner *Speech*, *Noise* und *Music* liegen. Die Trainingsdateien werden je nach ihrem Inhalt in den jeweiligen Unterordner abgelegt. So wie das Testfile, müssen auch die Trainingsdateien im .wav-Format vorliegen. Für die Vergleichbarkeit mit dem trainierten Klassifikator sollten die einzelnen Audiodateien etwa 3 s lang sein. Hierzu kann das Skript *cut_my_file_into_pieces.py* verwendet werden (weitere Anweisungen im zugehörigen Datei-Header).

Zunächst wird das Skript *classifier_speech_features.py* ausgeführt.
Im Directory wird dadurch die Datei *classifier_file.obj* neu erstellt, bzw. überschrieben. Falls das Überschreiben des bestehenden Klassifikators  nicht gewünscht ist, sollte der Dateiname dieser Datei geändert werden. Jedoch funktioniert *classifier_speech_features_application.py* nur mit dem Dateinamen *classifier_file.obj*, sodass immer nur der in der Datei enthaltene Klassifikator verwendet wird.
Danach kann analog zu 1.) das Skript *classifier_speech_features_application.py* zum Komprimieren des Testfiles ausgeführt werden.
Auch hier muss beachtet werden, dass das ausgewählte Feature, dem entsprechen muss, mit welchem der Klassifikator trainiert wurde.


## Was kann verbessert werden?
- Klassifikator mit noch mehr Daten in höherer Variabilität trainieren
- Weitere Kombination von Features austesten
- Abändern (vielleicht Absenken) des Signals, wenn keine Sprache vorhanden ist
- Denoising der Sprache
- mehrere Klassifikatoren trainieren und abrufbar machen
- nutzerfreundliche Ordnereingabe und Abfrage bestimmer Aspekte

## Lizenz
***BSD 3-clause***:

Copyright 2021 Alcala Padilla, Danneleit, Hochberger

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may
be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
