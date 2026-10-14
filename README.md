# Regenschirm Basel — Projektarbeit MLOps

Muss ich einen Schirm einpacken?
Das Projekt sagt vorher, ob es in Basel in den nächsten 2 Stunden regnet.

Umgesetzt als FTI-Architektur (Feature / Training / Inference) mit drei
Pipelines und dem Hopsworks Featurestore.

## Inhalt

1. [Daten](#1-daten)
2. [Features und Label](#2-features-und-label)
3. [Modell](#3-modell)
4. [Die drei Pipelines](#4-die-drei-pipelines)
5. [Setup](#5-setup)
6. [Ausführen](#6-ausführen)
7. [Limitationen](#7-limitationen)
8. [Verwendung von KI-Werkzeugen](#8-verwendung-von-ki-werkzeugen)

---

## 1. Daten

Quelle: [Open-Meteo Forecast API](https://open-meteo.com/en/docs), gratis und
ohne API-Key.

| | |
|---|---|
| Ort | Basel (47.56 N, 7.59 E) |
| Auflösung | stündlich |
| Zeitraum | die letzten 92 Tage bis zur aktuellen Stunde (`past_days=92`) |
| Werte | `precipitation` (Regen in mm), `relative_humidity_2m` (Luftfeuchtigkeit in %), `cloud_cover` (Bewölkung in %) |

Historische und aktuelle Daten kommen aus demselben Aufruf:

- Stunden, bei denen man schon weiss, ob es danach geregnet hat, sind die
  historischen Daten fürs Training.
- Die neuesten 2 Stunden haben noch kein Label, weil ihre Zukunft noch
  nicht bekannt ist. Das sind die aktuellen Daten für die Inferenz.

Open-Meteo liefert auch Vorhersagen für die kommenden Stunden. Diese werden
weggeworfen, verwendet werden nur Stunden bis jetzt.

### Zeitzone: intern UTC, angezeigt in Schweizer Zeit

Alle Zeiten werden in **UTC** abgefragt, gespeichert und verglichen. Die
Schweiz hat nicht UTC, sondern UTC+1 (Winterzeit) bzw. UTC+2 (Sommerzeit).
Trotzdem ist UTC hier die richtige Wahl:

- Keine Zeitumstellung: In Schweizer Zeit gibt es Ende Oktober die Stunde
  02:00–03:00 zweimal und Ende März fehlt eine Stunde. `rolling(24)`,
  `shift(-1)` und `shift(-2)` zählen Zeilen. Bei einer doppelten oder
  fehlenden Stunde wären "die letzten 24 Stunden" bzw. "die nächsten
  2 Stunden" falsch, und der Primary Key `hour_id` wäre nicht mehr eindeutig.
- Alle beteiligten Systeme verwenden UTC. Open-Meteo liefert UTC
  (`"timezone": "UTC"` in `feature_pipeline.py`), Hopsworks speichert
  Zeitstempel in UTC (sichtbar am `+00:00`).
- Vergleiche stimmen immer. Die Inferenz prüft, ob die neueste Stunde
  älter als 2 Stunden ist, mit `datetime.now(timezone.utc)`. Beide Seiten
  sind UTC, die Differenz stimmt also auch an den Tagen der Zeitumstellung.

Umrechnung für die Anzeige: Nur die Ausgabe der Inferenz zeigt die Zeit
in Schweizer Zeit. Dafür wird der UTC-Zeitstempel mit pandas umgerechnet:

```python
zeit_schweiz = neueste["time"].iloc[0].tz_convert("Europe/Zurich")
```

`"Europe/Zurich"` ist der Name der Schweizer Zeitzone. pandas weiss selbst,
ob an diesem Datum Sommer- oder Winterzeit gilt. So wird z. B. 16:00 UTC am
13.10.2026 zu 18:00, am 26.10.2026 (nach der Umstellung) aber zu 17:00. Die
Tabelle der Feature-Pipeline zeigt dagegen die gespeicherten Werte und bleibt
in UTC (`Die neuesten Zeilen (Zeiten in UTC)`).

## 2. Features und Label

| Name | Art | Beschreibung |
|---|---|---|
| `cloud_cover` | **Echtzeit-Feature (RT)** | Bewölkung zur aktuellen Stunde in %. Ist erst zum Zeitpunkt der Vorhersage bekannt. |
| `humidity_mean_24h` | **aggregiertes Feature** | Mittelwert der Luftfeuchtigkeit über die letzten 24 Stunden (`rolling(24).mean()`, nur Vergangenheit) |
| `rain_next_2h` | **Label** | 1, wenn in den 2 Stunden danach mehr als 0.1 mm Regen fällt, sonst 0 |

Das Label wird mit `shift(-1)` und `shift(-2)` gebildet, also aus den zwei
Stunden nach der Zeile. Die Features schauen dagegen nur zurück.

Gespeichert wird alles in der Feature Group `weather_basel` (Version 1):

- **Primary Key:** `hour_id`, die Stunde als Text (z. B. `"2026-10-13 13:00:00"`),
  eine Zeile pro Stunde. Läuft die Feature-Pipeline nochmals, wird eine
  vorhandene Stunde überschrieben statt doppelt gespeichert. So bekommen die
  neuesten Stunden beim nächsten Lauf ihr Label.
- **Event Time:** `time`

Warum nicht einfach `time` als Primary Key? Das war die erste Version. Beim
zweiten Lauf der Feature-Pipeline brach der Hopsworks-Job dann ab
(`ClassCastException: java.time.Instant cannot be cast to java.lang.Long`):
Mit einer Zeitspalte als Primary Key kann Hopsworks im Format HUDI eine schon
gespeicherte Zeile nicht überschreiben. Mit dem Text-Key funktioniert es.

## 3. Modell

Logistische Regression: (scikit-learn) mit `class_weight="balanced"`.

Regen ist selten: Nur in etwa 7 % der Stunden regnet es in den 2 Stunden
danach. Ohne `class_weight="balanced"` lernt das Modell, immer "kein Regen" zu
sagen, und ist damit trotzdem zu über 90 % "richtig". Mit der Einstellung
zählen die Regenstunden beim Training stärker.

Aufteilung: 80 % Training, 20 % Test (`train_test_split`, zufällig).

### Ergebnis

Lauf vom 14.10.2026 (`rain_model` Version 8), 1813 Stunden mit Label
(1450 Training, 363 Test):

| | Wert |
|---|---|
| Accuracy des Modells | 0.625 |
| Accuracy, wenn man immer "kein Regen" sagt | 0.934 |
| Anteil erkannter Regenstunden (Recall) | 1.0 |
| Anteil "Schirm einpacken", der stimmt (Precision) | 0.15 |

Die Accuracy des Modells ist tiefer als bei "immer kein Regen". Das ist
gewollt: Das Modell schlägt lieber einmal zu oft einen Schirm vor, erkennt
dafür aber die Regenstunden (in diesem Lauf alle im Test). Ein Modell, das nie Regen vorhersagt,
hätte eine hohe Accuracy, wäre aber nutzlos.

Der Preis dafür steht in der letzten Zeile: Nur etwa jede siebte Empfehlung
"Schirm einpacken!" trifft wirklich zu. Mit nur zwei Features kann das Modell
"bewölkt und feucht" kaum von "es regnet gleich" unterscheiden.

Mit denselben Daten ergibt das Training jedes Mal genau dieselben Zahlen
(zweimal hintereinander geprüft). Dafür werden die Zeilen vor dem Split nach
der Zeit sortiert: Hopsworks liefert sie nicht immer in derselben Reihenfolge,
und vorher schwankte die Accuracy bei fast gleichen Daten zwischen 0.603 und
0.668. Kommen neue Stunden dazu, ändern sich die Zahlen natürlich. Bei nur
rund 24 Regenstunden im Test macht schon eine Stunde mehr oder weniger viel
aus.

Die Modellgüte wird laut Aufgabenstellung nicht bewertet.

## 4. Die drei Pipelines

```
feature_pipeline.py      Open-Meteo -> Features + Label -> Feature Group "weather_basel"
        |
        v
training_pipeline.py     Feature Group -> Feature View -> Training Dataset
                         -> Modell -> Model Registry "rain_model"
        |
        v
inference_pipeline.py    Feature View -> neueste Stunde -> Modell -> "Schirm ja/nein"
```

Jede Datei hat eine Funktion pro Schritt der Aufgabenstellung. Der
Docstring jeder Funktion nennt die Schrittnummer (`Schritt 1.1` bis
`Schritt 3.4`). Ganz unten in jeder Datei steht ein kurzer Hauptteil, der die
Funktionen der Reihe nach aufruft.

### Feature Pipeline (`feature_pipeline.py`)

1. `wetterdaten_holen()`: Wetterdaten von Open-Meteo abrufen
2. `features_berechnen()`: Feature `humidity_mean_24h` berechnen
3. `label_berechnen()`: Label `rain_next_2h` berechnen
4. `feature_group_erstellen()`: Feature Group `weather_basel` erstellen
   (Primary Key `hour_id`, Event Time `time`)
5. `daten_einfuegen()`: Daten in die Feature Group schreiben

### Training Pipeline (`training_pipeline.py`)

1. `feature_group_laden()`: Feature Group `weather_basel` laden
2. `feature_view_erstellen()`: Feature View `weather_basel_view` erstellen.
   Die Query wählt `time`, die zwei Features und das Label aus. `time` wird
   für die Inferenz gebraucht, geht aber nicht ins Modell.
3. `trainingsdaten_holen()`: Training Dataset aus der Feature View holen,
   Zeilen ohne Label weglassen, 80/20 aufteilen
4. `modell_trainieren()` und `modell_bewerten()`: logistische Regression
   trainieren und auf den Testdaten bewerten (Accuracy, Recall, Precision)
5. `modell_speichern()`: Modell mit `joblib.dump` lokal speichern und in die
   Model Registry hochladen (`rain_model`)

Jeder Lauf erzeugt eine neue Modellversion.

### Inference Pipeline (`inference_pipeline.py`)

1. Die Live-Daten holt die Feature-Pipeline, die direkt vorher läuft (dafür
   gibt es in dieser Datei keine eigene Funktion)
2. `neueste_features_laden()`: Feature View holen und mit `get_batch_data`
   die Stunden der letzten 24 Stunden laden, davon die neueste nehmen
3. `modell_laden()`: die neueste Version von `rain_model` aus der Model
   Registry herunterladen
4. `vorhersage_machen()`: Vorhersage machen und ausgeben

Beispielausgabe:

```
Neueste Stunde im Featurestore: 14.10.2026 15:00 (Schweizer Zeit)
  Bewölkung:                    100.0 %
  Luftfeuchtigkeit Mittel 24 h: 79.9 %

Verwende rain_model Version 8

==============================================
  Regen in den nächsten 2 Stunden erwartet.
  -> Schirm einpacken!
==============================================
```

## 5. Setup

### Python 3.12

Hopsworks läuft nicht mit Python 3.14 oder neuer. Getestet mit Python 3.12.

```bash
python --version
```

### Pakete installieren

Mit conda:

```bash
conda create -n regenschirm python=3.12
conda activate regenschirm
python -m pip install -r requirements.txt
```

Oder mit venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`python -m pip` statt nur `pip` verwenden. So wird sicher das richtige Python
genommen, falls mehrere installiert sind.

`requirements.txt` enthält nicht `hopsworks[python]`, sondern die einzelnen
Pakete daraus. Der Grund: `hopsworks[python]` installiert auch
`hops-deltalake`, das sich auf ARM-Rechnern (z. B. Raspberry Pi) nicht
installieren lässt. Dieses Projekt braucht es nicht, weil die Feature Group
das Format HUDI verwendet.

### Hopsworks

1. Auf [hopsworks.ai](https://hopsworks.ai) einen Account und ein leeres
   Projekt anlegen.
2. Einen API-Key erstellen: oben rechts auf den Namen klicken, dann
   *Account Settings → API keys*. Alle Scopes anhaken. Wichtig: auch
   `serving`, sonst bricht schon der Login mit einer unverständlichen
   Fehlermeldung ab.
3. Zugangsdaten eintragen:

```bash
cp .env.example .env
```

Dann in `.env` den API-Key und den Projektnamen eintragen. Die Datei `.env`
steht in `.gitignore` und wird nicht hochgeladen.

## 6. Ausführen

In **jedem neuen Terminal** zuerst die Umgebung aktivieren und in den
Projektordner wechseln. Sonst fehlen die Pakete und es kommt z. B.
`ModuleNotFoundError: No module named 'hopsworks'`:

```bash
conda activate regenschirm      # bzw. source .venv/bin/activate
cd fhnw_regenschirm
```

Einmal am Anfang alle drei in dieser Reihenfolge:

```bash
python feature_pipeline.py      # ca. 5 Minuten, Hopsworks startet einen Job
python training_pipeline.py
python inference_pipeline.py
```

Für eine neue Vorhersage später:

```bash
python feature_pipeline.py      # neueste Wetterdaten in den Featurestore
python inference_pipeline.py
```

Ist die neueste Stunde im Featurestore älter als 2 Stunden, bricht die
Inferenz mit einem Hinweis ab. Sonst würde sie "die nächsten 2 Stunden" für
einen Zeitraum vorhersagen, der schon vorbei ist:

```
Neueste Stunde im Featurestore: 13.10.2026 18:00 (Schweizer Zeit)
  Bewölkung:                    100.0 %
  Luftfeuchtigkeit Mittel 24 h: 68.4 %

Die neueste Stunde ist 4.6 Stunden alt, das ist zu alt.
Bitte zuerst feature_pipeline.py laufen lassen.
```

Das Modell neu trainieren (mit den neuesten Daten):

```bash
python training_pipeline.py
```

### Wenn die Feature-Pipeline mit "The Hopsworks Job failed" abbricht

Das Schreiben in die Feature Group übernimmt ein Job in Hopsworks. Am
14.10.2026 brach dieser Job einmal mit einem internen HUDI-Fehler ab
(`ArrayIndexOutOfBoundsException` beim Aktualisieren der Verwaltungsdaten der
Tabelle), ohne dass sich am Code oder an den Daten etwas geändert hatte. Ein
zweiter Lauf direkt danach funktionierte, und die Feature Group war danach
vollständig (keine fehlenden Stunden, keine Duplikate).

Darum: zuerst einfach `python feature_pipeline.py` nochmals starten. Bricht
es wieder ab, in Hopsworks unter *Jobs →
weather_basel_1_offline_fg_materialization* das Log der letzten Ausführung
ansehen.

## 7. Limitationen

- Wenig Daten: Open-Meteo liefert über diesen Aufruf höchstens 92 Tage,
  davon waren beim Test nur etwa 75 Tage mit Werten. Das sind keine ganzen
  Jahreszeiten: Ein Modell aus Sommer und Herbst kennt keinen Winter.
- Modelldaten, keine Messungen: Open-Meteo liefert Werte aus
  Wettermodellen für einen Gitterpunkt, nicht die Messung einer Wetterstation
  in Basel.
- Nur zwei Features: Bewölkung und Luftfeuchtigkeit sagen Regen nur grob
  voraus. Wind, Luftdruck oder Regen in der Umgebung würden helfen, sind aber
  bewusst weggelassen, damit das Projekt einfach bleibt.
- Zufälliger Split: Die Testdaten sind zufällig ausgewählt. Benachbarte
  Stunden sind sich sehr ähnlich, deshalb landet oft eine fast gleiche Stunde
  im Training. Das Testergebnis ist dadurch etwas zu optimistisch. Ehrlicher
  wäre ein Split nach Zeit (die letzten Wochen als Test).
- Ja/Nein statt Wahrscheinlichkeit: Wegen `class_weight="balanced"` gibt
  das Modell keine verlässliche Regenwahrscheinlichkeit aus, sondern nur eine
  Entscheidung (Schirm ja oder nein).
- Die Feature-Pipeline muss vor der Inferenz laufen: Sonst ist die
  neueste Stunde im Featurestore veraltet. Ist sie älter als 2 Stunden,
  bricht die Inferenz mit einem Hinweis ab, statt für einen schon vergangenen
  Zeitraum vorherzusagen. Automatisch geplant (z. B. stündlich) ist der Lauf
  nicht, man muss ihn von Hand starten.
- Zeilen mit fehlenden Werten: werden gelöscht. Bei Open-Meteo fehlen
  Werte bisher nur ganz am Anfang des Zeitraums. Fehlten sie mitten drin,
  würde das 24-Stunden-Fenster über die Lücke hinweg rechnen.
- Immer das neueste Modell: Die Inferenz nimmt die neueste Version von
  `rain_model`, ohne zu prüfen, ob sie besser ist als die vorherige. Ein
  schlechteres neues Training würde also ohne Warnung verwendet.
- Nur auf einem Rechner getestet: Installation und alle drei Pipelines
  wurden mit Python 3.12 unter Linux auf einem ARM-Rechner (Raspberry Pi)
  getestet. Unter Windows, macOS oder in GitHub Codespaces (x86) wurde das
  Projekt nicht ausprobiert.

## 8. Verwendung von KI-Werkzeugen

Für dieses Projekt habe ich Claude (Anthropic) als Assistenten eingesetzt,
vor allem für:

**Code-Generierung:** Verbesserung und Überarbeitung der drei Pipelines.

**Fehlersuche:** z. B. bei der Installation von Hopsworks auf ARM und bei
der fehlenden Zeitspalte in der Feature View

Die Wahl des Themas nach dem Beispiel aus der Aufgabenstellung, des Orts, der
Datenquelle und des Aufbaus stammt von mir.
Am Ende habe ich Claude noch weitere Limitationen suchen lassen (siehe Limitationen: ab 4. Limitation)
Verantwortlich für den gesamten Inhalt bin ich.