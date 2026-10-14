# Inference Pipeline
# ==================
# Sagt für die aktuelle Stunde vorher, ob es in Basel in den nächsten
# 2 Stunden regnet: Muss ich einen Schirm einpacken?
#
# Ausführen:
#     python feature_pipeline.py     (holt die neuesten Wetterdaten)
#     python inference_pipeline.py   (macht die Vorhersage)
#
# Oben stehen die Funktionen, ganz unten der Hauptteil, der sie der Reihe
# nach aufruft.

import os
from datetime import datetime, timedelta, timezone

import hopsworks
import joblib
from dotenv import load_dotenv


def bei_hopsworks_anmelden():
    """Meldet sich bei Hopsworks an und gibt das Projekt zurück."""
    # Der API-Key und der Projektname stehen in der Datei .env
    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT"),
    )
    return project


# Schritt 3.1: Allfällige Live-Daten abrufen
# Dafür gibt es hier keine Funktion. Die Live-Daten holt die Feature-Pipeline
# von Open-Meteo und schreibt sie in den Featurestore. Deshalb muss
# feature_pipeline.py direkt vorher laufen. Hier werden keine Daten an
# Hopsworks vorbei geholt.


def neueste_features_laden(fs):
    """Schritt 3.2: Feature View holen und mit der View die neuesten Features
    laden."""
    feature_view = fs.get_feature_view(name="weather_basel_view", version=1)

    # Nur die Stunden der letzten 24 Stunden laden, nicht alles
    # Grosszügig damit wir sicher gehen dass wir die neueste Stunde auch bekommen
    seit = datetime.now(timezone.utc) - timedelta(hours=24)
    features = feature_view.get_batch_data(start_time=seit)

    if len(features) == 0:
        print("Keine Daten aus den letzten 24 Stunden gefunden.")
        print("Bitte zuerst feature_pipeline.py laufen lassen.")
        raise SystemExit(1)

    # Nach der Zeit sortieren und die neueste Stunde nehmen
    features = features.sort_values("time")
    neueste = features.tail(1)

    # Gerechnet wird überall in UTC. Nur für die Anzeige wird die Zeit in die
    # Schweizer Zeit umgerechnet (Sommerzeit UTC+2, Winterzeit UTC+1).
    zeit_schweiz = neueste["time"].iloc[0].tz_convert("Europe/Zurich")
    print("Neueste Stunde im Featurestore:", zeit_schweiz.strftime("%d.%m.%Y %H:%M"), "(Schweizer Zeit)")
    print("  Bewölkung:                   ", neueste["cloud_cover"].iloc[0], "%")
    print("  Luftfeuchtigkeit Mittel 24 h:", round(neueste["humidity_mean_24h"].iloc[0], 1), "%")

    # Ist die neueste Stunde älter als 2 Stunden, wären "die nächsten
    # 2 Stunden" schon vorbei. Dann lieber abbrechen als falsch vorhersagen.
    alter = datetime.now(timezone.utc) - neueste["time"].iloc[0]
    if alter > timedelta(hours=2):
        print("\nDie neueste Stunde ist", round(alter.total_seconds() / 3600, 1), "Stunden alt, das ist zu alt.")
        print("Bitte zuerst feature_pipeline.py laufen lassen.")
        raise SystemExit(1)

    return neueste


def modell_laden(project):
    """Schritt 3.3: Modell aus der Model Registry herunterladen."""
    mr = project.get_model_registry()

    # Alle Versionen von rain_model holen und die neueste nehmen
    alle_versionen = mr.get_models(name="rain_model")
    if len(alle_versionen) == 0:
        print("Es gibt noch kein Modell. Bitte zuerst training_pipeline.py laufen lassen.")
        raise SystemExit(1)

    rain_model = alle_versionen[0]
    for eintrag in alle_versionen:
        if eintrag.version > rain_model.version:
            rain_model = eintrag
    print("\nVerwende rain_model Version", rain_model.version)

    modell_ordner = rain_model.download()
    model = joblib.load(modell_ordner + "/model.pkl")
    return model


def vorhersage_machen(model, neueste):
    """Schritt 3.4: Prediction durchführen."""
    # Das Modell bekommt genau die zwei Features, mit denen es trainiert wurde
    X = neueste[["cloud_cover", "humidity_mean_24h"]]
    vorhersage = model.predict(X)[0]   # INhalt nehmen: Element 0, ohne [0] wäre es ein Array mit 1 Element

    print()
    print("==============================================")
    if vorhersage == 1:
        print("  Regen in den nächsten 2 Stunden erwartet.")
        print("  -> Schirm einpacken!")
    else:
        print("  Kein Regen in den nächsten 2 Stunden erwartet.")
        print("  -> Kein Schirm nötig.")
    print("==============================================")


# ------------------------------------------------------------------
# Hauptteil: hier laufen die Schritte der Reihe nach ab
# ------------------------------------------------------------------

project = bei_hopsworks_anmelden()
fs = project.get_feature_store()

neueste = neueste_features_laden(fs)
model = modell_laden(project)
vorhersage_machen(model, neueste)
