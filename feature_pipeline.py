# Feature Pipeline
# ================
# Holt die Wetterdaten für Basel von Open-Meteo, berechnet die Features und
# das Label und speichert alles in einer Feature Group im Hopsworks
# Featurestore.
#
# Ausführen:  python feature_pipeline.py
#
# Die Pipeline darf man beliebig oft laufen lassen. Stunden, die schon in der
# Feature Group sind, werden überschrieben (gleicher Primary Key), neue
# Stunden kommen dazu. Vor jeder Vorhersage sollte sie einmal laufen, damit
# die neueste Stunde im Featurestore ist.
#
# Oben stehen die Funktionen, ganz unten der Hauptteil, der sie der Reihe
# nach aufruft.

import os
from datetime import datetime, timezone

import hopsworks
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

def wetterdaten_holen():
    """Schritt 1.1: Rohdaten via API von der Quelle abrufen."""
    print("Lade Wetterdaten von Open-Meteo ...")

    url = "https://api.open-meteo.com/v1/forecast"
    parameter = {
        "latitude": 47.56,   # Basel
        "longitude": 7.59,
        "hourly": "precipitation,relative_humidity_2m,cloud_cover",
        "past_days": 92,     # so weit zurück wie möglich, 92 Tage ist das Maximum
        "forecast_days": 1,
        "timezone": "UTC",
    }

    antwort = requests.get(url, params=parameter, timeout=60)
    antwort.raise_for_status()   # bricht ab, falls der Server einen Fehler meldet
    daten = antwort.json()

    df = pd.DataFrame(daten["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    print("Geladene Stunden:", len(df))

    # Open-Meteo liefert auch eine Vorhersage für die kommenden Stunden.
    # Die brauchen wir nicht, wir nehmen nur Stunden bis jetzt.
    jetzt = datetime.now(timezone.utc).replace(tzinfo=None)
    df = df[df["time"] <= jetzt]

    # Ganz am Anfang fehlen bei Open-Meteo einige Werte. Diese Zeilen löschen.
    df = df.dropna()
    print("Stunden mit Werten bis jetzt:", len(df))

    # Alle Werte als Kommazahl speichern. Sonst kann es passieren, dass eine
    # Spalte einmal als ganze Zahl und einmal als Kommazahl ankommt, und dann
    # passt sie beim nächsten Lauf nicht mehr zur Feature Group.
    df["precipitation"] = df["precipitation"].astype(float)
    df["relative_humidity_2m"] = df["relative_humidity_2m"].astype(float)
    df["cloud_cover"] = df["cloud_cover"].astype(float)

    return df

def features_berechnen(df):
    """Schritt 1.2: Features engineeren."""
    # Aggregiertes Feature: mittlere Luftfeuchtigkeit der letzten 24 Stunden.
    # rolling(24) nimmt die aktuelle Zeile und die 23 Zeilen davor, also nur
    # Werte aus der Vergangenheit.
    df["humidity_mean_24h"] = df["relative_humidity_2m"].rolling(24).mean()

    # Echtzeit-Feature (RT): die Bewölkung zur aktuellen Stunde in Prozent.
    # Diese Spalte (cloud_cover) kommt direkt von Open-Meteo.

    # Die ersten 23 Zeilen haben noch keine 24 Stunden davor, dort ist der
    # Mittelwert leer. Diese Zeilen löschen.
    df = df.dropna(subset=["humidity_mean_24h"])

    return df

def label_berechnen(df):
    """Schritt 1.3: Dataframe mit Features und Label erstellen."""
    # Label: Regnet es in den nächsten 2 Stunden? (1 = ja, 0 = nein)
    # precipitation ist die Regenmenge in mm während der Stunde davor.
    # shift(-1) holt den Wert aus der nächsten Zeile, shift(-2) aus der
    # übernächsten. Zusammen ist das der Regen der nächsten 2 Stunden.
    regen_naechste_2h = df["precipitation"].shift(-1) + df["precipitation"].shift(-2)
    df["rain_next_2h"] = (regen_naechste_2h > 0.1).astype(float)

    # Für die neuesten 2 Stunden liegt die Zukunft noch nicht vor, das Label
    # ist also noch unbekannt. Diese Zeilen behalten wir trotzdem: Ihre
    # Features braucht die Inferenz-Pipeline. Beim nächsten Lauf wird das
    # Label ergänzt.
    df.loc[regen_naechste_2h.isna(), "rain_next_2h"] = np.nan

    # Primary Key: die Stunde als Text, z. B. "2026-10-13 13:00:00".
    # Die Spalte time direkt als Primary Key geht nicht: Dann kann Hopsworks
    # eine schon gespeicherte Stunde beim nächsten Lauf nicht überschreiben,
    # und der Job bricht ab.
    df["hour_id"] = df["time"].astype(str)

    df = df[["hour_id", "time", "precipitation", "relative_humidity_2m",
             "cloud_cover", "humidity_mean_24h", "rain_next_2h"]]

    print("\nDie neuesten Zeilen (Zeiten in UTC):")
    print(df.tail().to_string())
    print("\nAnteil Stunden mit Regen in den nächsten 2 Stunden:",
          round(df["rain_next_2h"].mean(), 3))

    return df

def bei_hopsworks_anmelden():
    """Meldet sich bei Hopsworks an und gibt das Projekt zurück."""
    # Der API-Key und der Projektname stehen in der Datei .env
    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT"),
    )
    return project

def feature_group_erstellen(fs):
    """Schritt 1.4: Feature Group erstellen mit Primary Key + Event Time."""
    weather_fg = fs.get_or_create_feature_group(
        name="weather_basel",
        version=1,
        description="Stündliches Wetter in Basel mit Features und dem Label "
                    "rain_next_2h (Regen in den nächsten 2 Stunden)",
        primary_key=["hour_id"],   # eine Zeile pro Stunde
        event_time="time",         # der Zeitpunkt, zu dem die Zeile gehört
        # Das Standardformat DELTA braucht ein Zusatzpaket, das sich nicht auf
        # jedem Rechner installieren lässt. HUDI funktioniert überall.
        time_travel_format="HUDI",
    )
    return weather_fg

def daten_einfuegen(weather_fg, df):
    """Schritt 1.5: Dataframe in die Feature Group einfügen."""
    print("\nSchreibe", len(df), "Zeilen in die Feature Group ...")
    weather_fg.insert(df, write_options={"wait_for_job": True})
    print("Fertig.")

# ------------------------------------------------------------------
# Hauptteil: hier laufen die Schritte der Reihe nach ab
# ------------------------------------------------------------------

df = wetterdaten_holen()
df = features_berechnen(df)
df = label_berechnen(df)

project = bei_hopsworks_anmelden()
fs = project.get_feature_store()
weather_fg = feature_group_erstellen(fs)
daten_einfuegen(weather_fg, df)
