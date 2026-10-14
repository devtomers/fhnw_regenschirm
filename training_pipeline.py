# Training Pipeline
# =================
# Liest die Features aus dem Featurestore, trainiert ein Modell, das
# vorhersagt, ob es in den nächsten 2 Stunden regnet, und lädt das Modell in
# die Hopsworks Model Registry.
#
# Ausführen:  python training_pipeline.py
# (vorher muss feature_pipeline.py mindestens einmal gelaufen sein)
#
# Bei jedem Lauf entsteht eine neue Modellversion in der Model Registry.
#
# Oben stehen die Funktionen, ganz unten der Hauptteil, der sie der Reihe
# nach aufruft.

import os

import hopsworks
import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def bei_hopsworks_anmelden():
    """Meldet sich bei Hopsworks an und gibt das Projekt zurück."""
    # Der API-Key und der Projektname stehen in der Datei .env
    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT"),
    )
    return project


def feature_group_laden(fs):
    """Schritt 2.1: Feature Group laden."""
    return fs.get_feature_group(name="weather_basel", version=1)


def feature_view_erstellen(fs, weather_fg):
    """Schritt 2.2: Feature View erstellen mit einer Query, die Features +
    Label auswählt."""
    # Die zwei Features, das Label und die Zeit. Die Zeit braucht die
    # Inferenz-Pipeline, um die neueste Stunde zu finden. Das Modell bekommt
    # sie nicht (siehe trainingsdaten_holen). Die Rohwerte (precipitation,
    # relative_humidity_2m) bleiben in der Feature Group.
    query = weather_fg.select(["time", "cloud_cover", "humidity_mean_24h", "rain_next_2h"])

    feature_view = fs.get_or_create_feature_view(
        name="weather_basel_view",
        version=1,
        query=query,
        labels=["rain_next_2h"],
        description="Bewölkung und mittlere Luftfeuchtigkeit (24 h), Label: "
                    "Regen in den nächsten 2 Stunden",
    )
    return feature_view


def trainingsdaten_holen(feature_view):
    """Schritt 2.3: Training Dataset aus der Feature View erstellen."""
    X, y = feature_view.training_data(description="Wetter Basel, alle Stunden")

    # Die neuesten 2 Stunden haben noch kein Label (die Zukunft ist noch
    # nicht bekannt). Mit diesen Zeilen kann man nicht trainieren, also
    # weglassen.
    # Man kanne es nicht so machen:
    #   y = y.dropna()      löscht Zeilen 3 und 4 aus y
    #   X = X.dropna()      löscht in X NICHTS, denn X hat ja keine NaN!
    # Weil sonst wäre die beiden DFs nicht gleich gross und es gäbe ein Fehler beim zusammenfügen.
    daten = X.copy()
    daten["rain_next_2h"] = y["rain_next_2h"]
    daten = daten.dropna()

    # Hopsworks liefert die Zeilen nicht immer in derselben Reihenfolge. Nach
    # der Zeit sortieren, damit random_state=42 bei gleichen Daten auch
    # wirklich jedes Mal dieselbe Aufteilung ergibt.
    daten = daten.sort_values("time")

    # Nur die zwei Features gehen ins Modell, die Zeit nicht
    X = daten[["cloud_cover", "humidity_mean_24h"]]
    y = daten["rain_next_2h"]
    print("Stunden für das Training:", len(X))

    # 80 % zum Trainieren, 20 % zum Testen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # 42: Weil es die Antwort auf alle Fragen ist
    )
    return X_train, X_test, y_train, y_test


def modell_trainieren(X_train, y_train):
    """Schritt 2.4, erster Teil: Modell trainieren."""
    # Regen ist selten (nur etwa jede 14. Stunde). 
    # Deshalb class_weight="balanced", damit das Modell die Regenstunden
    # mehr zählt. Aber das hat dann Einfluss auf die Accuracy

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def modell_bewerten(model, X_test, y_test):
    """Schritt 2.4, zweiter Teil: Modell auf den Testdaten evaluieren."""
    vorhersage = model.predict(X_test)
    accuracy = accuracy_score(y_test, vorhersage)
    erkannter_regen = recall_score(y_test, vorhersage)
    # zero_division=0: falls das Modell nie "Regen" sagt, gibt es einfach 0,
    # ohne Warnung
    richtiger_schirm = precision_score(y_test, vorhersage, zero_division=0)

    # Zum Vergleich: Wie gut wäre es, einfach immer "kein Regen" zu sagen?
    immer_kein_regen = 1 - y_test.mean()

    print("\nErgebnis auf den Testdaten:")
    print("  Accuracy des Modells:             ", round(accuracy, 3))
    print("  Accuracy bei immer 'kein Regen':  ", round(immer_kein_regen, 3))
    print("  Anteil erkannter Regenstunden:    ", round(erkannter_regen, 3))
    print("  Anteil 'Schirm einpacken' richtig:", round(richtiger_schirm, 3))

    return {"accuracy": accuracy, "recall": erkannter_regen, "precision": richtiger_schirm} # Für das Model Registry zusammegepackt in ein Dict


def modell_speichern(project, model, messwerte, feature_view):
    """Schritt 2.5: Modell lokal abspeichern, dann in die Model Registry
    hochladen."""
    os.makedirs("model", exist_ok=True)     # Ordner model anlegen, falls er noch nicht existiert
    joblib.dump(model, "model/model.pkl")

    mr = project.get_model_registry()
    rain_model = mr.python.create_model(
        name="rain_model",
        metrics=messwerte,
        description="Logistische Regression: Regen in Basel in den nächsten 2 Stunden",
        feature_view=feature_view,
    )
    rain_model.save("model")
    print("\nModell gespeichert als rain_model, Version", rain_model.version)


# ------------------------------------------------------------------
# Hauptteil: hier laufen die Schritte der Reihe nach ab
# ------------------------------------------------------------------

project = bei_hopsworks_anmelden()
fs = project.get_feature_store()

weather_fg = feature_group_laden(fs)
feature_view = feature_view_erstellen(fs, weather_fg)
X_train, X_test, y_train, y_test = trainingsdaten_holen(feature_view)

model = modell_trainieren(X_train, y_train)
messwerte = modell_bewerten(model, X_test, y_test)
modell_speichern(project, model, messwerte, feature_view)
