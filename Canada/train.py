"""Train linear hazard models from supplied synthetic text only.

No access to sealed_reports.npz or vault.npz. General hazard vocabulary below
is synthetic semantic augmentation, not external incident records.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, vstack

from featurize import featurize, tokens, bucket, DIM

CLASSES = ["flood", "storm", "winter", "wildfire", "drought", "ground", "industrial", "transport", "epidemic"]
TERMS = {
    "flood": "flood;floods;flooding;flooded;inundation;inundated;overflow;overflowed;overflowing;freshet;high water;river rose;river overflowed;water levels;flood waters;flash flooding;flood plain;inondation;inondations;inondé;inondés;inondée;inondées;crue;crues;débordement;débordé;eaux de crue;hautes eaux;sortir de son lit",
    "storm": "storm;storms;thunderstorm;thunderstorms;hail;hailstorm;hailstones;hurricane;hurricanes;tornado;tornadoes;cyclone;windstorm;storm surge;tropical storm;violent winds;tempête;tempêtes;orage;orages;ouragan;ouragans;tornade;tornades;grêle;grêlons;rafales;vents violents;onde de tempête;tempête tropicale",
    "winter": "snow;snowfall;snowstorm;blizzard;blizzards;ice;icy;freezing;frozen;frost;cold;sleet;winter;winter storm;freezing rain;ice storm;cold wave;neige;neiges;neigeux;neigeuse;enneigement;verglas;glace;glacial;gel;gelée;froid;froide;hiver;hivernal;hivernale;grésil;pluie verglaçante;tempête hivernale;vague de froid;poudrerie",
    "wildfire": "wildfire;wildfires;wildland;bushfire;brushfire;forest fire;forest fires;grass fire;forest burned;forest burning;hectares;forests;woodlands;forêt;forêts;forestier;forestiers;forestière;broussailles;feu de forêt;feux de forêt;incendie de forêt;incendies de forêt;hectares brûlés",
    "drought": "drought;droughts;heat;heatwave;heatwaves;heat wave;dry weather;dry spell;water shortage;lack of rain;crop failure;crop losses;record temperatures;dry conditions;water restrictions;extreme heat;sécheresse;sécheresses;canicule;chaleur;chaud;aridité;vague de chaleur;manque de pluie;pénurie d'eau;pertes de récoltes;récoltes perdues",
    "ground": "landslide;landslides;mudslide;mudslides;rockslide;rockfall;avalanche;avalanches;earthquake;earthquakes;tsunami;tsunamis;tremor;seismic;landslip;debris slide;ground movement;glissement;glissements;éboulement;éboulements;séisme;séismes;tremblement;secousse;secousses;raz-de-marée;coulée de boue;glissement de terrain;avalanche de débris",
    "industrial": "explosion;explosions;exploded;explosive;spill;spills;chemical;chemicals;toxic;factory;warehouse;refinery;industrial fire;urban fire;building fire;oil spill;gas leak;structural failure;bridge collapse;fire destroyed;fire broke;blaze;incendie;incendies;usine;entrepôt;raffinerie;explosé;explose;déversement;déversements;toxique;produits chimiques;fuite de gaz;incendie industriel;incendie urbain;rupture de barrage;effondrement du pont;feu se déclare",
    "transport": "train;trains;rail;railway;railroad;derailed;derailment;collision;collided;crash;crashed;aircraft;airplane;plane;passenger;passengers;vessel;ferry;ship;ships;boat;sank;sinking;capsized;foundered;bus;truck;road accident;rail accident;avion;avions;navire;navires;bateau;traversier;déraillement;déraille;déraillent;collision;écrase;écrasement;naufrage;naufrages;passagers;voitures;autobus;cargo;coule;sombre;transport maritime",
    "epidemic": "epidemic;epidemics;pandemic;outbreak;outbreaks;influenza;flu;polio;smallpox;cholera;typhoid;infection;infected;disease;virus;viral;cases;illness;vaccination;gastroenteritis;epidemic disease;épidémie;épidémies;pandémie;éclosion;éclosions;grippe;variole;choléra;typhoïde;infection;infectés;infectées;maladie;maladies;cas;gastroentérite;fièvre;contaminée;poumons d'acier",
}


def plain(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def core_sentences(text, label):
    # Date/location headers precede the first year. Remove them from training,
    # not from the official featurizer or sealed inference.
    text = re.sub(r"^.*?\b(?:18|19|20)\d{2}[.:,]\s*", "", text, count=1)
    anchors = [plain(s) for s in TERMS[label].split(";")]
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        value = plain(sentence)
        if any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", value) for a in anchors):
            kept.append(sentence)
    return " ".join(kept)


def train(corpus, output):
    from finalize import finalize_model
    with open(corpus, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    y = np.array([CLASSES.index(row["label"]) for row in rows])
    core = [core_sentences(row["text"], row["label"]) for row in rows]
    keep = np.array([bool(text) for text in core])
    X = csr_matrix(featurize([text for text in core if text]))
    labels = y[keep]
    print("Core synthetic rows:", len(labels), flush=True)
    seed_text, seed_y = [], []
    rng = np.random.default_rng(42)
    for label in CLASSES:
        phrases = TERMS[label].split(";")
        for phrase in phrases:
            seed_text.append(phrase)
            seed_y.append(CLASSES.index(label))
        for _ in range(200):
            seed_text.append(". ".join(rng.choice(phrases, size=3, replace=False)))
            seed_y.append(CLASSES.index(label))
    Xseed = csr_matrix(featurize(seed_text))
    A = vstack([X, Xseed])
    target = np.concatenate([labels, seed_y])
    weights = np.concatenate([np.ones(len(labels)), np.full(len(seed_y), 5.0)])
    model = LogisticRegression(C=3, max_iter=1000, class_weight="balanced", random_state=42).fit(A, target, sample_weight=weights)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = output / "base_model.npz"
    np.savez_compressed(base, W=model.coef_, b=model.intercept_)
    active = finalize_model(base, corpus, output / "model.npz")
    metrics = {"source_rows": len(rows), "core_rows": len(labels), "seed_rows": len(seed_y),
               "source_seed_training_accuracy": float(model.score(A, target)),
               "active_feature_cells": active, "random_state": 42,
               "sealed_data_used_in_training": False}
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    train(args.corpus, args.output)
