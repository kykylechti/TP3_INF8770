import cv2
import numpy as np
import matplotlib.pyplot as plt

videos = ["resultat_10s.mp4", "resultat_10s2.mp4"]

transitions = [[91, 136, 197], [67, 180]]
transitions_gradual = [[], []]

# Paramètres pour la méthode par histogrammes
NBINS_H = 16
NBINS_S = 16
NBINS_V = 16

THRESHOLD_HISTO_CUT = 0.5
WINDOW_SIZE = 10
THRESHOLD_HISTO_GRADUAL = 0.4

def compute_histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [NBINS_H], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [NBINS_S], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [NBINS_V], [0, 256])

    hist = np.concatenate([hist_h, hist_s, hist_v], axis=0)
    hist = cv2.normalize(hist, hist).flatten()
    return hist

for video_path in videos:
    cap = cv2.VideoCapture(video_path)

    prev_hist = None
    dist_list = []
    frame_counter = 0

    # Calcul des distances entre histogrammes successifs
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        hist = compute_histogram(frame)

        if prev_hist is not None:
            # distance euclidienne entre histogrammes
            dist = np.linalg.norm(hist - prev_hist)
            dist_list.append(dist)

        prev_hist = hist

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    dist_array = np.array(dist_list)
    frames = np.arange(1, len(dist_array) + 1)

    # Détection des coupures franches (cuts) avec un seuil sur la distance
    cut_detections = np.where(dist_array >= THRESHOLD_HISTO_CUT)[0] + 1

    print(f"\n--- Résultats (méthode histogrammes) pour {video_path} ---")
    print(f"Seuil utilisé pour les cuts : dist >= {THRESHOLD_HISTO_CUT}")
    print(f"Coupures détectées à la frame : {list(cut_detections)}")

    # Détection des transitions progressives
    window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
    dist_smooth = np.convolve(dist_array, window, mode='valid')
    dist_delta = np.abs(np.diff(dist_smooth))

    gradual_detections_indices = np.where(dist_delta >= THRESHOLD_HISTO_GRADUAL)[0]
    gradual_detections = gradual_detections_indices + WINDOW_SIZE

    print(f"Début de transitions progressives (Gradual) détectées à la frame : {list(gradual_detections)}")

    # Visualisation du signal de distance + détection des cuts/gradual
    plt.figure(figsize=(12, 6))
    plt.plot(frames, dist_array, label='Distance entre histogrammes', color='blue')

    # Marquer les coupes franches
    for idx, frame_index in enumerate(cut_detections):
        plt.axvline(x=frame_index, color='green', linestyle='--', linewidth=1,
                    label='Cut détectée' if idx == 0 else "")

    # Marquer les transitions progressives
    if len(gradual_detections) > 0:
        for idx, frame_index in enumerate(gradual_detections):
            plt.axvline(x=frame_index, color='orange', linestyle='-', linewidth=1,
                        label='Gradual détectée' if idx == 0 else "")

    plt.axhline(y=THRESHOLD_HISTO_CUT, color='red', linestyle=':', linewidth=0.8,
                label=f'Seuil cut ({THRESHOLD_HISTO_CUT})')

    plt.title(f"Évolution de la distance d'histogrammes pour {video_path}")
    plt.xlabel("Frame")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualisation du signal lissé et de la variation pour les graduals
    frames_smooth = np.arange(WINDOW_SIZE, len(dist_smooth) + WINDOW_SIZE)
    frames_delta = np.arange(WINDOW_SIZE, len(dist_delta) + WINDOW_SIZE)

    plt.figure(figsize=(12, 4))
    plt.plot(frames_smooth, dist_smooth, label=f'Distance lissée (Fenêtre {WINDOW_SIZE})', color='purple')
    plt.title(f"Signal de distance lissé (histogrammes) – {video_path}")
    plt.xlabel("Frame")
    plt.ylabel("Distance lissée")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(frames_delta, dist_delta, label='Delta de la distance lissée', color='brown')
    plt.axhline(y=THRESHOLD_HISTO_GRADUAL, color='red', linestyle=':', linewidth=0.8,
                label=f'Seuil gradual ({THRESHOLD_HISTO_GRADUAL})')
    for frame_index in gradual_detections:
        plt.axvline(x=frame_index, color='orange', linestyle='-', linewidth=1)
    plt.title(f"Détection de transitions progressives (histogrammes) – {video_path}")
    plt.xlabel("Frame")
    plt.ylabel("Delta")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Évaluation :

    tolerance = 5

    # Vérité terrain
    true_cuts = transitions[videos.index(video_path)]
    predicted_cuts = cut_detections

    TP = 0
    FP = 0
    FN = 0
    used_predictions = set()

    for t in true_cuts:
        matched = False
        for p in predicted_cuts:
            if abs(p - t) <= tolerance:
                TP += 1
                used_predictions.add(p)
                matched = True
                break
        if not matched:
            FN += 1

    FP = len(predicted_cuts) - len(used_predictions)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\nÉvaluation de la détection des coupes (histogrammes) :")
    print(f" - Vrais Positifs (TP) : {TP}")
    print(f" - Faux Positifs (FP) : {FP}")
    print(f" - Faux Négatifs (FN) : {FN}")
    print(f" - Précision : {precision:.3f}")
    print(f" - Rappel    : {recall:.3f}")
    print(f" - Score F1  : {f1:.3f}")

    true_gradual = transitions_gradual[videos.index(video_path)]
    predicted_gradual = gradual_detections

    TPg = 0
    FPg = 0
    FNg = 0
    used_predictions_g = set()

    for t in true_gradual:
        matched = False
        for p in predicted_gradual:
            if abs(p - t) <= tolerance:
                TPg += 1
                used_predictions_g.add(p)
                matched = True
                break
        if not matched:
            FNg += 1

    FPg = len(predicted_gradual) - len(used_predictions_g)

    precision_g = TPg / (TPg + FPg) if (TPg + FPg) > 0 else 0
    recall_g = TPg / (TPg + FNg) if (TPg + FNg) > 0 else 0
    f1_g = 2 * precision_g * recall_g / (precision_g + recall_g) if (precision_g + recall_g) > 0 else 0

    print("\nÉvaluation des transitions progressives (histogrammes) :")
    print(f" - Vrais Positifs (TP) : {TPg}")
    print(f" - Faux Positifs (FP) : {FPg}")
    print(f" - Faux Négatifs (FN) : {FNg}")
    print(f" - Précision : {precision_g:.3f}")
    print(f" - Rappel    : {recall_g:.3f}")
    print(f" - Score F1  : {f1_g:.3f}")
