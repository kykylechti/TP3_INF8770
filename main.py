import cv2
import numpy as np
import matplotlib.pyplot as plt

videos = ["resultat_10s.mp4", "resultat_10s2.mp4"]

transitions = [[91, 136, 197], [67, 180]]

Sobelx = np.array([[-1.0, 0.0, 1.0],
                   [-2.0, 0.0, 2.0],
                   [-1.0, 0.0, 1.0]])

Sobely = np.array([[-1.0, -2.0, -1.0],
                   [ 0.0,  0.0,  0.0],
                   [ 1.0,  2.0,  1.0]])

THRESHOLD_RHO = 0.5
WINDOW_SIZE = 10
THRESHOLD_GRADUAL = 0.4

def dilate_edges(img, r):
    struct_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (2*r + 1, 2*r + 1)
    )
    return cv2.dilate(img, struct_element)

for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    
    prev_bin_edges = None 
    prev_dilated_edges = None

    rho_in_list = []
    rho_out_list = []

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        Gx = cv2.filter2D(gray, cv2.CV_64F, Sobelx)
        Gy = cv2.filter2D(gray, cv2.CV_64F, Sobely)
        magnitude = np.sqrt(Gx**2 + Gy**2)

        magnitude = (magnitude * 255 / magnitude.max()).astype(np.uint8)

        threshold = 80
        edges_bin = (magnitude > threshold).astype(np.uint8)
        
        edges_dilated = dilate_edges(edges_bin, r=10)

        if prev_dilated_edges is not None and prev_bin_edges is not None:
            intersection_in = np.sum(np.logical_and(prev_dilated_edges, edges_bin))
            sum_edges_bin = np.sum(edges_bin)
            
            rho_in = 1 - (intersection_in / sum_edges_bin if sum_edges_bin > 0 else 0)
            rho_in_list.append(rho_in)

            intersection_out = np.sum(np.logical_and(prev_bin_edges, edges_dilated))
            sum_prev_bin_edges = np.sum(prev_bin_edges)
            
            rho_out = 1 - (intersection_out / sum_prev_bin_edges if sum_prev_bin_edges > 0 else 0)
            rho_out_list.append(rho_out)

        prev_dilated_edges = edges_dilated.copy() 
        prev_bin_edges = edges_bin.copy()  


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    rho_in_array = np.array(rho_in_list)
    rho_out_array = np.array(rho_out_list)
    frames = np.arange(1, len(rho_in_array) + 1)

    rho_array = np.maximum(rho_in_array, rho_out_array)

    cut_detections = np.where(
        (rho_array >= THRESHOLD_RHO)
    )[0] + 1

    print(f"\n--- Résultats pour {video_path} ---")
    print(f"Seuils utilisés : rho >= {THRESHOLD_RHO}")
    print(f"Coupures détectées à la frame : {list(cut_detections)}")

    window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
    rho_smooth = np.convolve(rho_array, window, mode='valid')
    rho_delta = np.abs(np.diff(rho_smooth))
    gradual_detections_indices = np.where(rho_delta >= THRESHOLD_GRADUAL)[0]
    gradual_detections = gradual_detections_indices + WINDOW_SIZE

    print(f"Début de Transitions Progressives (Gradual) détectées à la frame : {list(gradual_detections)}")

    plt.figure(figsize=(12, 6))
    plt.plot(frames, rho_in_array, label='rho_in', color='blue')
    plt.plot(frames, rho_out_array, label='rho_out', color='red')
    
    # Marquer les coupes franches
    for frame_index in cut_detections:
        plt.axvline(x=frame_index, color='green', linestyle='--', linewidth=1, label='Cut Détectée' if frame_index == cut_detections[0] else "")
    
    # Marquer les transitions progressives
    if len(gradual_detections) > 0:
        for frame_index in gradual_detections:
            plt.axvline(x=frame_index, color='orange', linestyle='-', linewidth=1, label='Gradual Détectée' if frame_index == gradual_detections[0] else "")
    
    plt.axhline(y=THRESHOLD_RHO, color='blue', linestyle=':', linewidth=0.8, label=f'Seuil rho_cut ({THRESHOLD_RHO})')
    
    plt.title(f"Évolution de rho_in et rho_out pour {video_path} avec Détection")
    plt.xlabel("Frame")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Affichage séparé pour le signal lissé et le delta pour mieux comprendre la détection progressive
    frames_smooth = np.arange(WINDOW_SIZE, len(rho_smooth) + WINDOW_SIZE)
    frames_delta = np.arange(WINDOW_SIZE, len(rho_delta) + WINDOW_SIZE)
    
    plt.figure(figsize=(12, 4))
    plt.plot(frames_smooth, rho_smooth, label=f'rho Lissé (Fenêtre {WINDOW_SIZE})', color='purple')
    plt.title(f"Signal rho lissé pour Détection Progressive ({video_path})")
    plt.xlabel("Frame")
    plt.ylabel("rho Lissé")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(frames_delta, rho_delta, label=f'Delta du rho Lissé', color='brown')
    plt.axhline(y=THRESHOLD_GRADUAL, color='red', linestyle=':', linewidth=0.8, label=f'Seuil Gradual ({THRESHOLD_GRADUAL})')
    for frame_index in gradual_detections:
        plt.axvline(x=frame_index, color='orange', linestyle='-', linewidth=1)
    plt.title(f"Détection de Changement dans le Signal Lissé ({video_path})")
    plt.xlabel("Frame")
    plt.ylabel("Delta")
    plt.legend()
    plt.grid(True)
    plt.show()