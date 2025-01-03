import cv2
import numpy as np
import os

# 1. Carica l'immagine
image = cv2.imread('.\Final Project\dataset\data\city_blue\pieces1.JPG')

# Verifica che l'immagine sia stata caricata correttamente
if image is None:
    raise FileNotFoundError("L'immagine non è stata trovata. Controlla il percorso specificato!")

# 2. Applica il Mean Shift per la segmentazione
# Parametri principali: sp (raggio spaziale) e sr (raggio del colore)
mean_shift_image = cv2.pyrMeanShiftFiltering(image, sp=10,sr=10)

#mean_shift_image = np.copy(image)

# 3. Converti in scala di grigi
gray = cv2.cvtColor(mean_shift_image, cv2.COLOR_BGR2GRAY)

# 4. Applica una soglia binaria inversa per segmentare
_, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

# 5. Operazioni morfologiche per eliminare dettagli interni e raffinare la segmentazione
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# 6. Trova i contorni
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 7. Filtra i contorni per area (opzionale)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

# 8. Crea una cartella per salvare i pezzi segmentati
output_folder = "puzzle_pieces"
os.makedirs(output_folder, exist_ok=True)

# 9. Segmenta e salva i pezzi
for i, contour in enumerate(filtered_contours):
    # Crea un rettangolo di bounding per ogni pezzo
    x, y, w, h = cv2.boundingRect(contour)
    
    # Estrai il pezzo
    piece = image[y:y+h, x:x+w]
    
    # Salva il pezzo nella cartella
    piece_filename = os.path.join(output_folder, f'piece_{i+1}.png')
    cv2.imwrite(piece_filename, piece)

    # Disegna il rettangolo sull'immagine originale (per visualizzazione)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, f'Piece {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# 10. Mostra i risultati
cv2.namedWindow('Segmented Puzzle Pieces', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('Binary Threshold', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('Mean Shift Segmentation', cv2.WINDOW_GUI_NORMAL)
cv2.imshow('Segmented Puzzle Pieces', image)
cv2.imshow('Binary Threshold', thresh)
cv2.imshow('Mean Shift Segmentation', mean_shift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Tutti i pezzi sono stati salvati nella cartella: {output_folder}")
