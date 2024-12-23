import cv2 as cv                                       
import numpy as np

# 1. Carica l'immagine
image = cv.imread('.\dataset\data\peppa\pz.png')

# 2. Converti in scala di grigi
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)



# 3. Applica una soglia
# Usa una soglia adattiva o una binarizzazione classica
_, thresh =cv.threshold(gray, 0, 255,cv.THRESH_BINARY_INV +cv.THRESH_OTSU)

# 4. Trova i contorni
contours, _ =cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

# 5. Disegna i contorni e visualizza tutti i pezzi
output_image = image.copy()  # Copia l'immagine originale per disegnare i contorni
for i, contour in enumerate(contours):
    # Crea un rettangolo di bounding per ogni pezzo
    x, y, w, h = cv.boundingRect(contour)
    
    # Disegna il rettangolo sull'immagine
    cv.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Etichetta i pezzi (opzionale)
    cv.putText(output_image, f'Piece {i+1}', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# 6. Mostra l'immagine con i contorni e le etichette
cv.imshow('Segmented Puzzle Pieces', output_image)

# 7. Mostra la maschera binaria (opzionale)
cv.imshow('Binary Threshold', thresh)

cv.waitKey(0)
cv.destroyAllWindows()
