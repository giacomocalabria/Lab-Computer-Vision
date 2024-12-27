import cv2
import numpy as np

def separate_image_by_brightness(image):
    """
    Divide un'immagine in base alla luminosità media lungo l'asse verticale.

    Args:
        image (numpy.ndarray): Immagine in input (a colori o in scala di grigi).

    Returns:
        tuple: (zona_sopra, zona_sotto, linea_separazione)
               dove "zona_sopra" e "zona_sotto" sono immagini ritagliate e
               "linea_separazione" è la coordinata y della linea di separazione.
    """
    # Converti l'immagine in scala di grigi se non lo è già
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applica un filtro di mediana per ridurre il rumore
    gray = cv2.GaussianBlur(gray, (9, 9), 1)
    

    # Ottieni le dimensioni dell'immagine
    height, width = gray.shape

    # Calcola la luminosità media per ogni riga
    row_brightness = np.mean(gray, axis=1)

    # Calcola la somma cumulativa della luminosità
    cumulative_brightness = np.cumsum(row_brightness)

    # Normalizza la somma cumulativa
    cumulative_brightness_normalized = cumulative_brightness / cumulative_brightness[-1]

    # Trova la linea di separazione basata su un punto medio (e.g., 50% cumulativo)
    separation_line = np.searchsorted(cumulative_brightness_normalized, 0.5)

    # Divide l'immagine in base alla riga di separazione
    zona_sopra = image[:separation_line, :]
    zona_sotto = image[separation_line:, :]

    return zona_sopra, zona_sotto, separation_line

# Esempio d'uso
if __name__ == "__main__":
    # Carica un'immagine di esempio
    image = cv2.imread('Final Project/dataset/data/hard/rs3h.jpg')

    # Separazione basata sulla luminosità
    zona_sopra, zona_sotto, linea_separazione = separate_image_by_brightness(image)

    # Disegna la linea di separazione sull'immagine originale (per visualizzare)
    output = image.copy()
    cv2.line(output, (0, linea_separazione), (output.shape[1], linea_separazione), (0, 0, 255), 2)

    # Mostra i risultati
    cv2.imshow('Immagine Originale con Linea di Separazione', output)
    cv2.imshow('Zona Sopra', zona_sopra)
    cv2.imshow('Zona Sotto', zona_sotto)
    cv2.waitKey(0)
    cv2.destroyAllWindows()