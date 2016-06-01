# LSC-Internship

- GetPhones_dictio.py
    Mets en place les données pour les programmes suivants
    cf options en entête de programme
    outputs dans /data

- phoneCBOW.py
    Inputs : Vecteurs représentant les phonèmes à gauche et à droite sont concaténés
    Architecture de idem article T. Mikolov
    
    MAUVAIS SCORE
    
- PhonesSkipgram1.py
    Inputs : vecteur correspondant à un phonème
    Target : phonème suivant seulement
    
    MAUVAIS SCORE
    
- PhonesSkipgram2.py
    Inputs : vecteur correspondant à un phonème
    Target : phonème à gauche et à droite. 
    Output de taille 2*n_in. 
    Architecture idem article T. Mikolov
    calcul de deux cost functions sur chaque moitié du vecteur de sortie puis additionnées. 
    Première moitié correspond au phonème précédent, deuxième moitié au suivant. 
    
    NE FONCTIONNE PAS

remarque : Probleme de taille des "train_set", "valid_set" et "test_set". ne respecte pas les tailles imposées 
            (Ce n'est donc pas qui fait buger PS2)
