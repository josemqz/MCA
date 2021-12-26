
# debería usar semáforos?
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

accFreno = -5

# NOTA - - - - 
# radio nacional: 232326169 (promocion de mente vital: $23k)
#  - - - - - -

# inicializar mapa con bicicletas 
# flujo de bicicletas entrantes
# en cada tick avanzan con su aceleración propia hasta una velocidad propia
# si detectan una bicicleta a una cierta distancia adelante, al pasar un tiempo tReaccion, comienza el freno 
# (que corresponde a modificar la aceleración a negativa)

# debo partir con un ejemplo chico

# aparece una bicicleta en el inicio con velocidad 0

class bicicleta:
    def _init_(self, vMax, tReaccion, initAcc):
        self.loc = [0,0]
        self.vAct = int(vMax/2) #probando
        self.vMax = vMax
        self.tReaccion = tReaccion
        self.initAcc = initAcc
        self.actAcc = initAcc
        
        # self.freno = False  # no es necesario, porque es True cuando actAcc == accFreno
        self.tFrenado = 0 # tiempo acumulado de frenado

    def activarFreno(self, actLoc, pista):
        self.aceleracion = accFreno


# necesita ser mapa??
class pista:
    def _init_(self, largo, ancho):
        self.map = np.zeros((ancho, largo))


# def initBici():


def main():

    flujo_in = 2 # 

    # inicializar pista
    largoP = 2000
    P = pista(largoP, 2) # va a depender de la granularidad y distancia

    t = 0
    T = 1000

    # Pob = bicicleta[int(T/flujo_in)] # cota superior, en caso de crear a todos y que no desapareciera ninguno en la meta
    Pob = []

    while t < T:


        if t % flujo_in == 0:
            vInit = 20
            # tReaccion depende de la velocidad máxima y tiempo de frenado
            tReaccion = 2
            # que en 4 pasos alcance la velocidad máxima
            accInit = vInit/4
            Pob.append(bicicleta(vInit, tReaccion, accInit))

            P.map[Pob.loc[0], Pob.loc[1]] = 1

        pobAccidentada = []
        b_index = 0
        for _ in Pob:

            # if bicicleta_cercana:
                # guardar index de la bici (y si hay más de una??)

            # si distancia a la bicicleta cercana es 0, se produce un accidente
            # se agrega la bicicleta y la cercana a la población accidentada, y fuera del loop se eliminan de Pob

            # pobAccidentada.append(b_index)
            # pobAccidentada.append()

            # verificar si hay una bicicleta cercana adelante para actualizar aceleración (si es que no ha cambiado ya)
            # aceleración de frenado variable??

            # si pasa un tiempo con aceleracion 0 y hay una bici a una distancia un poquito mayor a una cercana
            # tiene que pasar a la bici, cambiando de pista

            # actualizar velocidad con aceleración actual (independiente de la situación)
            # else (del primer if)
            prev_loc = Pob[b_index].loc
            if Pob[b_index].vAct < Pob[b_index].vMax:
                Pob[b_index].vAct += Pob[b_index].actAcc

            Pob[b_index].loc[1] += Pob[b_index].vAct
            # actualizar mapa
            P.map[Pob[b_index].loc[0], Pob[b_index].loc[1]] = 1
            P.map[prev_loc[0], prev_loc[1]] = 0

            b_index += 1


        Y, X = np.where(P.map)

        fig, ax = plt.subplots(figsize=(12,12))
        ax.plot(X,Y, ".")

        plt.show()
        plt.pause(0.01)
        clear_output(wait=True)

        t += 1
    