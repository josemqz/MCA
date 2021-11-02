import numpy as np
import time
import sys
import signal
from matplotlib import pyplot as plt
import scipy.linalg as spla
from colorama import init, Back, Fore, Style
# import os

# colorama init
init(convert=True)


# imprimir iteraciones y mapa
debug = False
debugMap = True

# dimensiones de mapa
mapDim = 50
mapDimY = mapDim # se puede cambiar para que sea rectangular
# variación de tiempo por iteración
tick = 1

# cantidad máxima de iteraciones
maxT = 300

# cantidad de pasto
probPasto = 0.7
# poblaciones (máximo c/u: 2^15-1 (por uso de int16))
pobZorros = 120
pobConejos = 380
# probabilidad de reproducción
probRepZorros = 0.65
probRepConejos = 0.65
# probabilidad de caza
probCaza = 0.65
# cuánta hambre reduce la caza
carneConejo = 4
# hambre necesaria para empezar a cazar
hambreCazar = 10
# muerte por hambre
hambreLimite = 12
# edad mínima para reproducirse
minEdadRepZorros = 40
minEdadRepConejos = 20
# tiempo mínimo para reproducirse
minTSinRepZorros = 25
minTSinRepConejos = 18
# edad de muerte natural
edadLimiteZorros = 200
edadLimiteConejos = 200

# ocupación máxima del mapa
maxOcupacion = 0.85

class Animal:

    def __init__(self):
        self.tipo = 0 # zorro = 0 | conejo = 1
        self.hambre = 0
        self.posicion = [0,0]
        self.sexo = np.random.randint(0, 2) # macho = 0 | hembra = 1
        self.edad = 0
        self.TSinRep = 0 # tiempo sin reproducirse

    # mov es una int list con coordenadas x,y
    def movimiento(self, mov, xMax, yMax):
        # verifica si llega al límite del mapa, para volver al inicio (como pacman)
        self.posicion[0], self.posicion[1] = vecinoBorde(self.posicion, mov, xMax, yMax)


# es necesario para mantener registro de qué posiciones están siendo ocupadas de manera eficiente
# matriz 2d con arreglos de uint16 (2 byte c/u) [pasto, ocupado, animal, identificador (index)]
class Mapa():
    
    vecindario = [[-1, 1], [0, 1], [1, 1],
                  [-1, 0],         [1, 0], 
                  [-1,-1], [0,-1], [1,-1] ]
    
    # xMax, yMax: límites del mapa
    def __init__(self, xMax, yMax):
        self.xMax = xMax
        self.yMax = yMax

        # dimensiones mínimas de mapa
        if xMax <= 2:
            self.xMax = 3
        if yMax <= 2:
            self.yMax = 3
    
        # np.array de 2 dimensiones de arreglos de int16 ([pasto, ocupado, animal, id])
        self.coord = np.zeros((self.xMax, self.yMax, 4), np.int16)

        #Pasto :D
        i = 0
        for col in self.coord:
            j = 0
            for _ in col:
                self.coord[i][j][0] = np.random.choice([0,1], p=[1 - probPasto, probPasto])
                j += 1
            i += 1


    # pos: int list con coordenadas x,y
    def disponible(self, pos):
        return self.coord[pos[0]][pos[1]][1] == 0


# inicializa posición inicial
def posInicial(mapa):

    initPos = [np.random.randint(0, mapa.xMax), np.random.randint(0, mapa.yMax)]
    
    # busca hasta encontrar una coordenada desocupada
    while not mapa.disponible(initPos):
        initPos = [np.random.randint(0, mapa.xMax), np.random.randint(0, mapa.yMax)]

    return initPos


# inicializar animal, modificando arreglo de su raza y el mapa
# tipoAnimal: zorro: 0 | conejo = 1
def inicializarAnimal(Animales, tipoAnimal, mapa, index):

    Animales.append(Animal())
    Animales[index].tipo = tipoAnimal

    Animales[index].posicion = posInicial(mapa) #cuando se llega a una posición inicial disponible
    mapa.coord[Animales[index].posicion[0]] [Animales[index].posicion[1]] [1] = 1
    mapa.coord[Animales[index].posicion[0]] [Animales[index].posicion[1]] [2] = tipoAnimal
    mapa.coord[Animales[index].posicion[0]] [Animales[index].posicion[1]] [3] = index

    return Animales, mapa


# a partir de un movimiento se retorna la posición correcta
# útil cuando posición actual está en un borde
def vecinoBorde(posActual, vecino, xMax, yMax):
    
    # coordenadas finales
    Xmov = posActual[0] + vecino[0]
    Ymov = posActual[1] + vecino[1]

    # en X
    if Xmov < xMax and Xmov >= 0:
        check_X = Xmov
    elif Xmov < 0:
        check_X = xMax - 1
    else:
        check_X = 0

    # en Y
    if Ymov < yMax and Ymov >= 0:
        check_Y = Ymov
    elif Ymov < 0:
        check_Y = yMax - 1
    else:
        check_Y = 0

    return check_X, check_Y


# define movimiento (random por el momento)
def defMov(mapa, Animal):

    movFactibles = []

    for v in mapa.vecindario:

        check_X, check_Y = vecinoBorde(Animal.posicion, v, mapa.xMax, mapa.yMax)

        # verifica las coordenadas vecinas, y agrega las disponibles como factibles
        if mapa.disponible([check_X, check_Y]):
            movFactibles.append(v)
    
    if Animal.hambre < hambreCazar:
        # posibilidad que se queden quietos
        movFactibles.append([0,0])

    n = len(movFactibles)
    if n <= 1:
        return [0,0]

    # disminuir probabilidad de quedarse quietos
    prob = 1/n
    probList = [prob + prob/(2*(n - 1)) for _ in range(n)]
    probList[-1] = prob/2
    
    mov_index = np.random.choice(range(n), p = probList)

    return movFactibles[mov_index]


# actualizar mapa, liberando una coordenada
# pos: lista con coordenada [x, y]
def liberarCoord(mapa, pos):

    mapa.coord[pos[0]] [pos[1]] [1] =  0
    mapa.coord[pos[0]] [pos[1]] [2] = -1
    mapa.coord[pos[0]] [pos[1]] [3] = 2**15 - 2 # máximo representable por int16

    return mapa


# elimina un elemento de un arreglo de manera eficiente
# además elimina los datos del elemento del mapa
def swapop(Animales, index, mapa):
   
    coordMapaDead = Animales[index].posicion
    coordMapaLast = Animales[-1].posicion

    # si el elemento a ser eliminado no es el último
    if index < (len(Animales) - 1) and index != -1 and index >= 0:

        # guardar en aux datos del animal muerto
        aux = Animales[index]
        Animales[index] = Animales[-1]
        Animales[-1] = aux
    
    elif index < 0:
        print("\n[swapop]: error: index menor a 0")

    # else:
    #     print("\n[swapop]: no se hizo swap, el animal a eliminar probablemente era el último en su lista")
    
    # entregarle posición correcta al animal desplazado
    Animales[index].posicion = coordMapaLast
    mapa.coord[coordMapaLast[0]] [coordMapaLast[1]] [3] = index
    
    # eliminar datos del mapa
    mapa = liberarCoord(mapa, coordMapaDead)
    # eliminar último elemento de arreglo
    Animales.pop()


    return Animales, mapa


#change: coordenada de movimiento actualizado
def verMapa(mapa, change = [-10,-10]):

    i = 0
    for m in mapa.coord:
        # print(i, ":", end=" ")
        print(f"{i:02d}:", end=" ")
        j = 0
        for n in m:
            
            if n[1] == 0 and n[0] == 1:
                data = Fore.GREEN + '^' + Style.RESET_ALL
            elif n[1] == 1 and n[2] == 0:
                data = Fore.RED + "z" + Style.RESET_ALL
            elif n[1] == 1 and n[2] == 1:
                data = "c"
            else:
                data = " "

            if [i,j] == change:
                print(Back.GREEN + data + Style.RESET_ALL, end = "")
            else:
                print(data, end = "")

            j += 1
        print("")
        i += 1
    print("")


def graficarPobs(histZorros, histConejos):

    tick_count = len(histZorros) - 1
    x = np.linspace(0, tick_count, tick_count + 1)
    
    plt.figure()
    plt.plot(x, histZorros, 'r.', label='Zorros(t)')
    plt.plot(x, histConejos, 'b.', label='Conejos(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('Población')
    plt.grid(True)
    plt.show()


def modelEnd(histZorros, histConejos):
    graficarPobs(histZorros, histConejos)
    sys.exit(0)


def main():

    # manejar ctrl + C
    def signal_handler(sig, frame):
        modelEnd(histZorros, histConejos)

    signal.signal(signal.SIGINT, signal_handler)


# - INICIALIZACIÓN -

    mapa = Mapa(mapDim, mapDimY)

    #arreglos de animales, listas enlazadas de animales, diccionarios de animales?
    Zorros = []
    Conejos = []

    i = 0
    while i < pobZorros:
        Zorros, mapa = inicializarAnimal(Zorros, 0, mapa, i)
        Zorros[i].hambre = hambreCazar / 4
        Zorros[i].edad = minEdadRepZorros
        Zorros[i].TSinRep = minTSinRepZorros / 2
        i += 1

    i = 0
    while i < pobConejos:
        Conejos, mapa = inicializarAnimal(Conejos, 1, mapa, i)
        # Conejos[i].hambre = hambreCazar
        Conejos[i].edad = minEdadRepConejos
        Conejos[i].TSinRep = minTSinRepConejos - 2
        i += 1


# - ITERACIONES -

    # poblaciones en cada iteracióm
    pobZorrosAct = pobZorros
    pobConejosAct = pobConejos
    ocupacionActual = (pobZorros + pobConejos) / (mapDim * mapDimY)
    casaSola = True # se permite la reproducción
    # registros de población (para graficar)
    tick_count = 0

    histZorros = []
    histConejos = []
    
    while(tick_count < maxT and pobZorrosAct < 1500 and pobZorrosAct < 1500 and pobZorrosAct > 5 and pobConejosAct > 5 ):

        # os.system('cls' if os.name == 'nt' else 'clear')
        if debugMap:
            print("\nMAPA", tick_count)
            verMapa(mapa)
        if debug:
            print("Zorros:", pobZorrosAct, end="\t|\t")
            print("Conejos:", pobConejosAct)

        z_index = 0
        zorrosNuevos = 0
        zorrosMuertos = []  #lista de índices
        conejosCazados = [] #lista de índices
        for _ in Zorros:

            # muerte por hambre o edad
            if Zorros[z_index].hambre >= hambreLimite or Zorros[z_index].edad >= edadLimiteZorros:

                zorrosMuertos.append(z_index)
                z_index += 1
                continue
            

            # cazar
            for v in mapa.vecindario:
                if Zorros[z_index].hambre >= hambreCazar:
                    
                    check_X, check_Y = vecinoBorde(Zorros[z_index].posicion, v, mapa.xMax, mapa.yMax)
                    
                    # si es conejo
                    if not mapa.disponible([check_X, check_Y]) and mapa.coord[check_X][check_Y][2] == 1:

                        # matar conejo
                        if (np.random.random() < probCaza):

                            idConejoKill = mapa.coord[check_X][check_Y][3]

                            conejosCazados.append(idConejoKill)

                            # liberar coordenada de presencia animal
                            mapa = liberarCoord(mapa, [check_X, check_Y])

                            Zorros[z_index].hambre -= carneConejo
                else:
                    break


            # reproducción
            if casaSola:
                for v in mapa.vecindario:
                    if Zorros[z_index].TSinRep >= minTSinRepZorros and Zorros[z_index].edad >= minEdadRepZorros:

                        check_X, check_Y = vecinoBorde(Zorros[z_index].posicion, v, mapa.xMax, mapa.yMax)

                        infoPareja = mapa.coord[check_X][check_Y]

                        # si es zorro y es del sexo opuesto
                        if not mapa.disponible([check_X, check_Y])\
                        and infoPareja[2] == 0:
                        # and Zorros[z_index].sexo != Zorros[infoPareja[3]].sexo:

                            if (np.random.random() < probRepZorros):

                                # cantidad aleatoria de hijos
                                zorrosNuevos += np.random.randint(1,3)

                                Zorros[z_index].TSinRep = 0
                                Zorros[infoPareja[3]].TSinRep = 0
                    else:
                        break
            

            # movimiento
            mov = defMov(mapa, Zorros[z_index])
            
            if mov != [0, 0]:
                
                # liberar posición inicial
                mapa = liberarCoord(mapa, Zorros[z_index].posicion)
                
                Zorros[z_index].movimiento(mov, mapa.xMax, mapa.yMax)
                
                # actualizar mapa con nueva posición
                mapa.coord[Zorros[z_index].posicion[0]] [Zorros[z_index].posicion[1]] [1] = 1
                mapa.coord[Zorros[z_index].posicion[0]] [Zorros[z_index].posicion[1]] [2] = 0
                mapa.coord[Zorros[z_index].posicion[0]] [Zorros[z_index].posicion[1]] [3] = z_index
            

            Zorros[z_index].hambre += 1
            Zorros[z_index].TSinRep += 1
            Zorros[z_index].edad += 1
            z_index += 1

        # eliminar zorros muertos de Zorros
        if debug:
            if len(zorrosMuertos) > 0:
                print("zorros muertos:", len(zorrosMuertos))
            else:
                print("")
        
        for zm in reversed(zorrosMuertos):
            Zorros, mapa = swapop(Zorros, zm, mapa)
            pobZorrosAct -= 1
        
        # agregar zorrosNuevos zorros nuevos a Zorros
        if debug:
            if zorrosNuevos > 0:
                print("zorros nuevos:", zorrosNuevos)
            else:
                print("")
        
        for zn in range(zorrosNuevos):
            inicializarAnimal(Zorros, 0, mapa, pobZorrosAct + zn)

        pobZorrosAct += zorrosNuevos

        # eliminar conejos cazados
        if debug:
            if len(conejosCazados) > 0:
                print("conejos cazados:", len(conejosCazados))
            else:
                print("")
        
        for cc in reversed(sorted(conejosCazados)):
            Conejos, mapa = swapop(Conejos, cc, mapa)
            pobConejosAct -= 1
            
        if debug:
            print("- - - - - - - - - - - -")


        c_index = 0
        conejosNuevos = 0
        conejosMuertos = []
        for _ in Conejos:

            # muerte por hambre o edad
            if Conejos[c_index].hambre >= hambreLimite or Conejos[c_index].edad >= edadLimiteConejos:
                
                conejosMuertos.append(c_index)
                c_index += 1
                continue

            
            # comer                             // si la coordenada tiene pasto
            if Conejos[c_index].hambre >= 2 and mapa.coord[Conejos[c_index].posicion[0]] [Conejos[c_index].posicion[1]][0] == 1:
                Conejos[c_index].hambre -= 4
                #pasto crece?


            # reproducción
            if casaSola:
                for v in mapa.vecindario:
                    
                    if Conejos[c_index].TSinRep >= minTSinRepConejos and Conejos[c_index].edad >= minEdadRepConejos:

                        check_X, check_Y = vecinoBorde(Conejos[c_index].posicion, v, mapa.xMax, mapa.yMax)

                        infoPareja = mapa.coord[check_X][check_Y]

                        # si es conejo y es del sexo opuesto
                        if not mapa.disponible([check_X, check_Y]) \
                        and infoPareja[2] == 1:
                        # and Conejos[c_index].sexo != Conejos[infoPareja[3]].sexo:

                            if (np.random.random() < probRepConejos):

                                #generar nuevo conejo (no puedo agregarlo en esta iteración)
                                conejosNuevos += np.random.randint(1,4)
                                Conejos[c_index].TSinRep = 0
                                Conejos[infoPareja[3]].TSinRep = 0
                    
                    else:
                        break


            # movimiento
            mov = defMov(mapa, Conejos[c_index])
            if mov != [0, 0]:
                
                # liberar posición inicial
                mapa = liberarCoord(mapa, Conejos[c_index].posicion)
                
                #quizás los conejos deberían buscar alejarse de los zorros
                Conejos[c_index].movimiento(mov, mapa.xMax, mapa.yMax)
                
                # actualizar mapa con nueva posición
                mapa.coord[Conejos[c_index].posicion[0]] [Conejos[c_index].posicion[1]] [1] = 1
                mapa.coord[Conejos[c_index].posicion[0]] [Conejos[c_index].posicion[1]] [2] = 1
                mapa.coord[Conejos[c_index].posicion[0]] [Conejos[c_index].posicion[1]] [3] = c_index


            Conejos[c_index].hambre += 1
            Conejos[c_index].TSinRep += 1
            Conejos[c_index].edad += 1
            c_index += 1
        
        # eliminar conejos muertos de Conejos
        if debug:
            if len(conejosMuertos) > 0:
                print("conejos muertos: ", len(conejosMuertos))
            else:
                print("")
        
        for cm in reversed(conejosMuertos):        
            Conejos, mapa = swapop(Conejos, cm, mapa)
            pobConejosAct -= 1
        
        # agregar conejosNuevos conejos nuevos a Conejos
        if debug:
            if conejosNuevos > 0:
                print("conejos nuevos", conejosNuevos)
            else:
                print("")
            
            print("\n________________________________________________________________________________________\n\n\n")

        
        for cn in range(conejosNuevos):
            inicializarAnimal(Conejos, 1, mapa, pobConejosAct + cn)

        pobConejosAct += conejosNuevos

        # limitar reproducción
        ocupacionActual = (pobZorros + pobConejos) / (mapDim * mapDimY)
        if ocupacionActual >= maxOcupacion:
            casaSola = False
        else:
            casaSola = True


        histZorros.append(pobZorrosAct)
        histConejos.append(pobConejosAct)

        time.sleep(tick)
        tick_count += 1


    # data = np.

    # create_model(, type_model='exponential'):
    modelEnd(histZorros, histConejos)




# # # # # # #

def QR(A, type_factorization = 'reduced', type_gram_schmidt='classic'):
    A.astype('float')
    m,n = A.shape # m: number of rows, n: number of columns.
    if type_factorization == 'reduced':
        Q = np.zeros((m,n))
        R = np.zeros((n,n))
    elif type_factorization == 'full':
        Q = np.zeros((m,m))
        R = np.zeros((m,n))
    for k in range(n):
        y = A[:,k]
        for i in range(k):
            if type_gram_schmidt == 'classic':
                R[i,k] = np.dot(Q[:,i],A[:,k])
            elif type_gram_schmidt == 'modified':
                R[i,k] = np.dot(Q[:,i],y)
            y=y-R[i,k]*Q[:,i]
        R[k,k] = np.linalg.norm(y)
        Q[:,k] = y/np.linalg.norm(R[k,k])
    return Q,R
    
def least_squares(A,b):
    Q,R = QR(A,type_gram_schmidt='modified')
    return spla.solve_triangular(R,np.dot(Q.T,b))

def solve_model(M):
    A=M['A']
    b=M['b']
    return least_squares(A,b)

def create_model(data, type_model='linear'):
    if type_model == 'exponential': #f(x)=a0 \exp(a1*x) = \exp(\log(a0)+a1*x) -> log(f(x))=log(a0)+a1*x = A0+a1+x (it is linear now!)
        A = np.ones((data.shape[0],2))
        A[:,1] = data[:,0]
        b = np.log(data[:,1])
    M = {'A':A,
         'b':b,
         'type_model':type_model}

    return solve_model(M)


main()