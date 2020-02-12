import turtle

import time
time.sleep(0.017)


class Pala:

    def __init__(self):
    
        self.hit, self.miss = 0, 0
        self.reward = 0
        self.done = False

        # Fondo
        self.win = turtle.Screen()  # Creamos la ventana
        self.win.title('Paddle')  # Le pomenos titulo
        self.win.bgcolor('black')  # Color del fondo
        self.win.tracer(0)  # Para las animaciones hasta win.update()
        self.win.listen()  # Escuchar las teclas
        self.win.setup(600, 600)  # Seteamos ancho y alto

        # Paleta
        self.paddle = turtle.Turtle()  # Creamos un objeto turtle
        self.paddle.shape('square')  # Le damos forma rectangular
        self.paddle.speed(0)  # Le seteamos la velocidad (un pixel por frame)
        self.paddle.shapesize(1, 5)  # Las dimensiones de la pala
        self.paddle.penup()  # Pen Up. No dibuja mientras se mueve
        self.paddle.color('white')  # Color de la pala
        self.paddle.goto(0, -275)  # Colocar la pala abajo en el centro

        # Bola
        self.ball = turtle.Turtle()  # Objeto turtle
        self.ball.speed(0)  # Velocidad de la pelota
        self.ball.shape('circle')  # Forma
        self.ball.color('red')  # color
        self.ball.penup()
        self.ball.goto(0, 100)  # Colocar la pelota un poco por encima del centro

        # Movimiento de la bola
        self.ball.dx = 3  # Velocidad de la bola en el eje X  win 0.03
        self.ball.dy = -3  # Velocidad en el eje Y

        # Puntuacion
        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.hideturtle()  # Ocultar el contorno del objeto
        self.score.goto(0, 250)
        self.score.penup()
        self.score.write("Hit: {} Missed: {}".format(self.hit, self.miss),
                         align='center', font=('Courier', 24, 'normal'))

        # Controles de teclado
        self.win.onkey(self.pr, 'Right')  # Llamar a la funcion cuando se pulse la flecha del teclado
        self.win.onkey(self.pl, 'Left')  # Lo mismo, con la izquierda

    # Movimiento de la pala
    def pr(self):
        x = self.paddle.xcor()  # Obtener la posicion X de la pala
        if x < 225:  # Si pala no toca lateral derecho
            self.paddle.setx(x+20)  # Previendo los limites de la pantalla movemos la pala

    def pl(self):
        x = self.paddle.xcor()
        if x >= -225:  # Lateral izdo
            self.paddle.setx(x-20)  # Moder pala izda

    # control de la ia  0-Left 1-nothing 2-right
    def reset(self):
        self.paddle.goto(0, -275)  # Al reiniciar el juego seteamos la pala en su pos inicial
        self.ball.goto(0, 100)  # Lo mismo con la bola
        # Devolvemos posiciones de pala y bola ademas de la direccion de la misma
        return [self.paddle.xcor() * 0.01, self.ball.xcor() * 0.01, self.ball.ycor() * 0.01, self.ball.dx, self.ball.dy]

    def step(self, action):

        self.reward = 0
        self.done = 0

        if action == 0:  # Si accion 0, movemos la pala a la izquierda
            self.pl()  # Usar el metodo implementado para ello
            self.reward -= 0.1  # menos 0.1 reward movimiento pala

        if action == 2:
            self.pr()  # Usar el metodo implementado para ello
            self.reward -= 0.1  # menos 0.1 reward movimiento pala

        self.frame()  # funcion. Corre el juego un frame, la recompensa tambien se updatea

        # Vector de estado
        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

        return self.reward, state, self.done

    def frame(self):  # Establecemos tod0 lo que ocurre durante un frame
        self.win.update()

        # Movimiento de la bola

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Colision con la pared

        if self.ball.xcor() > 290 or self.ball.xcor() < -290:
            self.ball.dx *= -1  # Cambiamos la direccion de la bola si choca con las paredes

        if self.ball.ycor() > 290:
            self.ball.dy *= -1  # Cambiamos direccion de la bola si choca con techo

        # Caida bola al suelo

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)  # Reset la posicion
            self.miss += 1  # Sumamos uno al conteo miss
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss),
                             align='center',font=('Courier', 24, 'normal'))
            self.reward -= 3  # Quitamos 3 puntos de recompensa
            self.done = True  # Juego terminado

        # Colision con la pala

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1  # Cambiamos direccion de la bola
            self.hit += 1  # Sumamos 1 al conteo hit
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss),
                             align='center',font=('Courier', 24, 'normal'))
            self.reward += 3  # Sumamos 3 de recompensa
