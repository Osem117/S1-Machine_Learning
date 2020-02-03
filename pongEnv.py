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
        self.paddle.speed(0)  # Le seteamos la velocidad (un pixel por frame(?) )
        self.paddle.shapesize(1, 5)  # Las dimensiones de la pala
        self.paddle.penup()  # Pen Up. No dibuja mientras se mueve (?)
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
        self.ball.dx = 3  # Velocidad de la bola en el eje X   quiza en win 0.03
        self.ball.dy = -3  # Velocidad en el eje Y

        # Puntuacion
        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.hideturtle()  # Ocultar el contorno del objeto
        self.score.goto(0, 250)
        self.score.penup()
        self.score.write("Hit: {} Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # Controles de teclado
        self.win.onkey(self.paddle_right, 'Right')  # Llamar a la funcion cuando se pulse la flecha del teclado
        self.win.onkey(self.paddle_left, 'Left')  # Lo mismo, con la izquierda

    # Movimiento de la pala
    def paddle_right(self):
        x = self.paddle.xcor()  # Obtener la posicion X de la pala
        if x < 225:
            self.paddle.setx(x+20)  # Previendo los limites de la pantalla

    def paddle_left(self):
        x = self.paddle.xcor()
        if x >= -225:
            self.paddle.setx(x-20)

    # control de la ia  0-Left 1-nothing 2-right
    def reset(self):
        self.paddle.goto(0, -275)  # Al reiniciar el juego seteamos la pala en su pos inicial
        self.ball.goto(0, 100)  # Lo mismo con la bola
        return [self.paddle.xcor() * 0.01, self.ball.xcor() * 0.01, self.ball.ycor() * 0.01, self.ball.dx, self.ball.dy]

    def step(self, action):

        self.reward = 0
        self.done = 0

        if action == 0:  # Si accion 0, movemos la pala a la izquierda
            self.paddle_left()
            self.reward -= .1  # quitamos 0.1 de reward cuando se mueve la pala

        if action == 2:
            self.paddle_right()
            self.reward -= .1

        self.run_frame()  # funcion. Corre el juego un frame, la recompensa tambien se updatea

        # Vector de estado
        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

        return self.reward, state, self.done

    def run_frame(self):
        self.win.update()

        # Movimiento de la bola

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Colision con la pared

        if self.ball.xcor() > 290 or self.ball.xcor() < -290:
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.dy *= -1

        # Caida bola al suelo

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center',font=('Courier', 24, 'normal'))
            self.reward -= 3
            self.done = True

        # Colision con la pala

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center',font=('Courier', 24, 'normal'))
            self.reward += 3
