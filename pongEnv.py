import turtle

import time
time.sleep(0.017)

win = turtle.Screen()  # Creamos la ventana
win.title('Paddle')  # Le pomenos titulo
win.bgcolor('black')  # Color del fondo
win.tracer(0)  # Para las animaciones hasta win.update()
win.listen()  # Escuchar las teclas
win.setup(600, 600)  # Seteamos ancho y alto

# Paleta
paddle = turtle.Turtle()  # Creamos un objeto turtle
paddle.shape('square')  # Le damos forma rectangular
paddle.speed(0)  # Le seteamos la velocidad (un pixel por frame(?) )
paddle.shapesize(1, 5)  # Las dimensiones de la pala
paddle.penup()  # Pen Up. No dibuja mientras se mueve (?)
paddle.color('white')  # Color de la pala
paddle.goto(0, -275)  # Colocar la pala abajo en el centro

# Bola
ball = turtle.Turtle()  # Objeto turtle
ball.speed(0)  # Velocidad de la pelota
ball.shape('circle')  # Forma
ball.color('red')  # color
ball.penup()
ball.goto(0, 100)  # Colocar la pelota un poco por encima del centro

# Movimiento de la bola
ball.dx = 2  # Velocidad de la bola en el eje X   quiza en win 0.03
ball.dy = -2  # Velocidad en el eje Y


# Movimiento de la pala
def paddle_right():
    x = paddle.xcor()  # Obtener la posicion X de la pala
    if x < 225:
        paddle.goto(paddle.xcor() + 30, paddle.ycor())  # Teniendo en cuenta los limites de la pantalla

def paddle_left():
    x = paddle.xcor()
    if x >= -225:
        paddle.goto(paddle.xcor() - 30, paddle.ycor())


# Controles de teclado
win.onkey(paddle_right, 'Right')  # Llamar a la funcion cuando se pulse la flecha del teclado
win.onkey(paddle_left, 'Left')  # Lo mismo, con la izquierda


def move_ball():
    ball.goto(ball.xcor()+ball.dx, ball.ycor()+ball.dy)

    # Colisiones con las pareces
    if ball.xcor() >= 280 or ball.xcor() <= -290:
        ball.dx *= -1

    if ball.ycor() >= 270:
        ball.dy *= -1

    # Reiniciar la pelota si cae al fondo
    if ball.ycor() <= -285:
        ball.goto(0, 0)
        ball.dy *= -1


# Colision con la pala
def ball_bounce():
    # si la pala y la pelota chocan cambiar la direccion de la pala
    if ball.dy < 0 and ball.ycor() <= -253 and (paddle.xcor()-60 <= ball.xcor() <= paddle.xcor()+60):
        ball.dy *= -1


while True:
    win.update()  # Mostramos la pantalla de forma contínua
    move_ball()
    ball_bounce()
    time.sleep(0.017)  # windows


