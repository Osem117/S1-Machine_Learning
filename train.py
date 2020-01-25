import gfootball.env as football_env
import keras.backend as K
import numpy as np
from keras.layers import Input, Dense, Flatten  # modulos para modelos de algunas capas de redes neuronales
from keras.models import Model  # import del modelo en si desde keras
from keras.optimizers import Adam  # para compilar la red  (buscar que es esto y las otras opciones
from keras.applications.mobilenet_v2 import MobileNetV2  # extractor de caracteristicas de imagen pre-entrenada (?)

gamma = 0.99  # Valor de gamma, el descuento de la recompensa
lambda_ = 0.95


# method obtain advantages using GAE alg
def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lambda_ * masks[i] * gae
        returns.insert(0, gae+values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv))/(np.std(adv) + 1e-10)  # advantaje normalizado (y el 1e-10 para evitar el /0 )
# -----------------------------------------------------------------------


# Image porque nuestro actor va a ser una imagen que viene del juego
def get_model_actor_image(input_dims):  # Actor, posibles acciones a tomar
    state_input = Input(shape=input_dims)  # definir la forma de entrada de nuestra imagen

    # weights -> pesos pre-entrenados
    # inc_top=false -> elimiar las capas de clasigicacion al final de la imagen pre entrenada del modelo de redes y
    # reemplazarlas con las nuestras
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False)

    for layer in feature_extractor.layers:
        # no queremos volver a entrenar las capas
        layer.trainable = False

    x = Flatten(name='flatten')(feature_extractor(state_input))  # aplanar la salida de la red. Toma el state como input

    # creamos una capa totalmente conectada con 1024 neuronas.  Buscar lo de 'relu' que es y sus opciones
    x = Dense(1024, activation='relu', name='fcl')(x)  # La entrada es x, la capa anterior

    # Salida -> Distribucion probabilistica sobre las posibles acciones disponibles
    # La distribucion probabilistica sobre la que queremos predecir   'name' (?)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)  # activation 'softmax' (?)

    # Combinar las capas en el modelo Keras
    model = Model(inputs=[state_input], outputs=[out_actions])

    # Compilar el modelo. lr learning rate   perdida/loss, mean squared error (luego sera sustituido por una perdida
    # custom del algoritmo ppo
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model
# --------------------------------------------------


def get_model_critic_image(input_dims):  # Modelo del interprete
    state_input = Input(shape=input_dims)  # definir la forma de entrada de nuestra imagen

    # weights -> pesos pre-entrenados
    # inc_top=false -> elimiar las capas de clasigicacion al final de la imagen pre entrenada del modelo de redes y
    # reemplazarlas con las nuestras
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False)

    for layer in feature_extractor.layers:
        # no queremos volver a entrenar las capas
        layer.trainable = False

    x = Flatten(name='flatten')(feature_extractor(state_input))  # aplanar la salida de la red. Toma el state como input

    # creamos una capa totalmente conectada con 1024 neuronas.  Buscar lo de 'relu' que es y sus opciones
    x = Dense(1024, activation='relu', name='fcl')(x)  # La entrada es x, la capa anterior

    # Salida -> Numero real
    # Esta salida dice lo buena o mala que es la accion que el actor ha realizao
    out_actions = Dense(1, activation='tanh', name='predictions')(x)  # activation 'tanh' (?)

    # Combinar las capas en el modelo Keras
    model = Model(inputs=[state_input], outputs=[out_actions])

    # Compilar el modelo. lr learning rate   perdida/loss, mean squared error (luego sera sustituido por una perdida
    # custom del algoritmo ppo
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model
# --------------------------------------------------- 


# arg1 -> env_name: entorno en el que vamos a ejecutar. En este caso, porteria vacia como primera toma de contacto
# arg2 -> representation: Tipo de observación para describir el estado de nuesto agente.
#         Pixel: vision basada en un frame del juego
# arg3 -> render:True, para renderizar el juego en nuestra pantalla (para poder verlo)
env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)

state = env.reset()

state_dims = env.observation_space.shape  # Dimension del estado. Puesto que es una imagen, es la resolución de la misma
print('dim ->', state_dims)

n_actions = env.action_space.n  # Numero de acciones disponibles (up, down, kick etc)
print('numActions ->', n_actions)

ppo_steps = 128  # numero de interacciones con el entorno. Numero de t que se va a interactuar

# Almacenar estados, acciones, valores, etc
states = []  # estados
actions = []  # acciones
values = []  # valores generados por el interprete del modelo
masks = []  # Check si el juego se encuentra finalizado o no para reiniciar
rewards = []  # Recompesas
actions_probs = []  # probabilidades sobre las acciones disponibles
actions_onehot = []  # el onehot sobre las acciones

# modelo del actor
model_actor = get_model_actor_image(input_dims=state_dims)

# Modelo del supervisor (critic model) generacion de recompensas etc
model_critic = get_model_critic_image(input_dims=state_dims)

for itr in range(ppo_steps):  # Steps -> t   (time steps,
    state_input = K.expand_dims(state, 0)  # Input tensor de un estado usando la extensión de K

    # predecir la accion desde el estado actual (steps = 1) (?)
    # significa que nuestro lote solo tiene un ejemplo para el cual queremos llevar a cabo la prediccion
    # Deberia devolver una distribucion de probabilidades de 21 espacios de estado correspondiente a
    # las 21 acciones disponibles para este entorno concreto
    action_dist = model_actor.predict([state_input], steps=1)  # Policy -> pi (decide action based on the state observed

    # valores de recompensa
    q_value = model_critic.predict([state_input], steps=1)

    # Seleccionar la accion para esta distribución. P es la probabilidad de muestreo, la cual viene del modelo del actor
    action = np.random.choice(n_actions, p=action_dist[0, :])

    # Convertimos nuestra accion preferida como oneHot para poder utilizarla despues en el entrenamiento
    action_onehot = np.zeros(n_actions)

    # Seteamos el index de la acción muestreada (la accion que queremos tomar basada en la salida del modelo) a 1
    action_onehot[action] = 1

    # observation -> siguiente estado después de tomar la acción
    # reward -> recompensa obtenida por la accion tomada
    # done -> como mask. Indica si el juego esta finalizado o no
    # info -> informacion (luego vere que trae)
    observation, reward, done, info = env.step(action)  # tupla que devuelve estos elementos despues de cada acción

    mask = not done  # Mas tarde se vera donde se usa esto.

    states.append(state)  # State -> S_t  (step of the game for every timestep)
    actions.append(action)  # Next state -> S_t+1   (next state in time t+1)
    actions_onehot.append(action_onehot)  # Action -> a_t     (action took by the agent)
    values.append(q_value)  # Value func(state s) -> V(s) (values collected from critic model. takes a state as input
    masks.append(mask)  # Mask -> m_t  (game done or not. 4 expl If goal scored, game is over and need 2 reset env)
    rewards.append(reward)  # Reward -> r_t  (rewards observed from interactions with the game/env
    actions_probs.append(action_dist)  # Action probability -> pi(s)  (last layer of neural network.
    # Vector containing the prob for each possible action

    state = observation  # Tras cada accion actualizar el estado, para no usar el mismo estado inicial siempre

    if done:  # si el juego se ha acabado, reseteamos el entorno para poder finalizar con todos los pasos que hemos
        env.reset()  # seteado ej. 128 interacciones, el juego acaba en 70, querremos que se ejecuten las 58 restantes

    state_input = K.expand_dims(state, 0)  # input tensor for the last step observed
    q_value = model_critic.predict(state_input, steps=1)  # Q-value of the last state observed in the game
    values.append(q_value)  # inside the loop we got this obs. but exited loop b4 runing it through the critic model

    # Calculate advantages from returns - to train model actor
    returns, advantages = get_advantages(values, masks, rewards)

env.close()
