import gfootball.env as football_env
import numpy as np
import tensorflow as tf

from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Flatten  # modulos para modelos de algunas capas de redes neuronales
from keras.models import Model  # import del modelo en si desde keras
from keras.optimizers import Adam  # para compilar la red  (buscar que es esto y las otras opciones
import keras.backend as K
from keras.applications.mobilenet_v2 import MobileNetV2  # extractor de caracteristics de imagen pre-train(?)

gamma = 0.99  # Valor de gamma, el descuento de la recompensa
lmbda = 0.95
clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001


# method obtain advantages using GAE alg
def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    print('ESTO EL ADV' + str(adv))
    print('ESTO SON LOS RETURNS' + str(np.array(returns)))
    print('ESTO SON LOS VALUES' + str(values[:-1]))
    return returns, ((adv - np.mean(adv)) / (np.std(adv) + 1e-10))  # advant normalizado (y el 1e-10 para evitar el /0)
# -----------------------------------------------------------------------


# Print de la perdida de ppo (?)
def ppo_loss_print(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        y_true = tf.Print(y_true, [y_true], 'y_true: ')
        y_pred = tf.Print(y_pred, [y_pred], 'y_pred: ')
        newpolicy_probs = y_pred
        # newpolicy_probs = y_true * y_pred
        newpolicy_probs = tf.Print(newpolicy_probs, [newpolicy_probs], 'new policy probs: ')

        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        ratio = tf.Print(ratio, [ratio], 'ratio: ')
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        actor_loss = tf.Print(actor_loss, [actor_loss], 'actor_loss: ')
        critic_loss = K.mean(K.square(rewards - values))
        critic_loss = tf.Print(critic_loss, [critic_loss], 'critic_loss: ')
        term_a = critic_discount * critic_loss
        term_a = tf.Print(term_a, [term_a], 'term_a: ')
        term_b_2 = K.log(newpolicy_probs + 1e-10)
        term_b_2 = tf.Print(term_b_2, [term_b_2], 'term_b_2: ')
        term_b = entropy_beta * K.mean(-(newpolicy_probs * term_b_2))
        term_b = tf.Print(term_b, [term_b], 'term_b: ')
        total_loss = term_a + actor_loss - term_b
        total_loss = tf.Print(total_loss, [total_loss], 'total_loss: ')
        return total_loss

    return loss


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss


# Image porque nuestro actor va a ser una imagen que viene del juego
def get_model_actor_image(input_dims, output_dims):  # Actor
    state_input = Input(shape=input_dims)  # definir la forma de entrada de nuestra imagen

    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    # weights -> pesos pre-entrenados
    # inc_top=false -> elimiar las capas de clasigicacion al final de la imagen pre entrenada del modelo de redes y
    # reemplazarlas con las nuestras
    feature_extractor = MobileNetV2(weights='imagenet', include_top=False)

    for layer in feature_extractor.layers:
        layer.trainable = False  # no queremos volver a entrenar las capas

    x = Flatten(name='flatten')(feature_extractor(state_input))  # aplanar la salida de la red. Toma el state como input

    # creamos una capa totalmente conectada con 1024 neuronas.  Buscar lo de 'relu' que es y sus opciones
    x = Dense(1024, activation='relu', name='fc1')(x)  # La entrada es x, la capa anterior

    # Salida -> Distribucion probabilistica sobre las posibles acciones disponibles
    # La distribucion probabilistica sobre la que queremos predecir   'name' (?)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)  # activation 'softmax' (?)

    # Combinar las capas en el modelo Keras
    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])

    # Compilar el modelo. lr learning rate   perdida/loss, mean squared error (luego sera sustituido por una perdida
    # custom del algoritmo ppo
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    model.summary()
    return model
# --------------------------------------------------


#  ver que es esto(?)(?)(?)
def get_model_actor_simple(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input)
    x = Dense(256, activation='relu', name='fc2')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    # model.summary()
    return model


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
    out_actions = Dense(1, activation='tanh')(x)  # activation 'tanh' (?)

    # Combinar las capas en el modelo Keras
    model = Model(inputs=[state_input], outputs=[out_actions])

    # Compilar el modelo. lr learning rate   perdida/loss, mean squared error (luego sera sustituido por una perdida
    # custom del algoritmo ppo
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model
# --------------------------------------------------- 


# ver que es esto.  hay dos simples de model image y critic
def get_model_critic_simple(input_dims):
    state_input = Input(shape=input_dims)

    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input)
    x = Dense(256, activation='relu', name='fc2')(x)
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    # model.summary()
    return model


# comentar~
def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 20:
            break
    return total_reward


# onehot encoding, definicion
def one_hot_encoding(probs):
    one_hot = np.zeros_like(probs)
    one_hot[:, np.argmax(probs, axis=1)] = 1
    return one_hot


# si el input es una imagen o no
image_based = False

if image_based:
    # arg1 -> env_name: entorno en el que vamos a ejecutar. En este caso, porteria vacia como primera toma de contacto
    # arg2 -> representation: Tipo de observación para describir el estado de nuesto agente.
    #         Pixel: vision basada en un frame del juego
    # arg3 -> render:True, para renderizar el juego en nuestra pantalla (para poder verlo)
    env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
else:
    env = football_env.create_environment(env_name='academy_empty_goal', representation='simple115')

state = env.reset()

state_dims = env.observation_space.shape  # Dimension del estado. Puesto que es una imagen, es la resolución de la misma
print('dim ->', state_dims)

n_actions = env.action_space.n  # Numero de acciones disponibles (up, down, kick etc)
print('numActions ->', n_actions)

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

# Para tener logs en un archivo
tensor_board = TensorBoard(log_dir='./logs')

# creamos los modelos segun si estan basados en imagen o no (?)  (hace exactamente lo mismo)
if image_based:
    model_actor = get_model_actor_image(input_dims=state_dims, output_dims=n_actions)
    model_critic = get_model_critic_image(input_dims=state_dims)
else:
    model_actor = get_model_actor_simple(input_dims=state_dims, output_dims=n_actions)
    model_critic = get_model_critic_simple(input_dims=state_dims)

# Var globales para el bucle while
ppo_steps = 128  # numero de interacciones con el entorno. Numero de t que se va a interactuar
target_reached = False
iters = 0
max_iters = 50
best_reward = 0

# Checkpoint for observation. Cuando recolectamos informacion necesitamos almacenarla para poder entrenar el modelo
# y llegar a conseguir uno que nos satisfaga como resultado
while not target_reached and iters < max_iters:

    # Almacenar estados, acciones, valores, etc en cada iteración
    states = []  # estados
    actions = []  # acciones
    values = []  # valores generados por el interprete del modelo
    masks = []  # Check si el juego se encuentra finalizado o no para reiniciar
    rewards = []  # Recompesas
    actions_probs = []  # probabilidades sobre las acciones disponibles
    actions_onehot = []  # el onehot sobre las acciones
    state_input = None  # El estado inicial (?)
    
    
#  --------------------------AQUI SE HACE EL PASO A PASO PPO ¿?¿?¿?¿?. VER DONDE SE HACE EL TRAIN PREDICT EN EL OTRO
    
    for itr in range(ppo_steps):  # Steps -> t   (time steps,
        state_input = K.expand_dims(state, 0)  # Input tensor de un estado usando la extensión de K

        # predecir la accion desde el estado actual (steps = 1) (?)
        # significa que nuestro lote solo tiene un ejemplo para el cual queremos llevar a cabo la prediccion
        # Deberia devolver una distribucion de probabilidades de 21 espacios de estado correspondiente a
        # las 21 acciones disponibles para este entorno concreto
        # Policy -> pi (decide action based on the state observed
        action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)

        # valores de recompensa
        q_value = model_critic.predict([state_input], steps=1)

        # Seleccionar la accion para esta distrib. P es la probabilidad de muestreo, la cual viene del modelo del actor
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
        print('itr: ' + str(itr) + ', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))
        mask = not done  # Mas tarde se vera donde se usa esto.

        states.append(state)  # State -> S_t  (step of the game for every timestep)
        actions.append(action)  # Next state -> S_t+1   (next state in time t+1)
        actions_onehot.append(action_onehot)  # Action -> a_t     (action took by the agent)
        values.append(q_value)  # Val func(state s) -> V(s) (values collected from critic model. takes a state as input
        masks.append(mask)  # Mask -> m_t  (game done or not. 4 expl If goal scored, game is over and need 2 reset env)
        rewards.append(reward)  # Reward -> r_t  (rewards observed from interactions with the game/env
        actions_probs.append(action_dist)  # Action probability -> pi(s)  (last layer of neural network.
        # Vector containing the prob for each possible action

        state = observation  # Tras cada accion actualizar el estado, para no usar el mismo estado inicial siempre

        if done:  # si el juego se ha acabado, reseteamos el entorno para poder finalizar con todos los pasos que hemos
            env.reset()  # seteado ej. 128 interacc, el juego acaba en 70, querremos que se ejecuten las 58 restantes

        q_value = model_critic.predict(state_input, steps=1)  # Q-value of the last state observed in the game
        values.append(q_value)  # inside the loop we got this obs. but exited loop b4 runing it through the critic model

        # Calculate advantages from returns - to train model actor
        returns, advantages = get_advantages(values, masks, rewards)

        # Train the model
        actor_loss = model_actor.fit(
            [states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
            [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=8)
        critic_loss = model_critic.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                       verbose=True)

        # Model evaluation. Para ello vamos a calcular la media de las recompensas. Esto va a rular el juego 5 veces
        avg_reward = np.mean([test_reward() for _ in range(10)])
        print('total test rewards=' + str(avg_reward))
        if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            model_actor.save('model_actor_{}_{}.hdf5'.format(iters, avg_reward))
            model_critic.save('model_critic_{}_{}.hdf5'.format(iters, avg_reward))
            best_reward = avg_reward
        if best_reward > 0.9 or iters > max_iters:
            target_reached = True
        iters += 1
        env.reset()

env.close()
