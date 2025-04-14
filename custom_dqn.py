from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import signal
import random
import logging
import time
from collections import deque
import threading
import tkinter as tk

import pygame
import cv2
import numpy as np

# TensorFlow configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = 'FALSE'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_v2_behavior()

sys.path.append("game/")
from game import wrapped_flappy_bird as game

# =============================================================================
# Global settings
# =============================================================================
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 100000
EXPLORE = 2000000
FINAL_EPSILON = 0.1    # final exploration rate
INITIAL_EPSILON = 1.0  # start fully random
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1
MAX_TIMESTEPS = 500000
REWARD_ALIVE = 0.1

stop_training = False
pause_training = False
score = 0
hiscore = 0
human_mode = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISCORE_FILE = os.path.join(BASE_DIR, "hiscore.txt")

# =============================================================================
# Helper functions
# =============================================================================
def load_hiscore():
    global hiscore
    try:
        if os.path.exists(HISCORE_FILE):
            with open(HISCORE_FILE, "r") as f:
                hiscore = int(f.read().strip())
    except:
        hiscore = 0


def save_hiscore():
    try:
        with open(HISCORE_FILE, "w") as f:
            f.write(str(hiscore))
    except:
        pass

load_hiscore()

# =============================================================================
# Signal handler
# =============================================================================
def signal_handler(sig, frame):
    global stop_training
    print("\n[INFO] Stopping training...")
    stop_training = True

signal.signal(signal.SIGINT, signal_handler)

# =============================================================================
# Control Panel UI
# =============================================================================
def launch_control_panel():
    global score, hiscore, control_root, human_mode

    def update_labels():
        score_label.config(text=f"{score}")
        hiscore_label.config(text=f"{hiscore}")
        control_root.after(100, update_labels)

    control_root = tk.Tk()
    control_root.title("Flappy AI Control")
    control_root.config(bg="#2c3e50")
    
    # Score Display
    score_frame = tk.Frame(control_root, bg="#2c3e50")
    score_frame.pack(pady=10)
    tk.Label(score_frame, text="SCORE", fg="#ecf0f1", bg="#2c3e50").pack()
    score_label = tk.Label(score_frame, text="0", fg="#e74c3c", bg="#2c3e50", font=("Arial", 16))
    score_label.pack()
    tk.Label(score_frame, text="HIGH SCORE", fg="#ecf0f1", bg="#2c3e50").pack()
    hiscore_label = tk.Label(score_frame, text=str(hiscore), fg="#2ecc71", bg="#2c3e50", font=("Arial", 16))
    hiscore_label.pack()

    # Control Buttons
    btn_frame = tk.Frame(control_root, bg="#2c3e50")
    btn_frame.pack(pady=10)
    btn_style = {"font": ("Arial", 10), "width": 12, "height": 1, "relief": "raised"}

    def make_btn(text, cmd, bg, fg):
        return tk.Button(btn_frame, text=text, command=cmd,
                          bg=bg, fg=fg,
                          activebackground=bg, activeforeground=fg,
                          highlightbackground=bg,
                          **btn_style)

    make_btn("PAUSE", lambda: globals().update(pause_training=True), "#3498db", "white").grid(row=0, column=0, padx=5)
    make_btn("RESUME", lambda: globals().update(pause_training=False), "#2ecc71", "white").grid(row=0, column=1, padx=5)
    make_btn("HUMAN MODE", lambda: globals().update(human_mode=True), "#e67e22", "white").grid(row=1, column=0, pady=5)
    make_btn("AI MODE", lambda: globals().update(human_mode=False), "#9b59b6", "white").grid(row=1, column=1, pady=5)
    make_btn("EXIT", lambda: [globals().update(stop_training=True), control_root.destroy()], "#e74c3c", "white").grid(row=2, column=0, columnspan=2, pady=5)

    update_labels()
    control_root.mainloop()

threading.Thread(target=launch_control_panel, daemon=True).start()

# =============================================================================
# Network Definition
# =============================================================================
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

def createNetwork():
    s = tf.placeholder(tf.float32, [None, 80, 80, 4], name='state')

    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1,4,4,1], padding="SAME") + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,2,2,1], padding="SAME") + b_conv2)

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1,1,1,1], padding="SAME") + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

# =============================================================================
# Training Loop
# =============================================================================
def trainNetwork(s, readout, h_fc1, sess):
    global stop_training, pause_training, score, hiscore

    pygame.init()
    screen = pygame.display.set_mode((288,512))
    pygame.display.set_caption("Flappy Bird DQN")
    clock = pygame.time.Clock()

    a = tf.placeholder(tf.float32, [None, ACTIONS], name='action')
    y = tf.placeholder(tf.float32, [None], name='target')
    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
        ckpt_vars = set(reader.get_variable_to_shape_map().keys())
        vars_to_restore = [v for v in tf.global_variables() if v.name.split(':')[0] in ckpt_vars]
        restore_saver = tf.train.Saver(var_list=vars_to_restore)
        restore_saver.restore(sess, checkpoint.model_checkpoint_path)
        print(f"[INFO] Restored {len(vars_to_restore)} vars from checkpoint")
        epsilon = 0.0
    else:
        print("[INFO] No checkpoint, starting fresh.")
        epsilon = INITIAL_EPSILON

    game_state = game.GameState()
    D = deque()
    do_nothing = np.zeros(ACTIONS); do_nothing[0] = 1
    x_t, _, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)), cv2.COLOR_BGR2GRAY)
    _, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t,)*4, axis=2)

    t = 0

    while not stop_training and t < MAX_TIMESTEPS:
        clock.tick(60)
        if pause_training:
            while pause_training and not stop_training:
                time.sleep(0.1)
            continue

        action = np.zeros(ACTIONS); action[0] = 1
        if human_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_training = True
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                action = np.array([0,1])
        else:
            if t % FRAME_PER_ACTION == 0:
                readout_t = sess.run(readout, feed_dict={s:[s_t]})
                if random.random() <= epsilon:
                    action_index = random.randrange(ACTIONS)
                else:
                    action_index = np.argmax(readout_t[0])
                action = np.eye(ACTIONS)[action_index]

        x_t1, reward, terminal = game_state.frame_step(action)
        score += int(reward)
        if not terminal: reward += REWARD_ALIVE

        x_t1 = cv2.cvtColor(cv2.resize(x_t1,(80,80)), cv2.COLOR_BGR2GRAY)
        _, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1,(80,80,1))
        s_t1 = np.append(s_t[:,:,1:], x_t1, axis=2)

        D.append((s_t, action, reward, s_t1, terminal))
        if len(D) > REPLAY_MEMORY: D.popleft()

        if t > OBSERVE and not human_mode:
            minibatch = random.sample(D, BATCH)
            s_batch = np.array([d[0] for d in minibatch])
            a_batch = np.array([d[1] for d in minibatch])
            r_batch = np.array([d[2] for d in minibatch])
            s1_batch = np.array([d[3] for d in minibatch])
            term_batch = np.array([d[4] for d in minibatch])

            y_batch = []
            readout1 = sess.run(readout, feed_dict={s: s1_batch})
            for i in range(len(minibatch)):
                y_batch.append(r_batch[i] if term_batch[i] else r_batch[i] + GAMMA * np.max(readout1[i]))

            sess.run(train_step, feed_dict={s: s_batch, a: a_batch, y: y_batch})

        s_t = s_t1
        t += 1
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if t % 10000 == 0 and not human_mode:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        if terminal:
            if score > hiscore:
                hiscore = score
                save_hiscore()

            game_state = game.GameState()
            x_t, _, terminal = game_state.frame_step(do_nothing)
            x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)), cv2.COLOR_BGR2GRAY)
            _, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
            s_t = np.stack((x_t,)*4, axis=2)
            score = 0
            time.sleep(0.5)

        pygame.display.flip()

    # end
    pygame.quit()
    # final high score update in case of exit during play
    if score > hiscore:
        hiscore = score
    save_hiscore()
    saver.save(sess, 'saved_networks/' + GAME + '-dqn-final')
    print("Training completed.")

# =============================================================================
# Main
# =============================================================================
def main():
    with tf.Session() as sess:
        s, readout, h_fc1 = createNetwork()
        trainNetwork(s, readout, h_fc1, sess)

if __name__ == '__main__':
    main()
