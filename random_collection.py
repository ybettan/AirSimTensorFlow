import os
import time
from datetime import datetime

import numpy as np

from AirSimClient import CarClient, CarControls
from prep_data import Point

class DataControl:
    def __init__(self):
        self.x1 = Point(0, 0)
        self.x2 = Point(0, 0)
        self.x3 = Point(0, 0)
        self.speed1 = 0
        self.speed2 = 0
        self.velocity2_x = 0
        self.velocity2_y = 0
        self.velocity1_x = 0
        self.velocity1_y = 0
        self.throttle2 = 0
        self.throttle1 = 0
        self.steering2 = 0
        self.steering1 = 0

    def getHeader(self):
        return "x1, y1, x2, y2, x3, y3," \
               " velocity1_x, velocity1_y, steering1, throttle1, " \
               "velocity2_x, velocity2_y, steering2, throttle2, \n"

    def __str__(self):
        return f"{self.x1.x}, {self.x1.y}, {self.x2.x}, {self.x2.y}, {self.x3.x}, {self.x3.y}, " \
               f"{self.velocity1_x}, {self.velocity1_y}, {self.steering1}, {self.throttle1}, " \
               f"{self.velocity2_x}, {self.velocity2_y},{self.steering2}, {self.throttle2}, \n"

    def advance_location(self, point):
        self.x1 = self.x2
        self.x2 = self.x3
        self.x3 = point

    def set_data(self, speed, avelocity_x, avelocity_y, asteering, athrottle):
        self.speed1 = self.speed2
        self.velocity1_x = self.velocity2_x
        self.velocity1_y = self.velocity2_y
        self.steering1 = self.steering2
        self.throttle1 = self.throttle2
        self.velocity2_x = avelocity_x
        self.velocity2_y = avelocity_y
        self.speed2 = speed
        self.steering2 = asteering
        self.throttle2 = athrottle

    def reset(self):
        self.x1 = Point(0, 0)
        self.x2 = Point(0, 0)
        self.x3 = Point(0, 0)
        self.speed1 = 0
        self.speed2 = 0
        self.velocity2 = 0
        self.velocity1 = 0
        self.throttle2 = 0
        self.throttle1 = 0
        self.steering2 = 0
        self.steering1 = 0


DATA_FREQUENCY = 0.3    # about 5 samples per second
DATA_DIR = "data_dir"   # directory for all the samples
FILE_SIZE = 5000       # max samples per file.

# Create image directory if it doesn't already exist
try:
    os.stat(DATA_DIR)
except:
    os.mkdir(DATA_DIR)

# connect to the AirSim simulator
client = CarClient()
client.confirmConnection()
print('Connected')
client.enableApiControl(True)
car_controls = CarControls()
client.reset()
cntrl = DataControl()

file_name = "collect_straight.csv"
s_mu = 0
with open(DATA_DIR + "/" + file_name, "w") as file:
    file.write(cntrl.getHeader())
    for j in range(1, 5):
        s_sigma = 0.01 * j
        t_sigma = 0.1 * j
        t_mu = 0.1 * j
        for k in range(6):
            car_controls.throttle = 0
            car_controls.steering = 0
            # set the new controls to the simul
            client.setCarControls(car_controls)
            time.sleep(1)
            client.reset()
            cntrl.reset()
            start_time = time.time()
            while True:
                collision_info = client.getCollisionInfo()
                if collision_info.has_collided or time.time() - start_time > 20:
                    break

                c_state = client.getCarState()
                cntrl.advance_location(Point(c_state.position[b'x_val'], c_state.position[b'y_val']))
                # now x1 is t-2, x2 & v & s & t are t-1, x3 is t.
                file.write(cntrl.__str__())
                n_steering = np.random.normal(s_mu, s_sigma, 1)[0]
                n_throttle = np.random.normal(t_mu, t_sigma, 1)[0]
                # set the commands and velocity for future knowledge
                cntrl.set_data(c_state.speed, c_state.velocity[b'x_val'], c_state.velocity[b'y_val'], n_steering,
                               n_throttle)
                car_controls.throttle = n_throttle
                car_controls.steering = n_steering
                # set the new controls to the simulator
                client.setCarControls(car_controls)
                # wait for the change to impact.
                time.sleep(DATA_FREQUENCY)


file_name = "collect_left.csv"
with open(DATA_DIR + "/" + file_name, "w") as file:
    file.write(cntrl.getHeader())
    for i in range(1, 5):
        for j in range(1, 5):
            s_mu = -0.2 * i
            t_mu = -0.2 * i
            s_sigma = 0.1 * j
            t_sigma = 0.1 * j
            for k in range(4):
                car_controls.throttle = 0
                car_controls.steering = 0
                # set the new controls to the simul
                client.setCarControls(car_controls)
                time.sleep(1)
                client.reset()
                cntrl.reset()
                start_time = time.time()
                while True:
                    collision_info = client.getCollisionInfo()
                    if collision_info.has_collided or time.time() - start_time > 20:
                        break

                    c_state = client.getCarState()
                    cntrl.advance_location(Point(c_state.position[b'x_val'], c_state.position[b'y_val']))
                    # now x1 is t-2, x2 & v & s & t are t-1, x3 is t.
                    file.write(cntrl.__str__())
                    n_steering = np.random.normal(s_mu, s_sigma, 1)[0]
                    n_throttle = np.random.normal(t_mu, t_sigma, 1)[0]
                    # set the commands and velocity for future knowledge
                    cntrl.set_data(c_state.speed, c_state.velocity[b'x_val'], c_state.velocity[b'y_val'],
                                   n_steering, n_throttle)
                    car_controls.throttle = n_throttle
                    car_controls.steering = n_steering
                    # set the new controls to the simulator
                    client.setCarControls(car_controls)
                    # wait for the change to impact.
                    time.sleep(DATA_FREQUENCY)

file_name = "collect_right.csv"
with open(DATA_DIR + "/" + file_name, "w") as file:
    file.write(cntrl.getHeader())
    for i in range(1, 5):
        for j in range(1, 5):
            s_mu = 0.2 * i
            t_mu = 0.2 * i
            s_sigma = 0.1 * j
            t_sigma = 0.1 * j
            for k in range(4):
                car_controls.throttle = 0
                car_controls.steering = 0
                # set the new controls to the simul
                client.setCarControls(car_controls)
                time.sleep(1)
                client.reset()
                cntrl.reset()
                start_time = time.time()
                while True:
                    collision_info = client.getCollisionInfo()
                    if collision_info.has_collided or time.time() - start_time > 20:
                        break

                    c_state = client.getCarState()
                    cntrl.advance_location(Point(c_state.position[b'x_val'], c_state.position[b'y_val']))
                    # now x1 is t-2, x2 & v & s & t are t-1, x3 is t.
                    file.write(cntrl.__str__())
                    n_steering = np.random.normal(s_mu, s_sigma, 1)[0]
                    n_throttle = np.random.normal(t_mu, t_sigma, 1)[0]
                    # set the commands and velocity for future knowledge
                    cntrl.set_data(c_state.speed, c_state.velocity[b'x_val'], c_state.velocity[b'y_val'],
                                   n_steering, n_throttle)
                    car_controls.throttle = n_throttle
                    car_controls.steering = n_steering
                    # set the new controls to the simulator
                    client.setCarControls(car_controls)
                    # wait for the change to impact.
                    time.sleep(DATA_FREQUENCY)

while True:
    s_mu, s_sigma, t_mu, t_sigma, = np.random.normal(0, 0.6, 4)
    s_sigma = abs(s_sigma)
    t_sigma = abs(t_sigma)
    file_name_head = datetime.now().strftime("%m_%d_%H_%S")
    file_name_tail = "_sm{}_ss{}_tm{}_ts{}.csv".format(int(s_mu*100), int(s_sigma*100), int(t_mu*100), int(t_sigma*100))
    file_name = file_name_head + file_name_tail

    with open(DATA_DIR + "/" + file_name, "w") as file:
        file.write(cntrl.getHeader())
        for i in range(FILE_SIZE):
            collision_info = client.getCollisionInfo()
            if collision_info.has_collided:
                car_controls.throttle = 0
                car_controls.steering = 0
                # set the new controls to the simul
                client.setCarControls(car_controls)
                time.sleep(1)
                client.reset()
                cntrl.reset()

            car_state = client.getCarState()
            cntrl.advance_location(Point(car_state.position[b'x_val'], car_state.position[b'y_val']))
            # now x1 is t-2, x2 & v & s & t are t-1, x3 is t.
            file.write(cntrl.__str__())
            new_throttle = np.random.normal(t_mu, t_sigma, 1)[0]
            new_steering = np.random.normal(s_mu, s_sigma, 1)[0]
            # set the commands and velocity for future knowledge
            cntrl.set_data(car_state.speed, car_state.velocity[b'x_val'], car_state.velocity[b'y_val'], new_steering, new_throttle)
            car_controls.throttle = new_throttle
            car_controls.steering = new_steering
            # set the new controls to the simulator
            client.setCarControls(car_controls)
            # wait for the change to impact.
            time.sleep(DATA_FREQUENCY)

