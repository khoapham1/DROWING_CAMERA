import time
import math
import numpy as np

#### PID parameters
class PIDController(object):
    def __init__(self, Kp, Ki, Kd, max_output):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.reset()

    def reset(self):
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            return 0

        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        output = np.clip(output, -self.max_output, self.max_output)

        self.last_error = error
        self.last_time = current_time
        return output
        
    def control_drone_to_center(errors):
        error_x, error_y = errors
        if error_x != 0.0 or error_y != 0.0:
            vx = PID_Y.update(-error_y)
            vy = PID_X.update(error_x)
            print("[PID CONTROL] Sending velocity vx: {:.3f}, vy: {:.3f} for color id".format(vx, vy))
            send_local_ned_velocity(vx, vy, 0)
            return True
        send_local_ned_velocity(0, 0, 0)
        return False

