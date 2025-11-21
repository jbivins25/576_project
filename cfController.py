import sys
import time
import csv

import logging
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E702')
logging.basicConfig(level=logging.ERROR)
position_estimate = [0,0,0]
writeFile = open("trajectoryEstimation.txt", 'w')

with open('testTraj.csv', newline='') as csvfile:
	next(csvfile)
	reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
	testTraj = list(reader)
zero_x = testTraj[1][1]
zero_y = testTraj[1][9]
print("Arranging data...")
for row in range(len(testTraj)):
	testTraj[row][1] = testTraj[row][1] - zero_x
	testTraj[row][9] = testTraj[row][9] - zero_y
print("Done!")
print("Verifying...")
for row in range(len(testTraj)):
    for column in range(len(testTraj[row])):
            assert(isinstance(testTraj[row][column],float))
for row in range(len(testTraj)):
	print(testTraj[row][1],testTraj[row][9],testTraj[row][17])

print("Done!")

#testTraj[:] = testTraj[:10]

def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)


def activate_mellinger_controller(cf):
    cf.param.set_value('stabilizer.controller', '2')


def upload_trajectory(cf, trajectory_id, trajectory):
    trajectory_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]
    trajectory_mem.trajectory = []

    total_duration = 0
    for row in trajectory:
        duration = row[0]
        x = Poly4D.Poly(row[1:9])
        y = Poly4D.Poly(row[9:17])
        z = Poly4D.Poly(row[17:25])
        yaw = Poly4D.Poly(row[25:33])
        trajectory_mem.trajectory.append(Poly4D(duration, x, y, z, yaw))
        total_duration += duration

    upload_result = trajectory_mem.write_data_sync()
    if not upload_result:
        print('Upload failed, aborting!')
        sys.exit(1)
    cf.high_level_commander.define_trajectory(trajectory_id, 0, len(trajectory_mem.trajectory))
    return total_duration


def run_sequence(cf, trajectory_id, duration):
    commander = cf.high_level_commander

    commander.takeoff(1.0, 2.0)
    time.sleep(3.0)
    relative = True
    commander.start_trajectory(trajectory_id, 1.0, relative)
    time.sleep(duration)
    commander.land(0.0, 2.0)
    time.sleep(2)
    commander.stop()

def log_pos_callback(timestamp,data,logconf):
    global positon_estimate
    global writeFile
    writeFile.write(str(data))
    writeFile.write('\n')
    position_estimate[0] = data['stateEstimate.x']
    position_estimate[1] = data['stateEstimate.y']
    position_estimate[2] = data['stateEstimate.z']

if __name__ == '__main__':
    print("Initializing drivers...")
    cflib.crtp.init_drivers()
    print("Done!")
    print("Connecting...")
    available = cflib.crtp.scan_interfaces(0xE7E7E7E703)
    if not available:
        print('No Crazyflies found!')
        sys.exit(1)
    uri = available[0][0]
    print("Found uri:", uri)
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("Done!")
        cf = scf.cf
        trajectory_id = 1
        print('Uploading...')
        duration = upload_trajectory(cf, trajectory_id, testTraj)
        print('Done!')
        print('The sequence is {:.1f} seconds long'.format(duration))
        print('Setting up logger...')
        logconf = LogConfig(name='Position', period_in_ms = 10)
        logconf.add_variable('stateEstimate.x','float')
        logconf.add_variable('stateEstimate.y','float')
        logconf.add_variable('stateEstimate.z','float')
        cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)
        print('Done!')
        reset_estimator(cf)
        logconf.start()
        print("Running trajectory...")
        run_sequence(cf, trajectory_id, duration)
        print("Done!")
        logconf.stop()
