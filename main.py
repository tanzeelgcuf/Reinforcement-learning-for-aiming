
import random
import sys

sys.path.append('MacAPI')
import numpy as np
import pandas as pd
import pickle
import sim

# Max movement along X
low, high = -0.05, 0.05
exp_rate = 0.4
learning_rate = 0.01
position_list=[]
position_list = np.linspace(-2.5,2.5,51)

qTable_dict = {} 
action_space = [-2,-1,0,1,2]
for i in position_list:
    rounded_i = round(i,1)
    qTable_dict[rounded_i] = {}
    for action in action_space:
        qTable_dict[rounded_i][action] = 0

state_action_list = []
reward_list = []

def setNumberOfBlocks(clientID, blocks, typeOf, mass, blockLength,
                      frictionCube, frictionCup):
    '''
        Function to set the number of blocks in the simulation
        '''
    emptyBuff = bytearray()
    res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
        clientID, 'Table', sim.sim_scripttype_childscript, 'setNumberOfBlocks',
        [blocks], [mass, blockLength, frictionCube, frictionCup], [typeOf],
        emptyBuff, sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        print(
            'Results: ', retStrings
        )  # display the reply from CoppeliaSim (in this case, the handle of the created dummy)
    else:
        print('Remote function call failed')


def triggerSim(clientID):
    e = sim.simxSynchronousTrigger(clientID)
    step_status = 'successful' if e == 0 else 'error'
    # print(f'Finished Step {step_status}')


def rotation_velocity(rng):
    ''' Set rotation velocity randomly, rotation velocity is a composition of two sinusoidal velocities '''
    #Sinusoidal velocity
    forward = [-0.3, -0.35, -0.4]
    backward = [0.75, 0.8, 0.85]
    freq = 60
    ts = np.linspace(0, 1000 / freq, 1000)
    velFor = rng.choice(forward) * np.sin(2 * np.pi * 1 / 20 * ts)
    velBack = rng.choice(backward) * np.sin(2 * np.pi * 1 / 10 * ts)
    velSin = velFor
    idxFor = np.argmax(velFor > 0)
    velSin[idxFor:] = velBack[idxFor:]
    velReal = velSin
    return velReal


def start_simulation():
    ''' Function to communicate with Coppelia Remote API and start the simulation '''
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19000, True, True, 5000,
                             5)  # Connect to CoppeliaSim
    if clientID != -1:
        print('Connected to remote API server')
    else:
        print("fail")
        sys.exit()

    returnCode = sim.simxSynchronous(clientID, True)
    returnCode = sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    if returnCode != 0 and returnCode != 1:
        # print("something is wrong")
        print(returnCode)
        exit(0)

    triggerSim(clientID)

    # get the handle for the source container
    res, pour = sim.simxGetObjectHandle(clientID, 'joint',
                                        sim.simx_opmode_blocking)
    res, receive = sim.simxGetObjectHandle(clientID, 'receive',
                                           sim.simx_opmode_blocking)
    # start streaming the data
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, receive, -1, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetJointPosition(
        clientID, pour, sim.simx_opmode_streaming)

    return clientID, pour, receive


def stop_simulation(clientID):
    ''' Function to stop the episode '''
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)


def get_object_handles(clientID, pour):
    # Drop blocks in source container
    triggerSim(clientID)
    number_of_blocks = 2
    # print('Initial number of blocks=', number_of_blocks)
    setNumberOfBlocks(clientID,
                      blocks=number_of_blocks,
                      typeOf='cube',
                      mass=0.002,
                      blockLength=0.025,
                      frictionCube=0.06,
                      frictionCup=0.8)
    triggerSim(clientID)

    # Get handles of cubes created
    object_shapes_handles = []
    obj_type = "Cuboid"
    for obj_idx in range(number_of_blocks):
        res, obj_handle = sim.simxGetObjectHandle(clientID,
                                                  f'{obj_type}{obj_idx}',
                                                  sim.simx_opmode_blocking)
        object_shapes_handles.append(obj_handle)

    triggerSim(clientID)

    for obj_handle in object_shapes_handles:
        # get the starting position of source
        returnCode, obj_position = sim.simxGetObjectPosition(
            clientID, obj_handle, -1, sim.simx_opmode_streaming)

    returnCode, position = sim.simxGetJointPosition(clientID, pour,
                                                    sim.simx_opmode_buffer)
    returnCode, obj_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)
    print(f'Pouring Cup Initial Position:{obj_position}')
    # Give time for the cubes to finish falling
    _wait(clientID)
    return object_shapes_handles, obj_position


def set_cup_initial_position(clientID, pour, receive, cup_position, rng):

    # Move cup along x axis
    global low, high
    move_x = low + (high - low) * rng.random()
    cup_position[0] = cup_position[0] + move_x

    returnCode = sim.simxSetObjectPosition(clientID, pour, -1, cup_position,
                                           sim.simx_opmode_blocking)
    triggerSim(clientID)
    returnCode, pour_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)
    print(f'Pouring Cup Moved Position:{cup_position}')
    returnCode, receive_position = sim.simxGetObjectPosition(
        clientID, receive, -1, sim.simx_opmode_buffer)
    print(f'Receiving Cup Position:{cup_position}')
    return pour_position, receive_position


def get_state(object_shapes_handles, clientID, pour, j):
    ''' Function to get the cubes and pouring cup position '''

    # Get position of the objects
    obj_pos = []
    for obj_handle in object_shapes_handles:
        # get the starting position of source
        returnCode, obj_position = sim.simxGetObjectPosition(
            clientID, obj_handle, -1, sim.simx_opmode_buffer)
        obj_pos.append(obj_position)

    returnCode, cup_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)

    return obj_pos, cup_position


def move_cup(clientID, pour, action, cup_position, center_position):
    ''' Function to move the pouring cup laterally during the rotation '''

    global low, high
    resolution = 0.001
    move_x = resolution * action
    movement = cup_position[0] + move_x
    if center_position + low < movement < center_position + high:
        cup_position[0] = movement
        returnCode = sim.simxSetObjectPosition(clientID, pour, -1,
                                               cup_position,
                                               sim.simx_opmode_blocking)


def rotate_cup(clientID, speed, pour):
    ''' Function to rotate cup '''
    errorCode = sim.simxSetJointTargetVelocity(clientID, pour, speed,
                                               sim.simx_opmode_oneshot)
    returnCode, position = sim.simxGetJointPosition(clientID, pour,
                                                    sim.simx_opmode_buffer)
    return position


def _wait(clientID):
    for _ in range(60):
        triggerSim(clientID)


def calculate_action(cup_position,cube1_position,cube2_position):
    cup_position= round(cup_position[0],1)
    cube1_position =round(cube1_position[0],1)
    cube2_position  = round(cube2_position[0],1)
    state = cup_position

    if np.random.uniform(0,1) <=exp_rate:
        action = np.random.choice(action_space)
    else:
        action = max(qTable_dict[state],key=qTable_dict[state].get)
    return action

def main():

    for i in range(40):
        print("Training epoch ", i)
        rng = np.random.default_rng()
        # Set rotation velocity randomly
        velReal = rotation_velocity(rng)

        # Start simulation
        clientID, pour, receive = start_simulation()
        object_shapes_handles, cup_position = get_object_handles(clientID, pour)

        # Get initial position of the cups
        cup_position, receive_position = set_cup_initial_position(
            clientID, pour, receive, cup_position, rng)
        _wait(clientID)
        center_position = cup_position[0]

    
        position_list = []
        for j in range(velReal.shape[0]):
            # 60HZ
            triggerSim(clientID)
            # Make sure simulation step finishes
            returnCode, pingTime = sim.simxGetPingTime(clientID)

            # Get current state
            cubes_position, cup_position = get_state(object_shapes_handles,
                                                    clientID, pour, j)

            # Rotate cup
            speed = velReal[j]

            position = 1
            # call rotate_cup function and assign the return value to position variable
            # it will be something like position = rotate_cup()
            position = rotate_cup(clientID, speed, pour)
            position_list.append(position)

            # Move cup laterally
            actions = [-2, -1, 0, 1, 2]

            # call move_cup function
            #action= random.choice(actions)
            action = calculate_action(cup_position,cubes_position[0],cubes_position[1])
            state_action_list.append([round(cup_position[0],1),action])
            move_cup(clientID, pour, action, cup_position, center_position)
            if j > 10 and position > 0:
                    break
        # Stop simulation
        stop_simulation(clientID)
        reward = 100 - (abs(cup_position[0] - receive_position[0])*100)
        reward_list.append(reward)
        for s in reversed(state_action_list):
            state,action = s[0],s[1]
            reward = qTable_dict[state][action] + learning_rate*(reward - qTable_dict[state][action])
            qTable_dict[state][action] = round(reward,3)

    
    # reward_list_df = pd.DataFrame(reward_list)
    # np.savetxt(r'./rewards_list.txt', reward_list_df.values, fmt='%.4e')
    
    for i in range(10):
        print("TESTING STARTS")
        # print("Testing epoch ", i)

        rng = np.random.default_rng()
        # Set rotation velocity randomly
        velReal = rotation_velocity(rng)

        # Start simulation
        clientID, pour, receive = start_simulation()
        object_shapes_handles, cup_position = get_object_handles(clientID, pour)

        # Get initial position of the cups
        cup_position, receive_position = set_cup_initial_position(
            clientID, pour, receive, cup_position, rng)
        _wait(clientID)
        center_position = cup_position[0]
        for j in range(velReal.shape[0]):
            # 60HZ
            triggerSim(clientID)
            # Make sure simulation step finishes
            returnCode, pingTime = sim.simxGetPingTime(clientID)

            # Get current state
            cubes_position, cup_position = get_state(object_shapes_handles,
                                                    clientID, pour, j)

            # Rotate cup
            speed = velReal[j]

            position = 1
            # call rotate_cup function and assign the return value to position variable
            # it will be something like position = rotate_cup()
            position = rotate_cup(clientID, speed, pour)
            position_list.append(position)
           

            # Move cup laterally
            actions = [-2, -1, 0, 1, 2]

            # call move_cup function
            #action= random.choice(actions)
            action = calculate_action(cup_position,cubes_position[0],cubes_position[1])
            state_action_list.append([round(cup_position[0],1),action])
            move_cup(clientID, pour, action, cup_position, center_position)
           
            if j > 10 and position > 0:
                    break
        print(f'The Cubes final positions are:{cubes_position}')

        # Stop simulation
        stop_simulation(clientID)
   


    

if __name__ == '__main__':

    main()
    

