import json
import numpy as np
from collections import namedtuple

Device = namedtuple('Device', ['id', 'processing', 'memory', 'energy'])
Task = namedtuple('Task', ['id', 'processing', 'memory'])

# Function to load test configuration from JSON file
def load_simulation_config(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract simulation parameters
    simulation_time = data['SIMULATION_TIME']
    task_arrival_rate = data['TASK_ARRIVAL_RATE']
    
    # Convert each dictionary in the devices list to a Device namedtuple
    devices = [Device(**device) for device in data['devices']]
    
    return simulation_time, task_arrival_rate, devices


# save results dictionary to json file
def save_simulation_metrics(file_path, file_contents):
    with open(file_path, 'w') as json_file:
        json.dump(file_contents, json_file, indent=4)

def generate_task(id):
    return Task(id, np.random.randint(100, 1000), np.random.randint(32, 256))