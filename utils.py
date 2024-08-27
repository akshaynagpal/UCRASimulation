import json
import numpy as np
import csv
from collections import namedtuple


Device = namedtuple('Device', ['id', 'processing', 'memory', 'energy'])
Task = namedtuple('Task', ['id', 'processing', 'memory'])


# Function to load test configuration from JSON file
def load_simulation_config(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract simulation parameters
    simulation_time = data['simulation_time']
    task_arrival_rates = data['task_arrival_rates']
    
    # Convert each dictionary in the devices list to a Device namedtuple
    devices = [Device(**device) for device in data['devices']]
    
    return simulation_time, task_arrival_rates, devices


def generate_task(id):
    return Task(id, np.random.randint(100, 1000), np.random.randint(32, 256))


# save results dictionary to json file
def save_simulation_metrics(file_path, file_contents):
    csv_header = [
    'Task_Arrival_Rate',
    'Greedy_Centralized_Throughput', 'Greedy_Centralized_Avg_Response_Time', 'Greedy_Centralized_Energy_Utilization',
    'Greedy_Decentralized_Throughput', 'Greedy_Decentralized_Avg_Response_Time', 'Greedy_Decentralized_Energy_Utilization',
    'Greedy_Hierarchical_Throughput', 'Greedy_Hierarchical_Avg_Response_Time', 'Greedy_Hierarchical_Energy_Utilization',
    'Auction_Centralized_Throughput', 'Auction_Centralized_Avg_Response_Time', 'Auction_Centralized_Energy_Utilization',
    'Auction_Decentralized_Throughput', 'Auction_Decentralized_Avg_Response_Time', 'Auction_Decentralized_Energy_Utilization',
    'Auction_Hierarchical_Throughput', 'Auction_Hierarchical_Avg_Response_Time', 'Auction_Hierarchical_Energy_Utilization'
    ]

    with open(file_path, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_header)
        for entry in file_contents:
            row=[
                entry['Task_Arrival_Rate'],
                entry['Results']['Greedy_Centralized']['Throughput'],
                entry['Results']['Greedy_Centralized']['Average_Response_Time'],
                entry['Results']['Greedy_Centralized']['Energy_Utilization'],
                entry['Results']['Greedy_Decentralized']['Throughput'],
                entry['Results']['Greedy_Decentralized']['Average_Response_Time'],
                entry['Results']['Greedy_Decentralized']['Energy_Utilization'],
                entry['Results']['Greedy_Hierarchical']['Throughput'],
                entry['Results']['Greedy_Hierarchical']['Average_Response_Time'],
                entry['Results']['Greedy_Hierarchical']['Energy_Utilization'],
                entry['Results']['Auction_Centralized']['Throughput'],
                entry['Results']['Auction_Centralized']['Average_Response_Time'],
                entry['Results']['Auction_Centralized']['Energy_Utilization'],
                entry['Results']['Auction_Decentralized']['Throughput'],
                entry['Results']['Auction_Decentralized']['Average_Response_Time'],
                entry['Results']['Auction_Decentralized']['Energy_Utilization'],
                entry['Results']['Auction_Hierarchical']['Throughput'],
                entry['Results']['Auction_Hierarchical']['Average_Response_Time'],
                entry['Results']['Auction_Hierarchical']['Energy_Utilization']   
            ]
            csvwriter.writerow(row)

