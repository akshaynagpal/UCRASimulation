import numpy as np
import simpy
import matplotlib.pyplot as plt
from collections import namedtuple
from utils import load_simulation_config, save_simulation_metrics, generate_task
import algorithms
from ubiquitous_system import UbiquitousSystem

algorithms = [
    ("Greedy_Centralized", algorithms.greedy_centralized),
    ("Greedy_Decentralized", algorithms.greedy_decentralized),
    ("Greedy_Hierarchical", algorithms.greedy_hierarchical),
    ("Auction_Centralized", algorithms.auction_centralized),
    ("Auction_Decentralized", algorithms.auction_decentralized),
    ("Auction_Hierarchical", algorithms.auction_hierarchical)
]

# initialize system parameters and devices
SIMULATION_TIME, task_arrival_rates, devices = load_simulation_config('tests/test_suite.json')

simulation_results = []
for task_arrival_rate in task_arrival_rates:
    # Run simulation for each algorithm
    results = {}
    for name, algorithm in algorithms:
        env = simpy.Environment()
        system = UbiquitousSystem(env, devices, algorithm, task_arrival_rate)
        env.run(until=SIMULATION_TIME)
        
        # Calculate metrics
        throughput = len(system.completed_tasks) / SIMULATION_TIME
        response_times = [end - start for _, start, end in system.completed_tasks]
        avg_response_time = np.mean(response_times) if response_times else 0
        energy_utilization = system.get_energy_utilization()
        
        results[name] = {
            "Throughput": throughput,
            "Average_Response_Time": avg_response_time,
            "Energy_Utilization": energy_utilization
        }
    simulation_results.append({
        "Task_Arrival_Rate": task_arrival_rate,
        "Results": results
    })
    # Print results
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

# Save results to a JSON file
save_simulation_metrics("results/test_result.csv", simulation_results)
