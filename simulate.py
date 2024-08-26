import numpy as np
import simpy
import matplotlib.pyplot as plt
from collections import namedtuple
from utils import load_simulation_config, save_simulation_metrics, generate_task
import algorithms
from ubiquitous_system import UbiquitousSystem

# initialize system parameters and devices
SIMULATION_TIME, TASK_ARRIVAL_RATE, devices = load_simulation_config('tests/test_1.json')

# Run simulation for each algorithm
algorithms = [
    ("Greedy_Centralized", algorithms.greedy_centralized),
    ("Greedy_Decentralized", algorithms.greedy_decentralized),
    ("Greedy_Hierarchical", algorithms.greedy_hierarchical),
    ("Auction_Centralized", algorithms.auction_centralized),
    ("Auction_Decentralized", algorithms.auction_decentralized),
    ("Auction_Hierarchical", algorithms.auction_hierarchical)
]

results = {}

for name, algorithm in algorithms:
    env = simpy.Environment()
    system = UbiquitousSystem(env, devices, algorithm, TASK_ARRIVAL_RATE)
    env.run(until=SIMULATION_TIME)
    
    # Calculate metrics
    throughput = len(system.completed_tasks) / SIMULATION_TIME
    response_times = [end - start for _, start, end in system.completed_tasks]
    avg_response_time = np.mean(response_times) if response_times else 0
    energy_utilization = system.get_energy_utilization()
    
    results[name] = {
        "Throughput": throughput,
        "Average Response Time": avg_response_time,
        "Energy Utilization": energy_utilization
    }

# Save results to a JSON file
save_simulation_metrics("results/test_1.json", results)

# Visualize results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

names = list(results.keys())
throughputs = [results[name]["Throughput"] for name in names]
response_times = [results[name]["Average Response Time"] for name in names]
energy_utilizations = [results[name]["Energy Utilization"] for name in names]

ax1.bar(names, throughputs)
ax1.set_ylabel("Throughput (tasks/second)")
ax1.set_title("Throughput Comparison")
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax2.bar(names, response_times)
ax2.set_ylabel("Average Response Time (seconds)")
ax2.set_title("Response Time Comparison")
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax3.bar(names, energy_utilizations)
ax3.set_ylabel("Energy Utilization (%)")
ax3.set_title("Energy Utilization Comparison")
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

# Print results
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")