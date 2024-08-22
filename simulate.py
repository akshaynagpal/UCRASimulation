import numpy as np
import simpy
import matplotlib.pyplot as plt
from collections import namedtuple

# Define data structures
Device = namedtuple('Device', ['id', 'processing', 'memory', 'energy', 'load', 'bandwidth'])
Task = namedtuple('Task', ['id', 'processing', 'memory', 'deadline', 'priority', 'data_size'])

# System parameters
SIMULATION_TIME = 100
TASK_ARRIVAL_RATE = 0.2

# Initialize devices and tasks
devices = [
    Device(1, 1000, 512, 10, 20, 100),
    Device(2, 2000, 1024, 20, 30, 1000),
    Device(3, 1500, 768, 15, 25, 500),
    Device(4, 800, 256, 5, 10, 50),
    Device(5, 3000, 2048, 30, 40, 2000)
]

def generate_task(id):
    return Task(id, 
                np.random.randint(100, 1000),
                np.random.randint(32, 256),
                np.random.randint(5, 30),
                np.random.randint(1, 4),
                np.random.randint(2, 20))

# Latency matrix
latency_matrix = np.array([
    [0, 10, 15, 20, 25],
    [10, 0, 12, 18, 22],
    [15, 12, 0, 16, 20],
    [20, 18, 16, 0, 24],
    [25, 22, 20, 24, 0]
])

# Allocation algorithms
def greedy_centralized(devices, tasks):
    allocations = []
    for task in tasks:
        best_device = min(devices, key=lambda d: (task.processing / d.processing) + (task.memory / d.memory))
        allocations.append((task, best_device))
    return allocations

def greedy_decentralized(devices, tasks):
    allocations = []
    device_indices = np.arange(len(devices))  # Create an array of indices for the devices

    for task in tasks:
        chosen_indices = np.random.choice(device_indices, size=min(3, len(device_indices)), replace=False)
        local_devices = [devices[i] for i in chosen_indices] 
        best_device = min(local_devices, key=lambda d: (task.processing / d.processing) + (task.memory / d.memory))
        allocations.append((task, best_device))
    return allocations

def greedy_hierarchical(devices, tasks):
    # Simplified hierarchical approach: divide devices into two clusters
    cluster1, cluster2 = devices[:len(devices)//2], devices[len(devices)//2:]
    allocations = []
    for task in tasks:
        cluster = cluster1 if np.random.rand() < 0.5 else cluster2
        best_device = min(cluster, key=lambda d: (task.processing / d.processing) + (task.memory / d.memory))
        allocations.append((task, best_device))
    return allocations

def auction_centralized(devices, tasks):
    allocations = []
    for task in tasks:
        bids = [(d, (d.processing / task.processing) + (d.memory / task.memory)) for d in devices]
        best_device = max(bids, key=lambda x: x[1])[0]
        allocations.append((task, best_device))
    return allocations

def auction_decentralized(devices, tasks):
    allocations = []
    device_indices = np.arange(len(devices))  # Create an array of indices for the devices

    for task in tasks:
        chosen_indices = np.random.choice(device_indices, size=min(3, len(device_indices)), replace=False)
        local_devices = [devices[i] for i in chosen_indices]
        bids = [(d, (d.processing / task.processing) + (d.memory / task.memory)) for d in local_devices]
        best_device = max(bids, key=lambda x: x[1])[0]
        allocations.append((task, best_device))
    return allocations

def auction_hierarchical(devices, tasks):
    cluster1, cluster2 = devices[:len(devices)//2], devices[len(devices)//2:]
    allocations = []
    for task in tasks:
        cluster = cluster1 if np.random.rand() < 0.5 else cluster2
        bids = [(d, (d.processing / task.processing) + (d.memory / task.memory)) for d in cluster]
        best_device = max(bids, key=lambda x: x[1])[0]
        allocations.append((task, best_device))
    return allocations

# Simulation class
class UbiquitousSystem:
    def __init__(self, env, devices, allocation_func):
        self.env = env
        self.devices = devices
        self.tasks = []
        self.allocation_func = allocation_func
        self.completed_tasks = []
        self.energy_consumption = {d.id: 0 for d in devices}
        self.env.process(self.task_generator())
        self.env.process(self.allocator())

    def task_generator(self):
        task_id = 1
        while True:
            yield self.env.timeout(np.random.exponential(1/TASK_ARRIVAL_RATE))
            new_task = generate_task(task_id)
            self.tasks.append(new_task)
            task_id += 1

    def allocator(self):
        while True:
            yield self.env.timeout(1)  # Allocate every second
            if self.tasks:
                allocations = self.allocation_func(self.devices, self.tasks)
                for task, device in allocations:
                    self.env.process(self.execute_task(task, device))
                self.tasks = []  # Clear allocated tasks

    def execute_task(self, task, device):
        execution_time = task.processing / device.processing
        energy_consumed = (task.processing / device.processing) * device.energy  # Simple energy model
        self.energy_consumption[device.id] += energy_consumed
        yield self.env.timeout(execution_time)
        self.completed_tasks.append((task, self.env.now - execution_time, self.env.now))

    def get_energy_utilization(self):
        total_energy_capacity = sum(d.energy for d in self.devices)
        total_energy_consumed = sum(self.energy_consumption.values())
        return total_energy_consumed / total_energy_capacity

# Run simulation for each algorithm
algorithms = [
    ("Greedy Centralized", greedy_centralized),
    ("Greedy Decentralized", greedy_decentralized),
    ("Greedy Hierarchical", greedy_hierarchical),
    ("Auction Centralized", auction_centralized),
    ("Auction Decentralized", auction_decentralized),
    ("Auction Hierarchical", auction_hierarchical)
]

results = {}

for name, algorithm in algorithms:
    env = simpy.Environment()
    system = UbiquitousSystem(env, devices, algorithm)
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
