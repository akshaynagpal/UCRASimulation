import numpy as np

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