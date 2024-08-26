import numpy as np
from utils import generate_task

# Simulation class
class UbiquitousSystem:
    def __init__(self, env, devices, allocation_func, task_arrival_rate):
        self.env = env
        self.devices = devices
        self.tasks = []
        self.allocation_func = allocation_func
        self.completed_tasks = []
        self.energy_consumption = {d.id: 0 for d in devices}
        self.task_arrival_rate = task_arrival_rate
        self.env.process(self.task_generator())
        self.env.process(self.allocator())

    def task_generator(self):
        task_id = 1
        while True:
            yield self.env.timeout(np.random.exponential(1/self.task_arrival_rate))
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
