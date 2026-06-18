import matplotlib.pyplot as plt


def plot_curve(steps, rewards, title, output_path):
    plt.figure(figsize=(7, 4))
    plt.plot(steps, rewards, marker="o")
    plt.xlabel("Environment Steps")
    plt.ylabel("Evaluation Reward")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


task1_steps = [0, 10000, 30000, 50000]
task1_rewards = [10, 200, 450, 500]

task2_steps = [100000, 200000, 500000, 700000, 900000, 1000000]
task2_rewards = [-17.33, -15.0, 5.0, 8.0, 14.0, 16.7]

task3_steps = [600000, 1000000, 1500000, 1900000, 2000000, 2500000]
task3_rewards = [7.0, 15.8, 15.6, 19.2, 18.0, 15.0]


plot_curve(task1_steps, task1_rewards, "Task 1 Training Curve", "task1_curve.png")
plot_curve(task2_steps, task2_rewards, "Task 2 Training Curve", "task2_curve.png")
plot_curve(task3_steps, task3_rewards, "Task 3 Training Curve", "task3_curve.png")

print("Saved task1_curve.png, task2_curve.png, task3_curve.png")