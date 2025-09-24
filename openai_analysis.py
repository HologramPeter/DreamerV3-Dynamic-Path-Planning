import numpy as np

#open webots_gym.log to read the log
log_file = open('data_final/webots_gym.log', 'r')


# "{name},{run_ind},{steps},{total_reward},{smoothness},{heading_deviation},{success}"

#read csv file into a numpy array
data = {}
for line in log_file:
    #split the line by comma
    parts = line.strip().split(',')
    if len(parts) < 7:
        continue  # Skip lines that don't have enough data

    key = parts[0]
    run_index = int(parts[1])
    steps = int(parts[2])
    total_reward = float(parts[3])
    smoothness = float(parts[4])
    heading_deviation = float(parts[5])
    success = int(parts[6])

    average_reward = total_reward / steps if steps > 0 else 0

    if key not in data:
        data[key] = ([], [], [], [], [], [])  # Initialize with empty lists for rewards, steps, and success

    data[key][0].append(steps)
    data[key][1].append(average_reward)
    data[key][2].append(success)
    data[key][3].append(smoothness)
    data[key][4].append(heading_deviation)

#close the log file
log_file.close()

#for each key, value pair
#first element is the mean reward, second is the steps, third is success or not
print(f"Model, Avg Reward, Steps, Success, Smooth, Deviation")
for key, value in data.items():
    #convert
    steps = np.mean(value[0])
    mean_reward = np.mean(value[1])
    #count success over length of np array
    success = np.sum(value[2])/ len(value[2])
    smoothness = np.mean(value[3])
    heading_deviation = np.mean(value[4])
    print(f"{key}, {mean_reward:.2f}, {steps:.2f}, {success:.2%}, {smoothness:.2f}, {heading_deviation:.2f}")