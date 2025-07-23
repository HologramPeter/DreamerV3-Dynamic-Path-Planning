import numpy as np

#open webots_gym.log to read the log
log_file = open('data/webots_gym.log', 'r')

#read csv file into a numpy array
data = {}
for line in log_file:
    #split the line by comma
    parts = line.strip().split(',')
    if len(parts) < 5:
        continue  # Skip lines that don't have enough data

    key = parts[0]
    average_reward = float(parts[2])
    steps = int(parts[3])
    success = parts[4] == 'True'

    if key not in data:
        data[key] = ([], [], [])  # Initialize with empty lists for rewards, steps, and success

    data[key][0].append(average_reward)
    data[key][1].append(steps)
    data[key][2].append(success)

#close the log file
log_file.close()

#for each key, value pair
#first element is the mean reward, second is the steps, third is success or not
for key, value in data.items():
    #convert
    mean_reward = np.mean(value[0])
    steps = np.mean(value[1])
    #count success over length of np array
    success = np.sum(value[2])/ len(value[2])
    print(f"{key}: Mean Reward: {mean_reward:.2f}, Steps: {steps:.2f}, Success Rate: {success:.2%}")