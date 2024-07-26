import json
import matplotlib.pyplot as plt

# Define the connections between key points
connections = [
    (0, 1),  # Nose to Neck
    (1, 2), (2, 3), (3, 4),  # Neck to Rshoulder to Relbow to Rwrist
    (1, 5), (5, 6), (6, 7),  # Neck to Lshoulder to Lelbow to Lwrist
    (1, 8), (8, 9), (9, 10),  # Neck to Rhip to Rknee to Rankle
    (1, 11), (11, 12), (12, 13),  # Neck to Lhip to Lknee to Lankle
    (0, 14), (0, 15),  # Nose to Reye and Leye
    (14, 16), (15, 17)  # Reye to Rear and Leye to Lear
]

# Define a function to plot stick figures for a single frame
def plot_stick_figure(key_points):
    plt.clf()  # Clear the current figure
    ax = plt.gca()
    
    for person in key_points:
        for connection in connections:
            i, j = connection
            if person[i] is not None and person[j] is not None:
                x = [person[i]['x_coordinate'], person[j]['x_coordinate']]
                y = [person[i]['y_coordinate'], person[j]['y_coordinate']]
                plt.plot(x, y, marker='o')
    
    ax.set_aspect('equal', 'box')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.draw()
    plt.pause(0.1)  # Pause for 1 second

# Define a function to extract key points for each frame
def extract_key_points(data):
    frames_key_points = []
    for frame in data['frames']:
        frame_key_points = []
        for person in frame['people']:
            key_points = [None] * 18  # Assuming there are 18 key points
            for point in person['key_points']:
                descriptor = point['descriptor']
                if 0 <= descriptor < len(key_points):
                    key_points[descriptor] = point
            frame_key_points.append(key_points)
        frames_key_points.append(frame_key_points)
    return frames_key_points

# Load JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract key points for each frame
frames_key_points = extract_key_points(data)

# Plot stick figures for each frame continuously
plt.ion()  # Turn on interactive mode
for key_points in frames_key_points:
    plot_stick_figure(key_points)