import pygame
import numpy as np
import sys
import math
import random
import cv2  # OpenCV for camera input
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# ------------------------------
# Configuration Parameters
# ------------------------------

# Screen dimensions
WIDTH, HEIGHT = 1600, 900  # Larger window for better visibility
BACKGROUND_COLOR = (10, 10, 10)  # Dark background

# Reservoir parameters
NUM_RESERVOIR_NEURONS = 50
DENSITY = 0.1  # Connection density
WEIGHT_SCALE = 0.5  # Scale for initial reservoir weights
LEAKY_RATE = 0.3  # Leaky integrate rate

# Clustering parameters
NUM_CLUSTERS = 3  # Initial number of clusters
CLUSTER_BUFFER_SIZE = 200  # Number of reservoir states to collect before clustering
SILHOUETTE_THRESHOLD = 0.9  # Threshold below which to consider adding a new cluster

# Time-series data parameters
DATA_FREQUENCY = 10  # Hz (Adjust based on camera frame rate)
TIME_STEP = 1.0 / DATA_FREQUENCY  # Seconds per simulation step

# Colors for clusters (extend as needed)
CLUSTER_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
]

# Visualization parameters
NEURON_RADIUS = 15
MAX_V = 1.0  # Maximum membrane potential for color scaling

# Cluster history visualization parameters
CLUSTER_HISTORY_LENGTH = 100  # Number of recent cluster assignments to display
CLUSTER_HISTORY_PIXEL_SIZE = 5  # Size of each cluster history pixel

# Font
pygame.font.init()
FONT = pygame.font.SysFont(None, 24)

# ------------------------------
# Initialize Pygame
# ------------------------------

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Liquid Neural Network Unsupervised Classification")
clock = pygame.time.Clock()

# Define fps globally to ensure accessibility within main
fps = DATA_FREQUENCY  # Frames per second for visualization

# ------------------------------
# Define Neuron Positions
# ------------------------------

def generate_neuron_positions(num_reservoir, width, height):
    positions = []
    margin_x = 150
    margin_y = 100
    reservoir_radius = 300

    # Arrange reservoir neurons in a circle
    center_x, center_y = margin_x, height // 2
    for i in range(num_reservoir):
        angle = 2 * math.pi * i / num_reservoir
        x = center_x + reservoir_radius * math.cos(angle)
        y = center_y + reservoir_radius * math.sin(angle)
        positions.append((int(x), int(y)))

    return positions

neuron_positions = generate_neuron_positions(NUM_RESERVOIR_NEURONS, WIDTH, HEIGHT)
RESERVOIR_START = 0

# ------------------------------
# Define Connectivity Matrix
# ------------------------------

# Initialize reservoir adjacency matrix with random weights
reservoir_adj_matrix = np.zeros((NUM_RESERVOIR_NEURONS, NUM_RESERVOIR_NEURONS))
for i in range(NUM_RESERVOIR_NEURONS):
    for j in range(NUM_RESERVOIR_NEURONS):
        if i != j and random.random() < DENSITY:
            reservoir_adj_matrix[j, i] = np.random.uniform(-WEIGHT_SCALE, WEIGHT_SCALE)

# ------------------------------
# Initialize Neuron States
# ------------------------------

V = np.zeros(NUM_RESERVOIR_NEURONS)  # Membrane potentials
V.fill(0.0)

# ------------------------------
# Synaptic Activation Function
# ------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------------------------------
# Time-Series Data Generator
# ------------------------------

def process_frame(frame):
    """
    Processes a camera frame to extract a meaningful scalar value.
    For demonstration, we'll use the average brightness of the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    avg_brightness = np.mean(gray) / 255.0  # Normalize between 0 and 1
    return avg_brightness * 2 - 1  # Scale to range [-1, 1]

# ------------------------------
# Initialize OpenCV Camera
# ------------------------------

def initialize_camera():
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()
    return cap

# ------------------------------
# Main Simulation Loop
# ------------------------------

def main():
    global NUM_CLUSTERS  # Declare that we intend to modify the global variable
    running = True
    simulation_time = 0.0
    data_buffer = []
    reservoir_states_buffer = []
    cluster_history = []  # To store recent cluster assignments
    current_cluster_label = None
    kmeans_model = None  # Initialize KMeans model

    # Initialize camera
    cap = initialize_camera()

    print("Camera initialized successfully.")
    print("Unsupervised Classification Mode.")
    print(f"Collecting {CLUSTER_BUFFER_SIZE} samples for clustering...")

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process the frame to extract a scalar value or feature vector
        input_signal = process_frame(frame)
        data_buffer.append(input_signal)

        # Update reservoir neurons
        I_ext = input_signal  # External input to all reservoir neurons can be scaled as needed
        for i in range(NUM_RESERVOIR_NEURONS):
            # Simple leaky integrate model
            # dV = (-V + I_ext + sum_j(W_ij * V_j)) * leak_rate * dt
            synaptic_input = np.dot(reservoir_adj_matrix[i], V)
            dV = (-V[i] + I_ext + synaptic_input) * LEAKY_RATE * TIME_STEP
            V[i] += dV

        # Collect reservoir states for clustering
        reservoir_states_buffer.append(V.copy())

        # Perform clustering once buffer is filled
        if len(reservoir_states_buffer) >= CLUSTER_BUFFER_SIZE and kmeans_model is None:
            try:
                print(f"Clustering {CLUSTER_BUFFER_SIZE} reservoir states into {NUM_CLUSTERS} clusters...")
                X = np.array(reservoir_states_buffer)  # Shape: (samples, reservoir_size)
                kmeans_model = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=CLUSTER_BUFFER_SIZE, random_state=42)
                kmeans_model.fit(X)
                print("Clustering completed.")
                print("Cluster centers:", kmeans_model.cluster_centers_)
                # Optionally, clear the buffer if no longer needed
                # reservoir_states_buffer.clear()
            except Exception as e:
                print(f"Clustering failed: {e}")

        # Assign cluster label to current state
        if kmeans_model is not None:
            current_cluster_label = kmeans_model.predict([V])[0]
            cluster_history.append(current_cluster_label)
            if len(cluster_history) > CLUSTER_HISTORY_LENGTH:
                cluster_history.pop(0)

            # Check if the current data point is far from any cluster center
            distances = kmeans_model.transform([V])[0]  # Distance to each cluster
            min_distance = np.min(distances)
            if min_distance > SILHOUETTE_THRESHOLD:
                print(f"Detected outlier with distance {min_distance}. Adding a new cluster.")
                NUM_CLUSTERS += 1
                # Reinitialize and retrain MiniBatchKMeans with the new number of clusters
                try:
                    print(f"Re-clustering with {NUM_CLUSTERS} clusters...")
                    # Refit on the accumulated reservoir states
                    kmeans_model = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=CLUSTER_BUFFER_SIZE, random_state=42)
                    kmeans_model.fit(X)
                    print("Re-clustering completed.")
                except Exception as e:
                    print(f"Re-clustering failed: {e}")
        else:
            current_cluster_label = None

        # Visualization
        screen.fill(BACKGROUND_COLOR)

        # Draw reservoir connections
        for i in range(NUM_RESERVOIR_NEURONS):
            for j in range(NUM_RESERVOIR_NEURONS):
                weight = reservoir_adj_matrix[i, j]
                if weight != 0:
                    start_pos = neuron_positions[i]
                    end_pos = neuron_positions[j]
                    # Color intensity based on weight
                    color_intensity = int(128 + 127 * (weight / WEIGHT_SCALE))  # Normalize between 1 and 255
                    color_intensity = max(0, min(color_intensity, 255))
                    connection_color = (color_intensity, color_intensity, color_intensity)
                    # Thickness based on absolute weight
                    thickness = max(1, int(abs(weight) * 2))
                    pygame.draw.line(screen, connection_color, start_pos, end_pos, thickness)

        # Draw reservoir neurons
        for i in range(NUM_RESERVOIR_NEURONS):
            pos = neuron_positions[i]
            # Color based on membrane potential
            norm_V = (V[i] + MAX_V) / (2 * MAX_V)  # Normalize between 0 and 1
            norm_V = max(0, min(norm_V, 1))
            color_intensity = int(255 * norm_V)
            color = (color_intensity, 255 - color_intensity, 0)  # From green to red
            pygame.draw.circle(screen, color, pos, NEURON_RADIUS)
            pygame.draw.circle(screen, (255, 255, 255), pos, NEURON_RADIUS, 1)  # White border

        # Draw current cluster information
        if current_cluster_label is not None:
            cluster_text = f"Current Cluster: {current_cluster_label}"
            # Ensure the cluster index does not exceed the defined colors
            cluster_color = CLUSTER_COLORS[current_cluster_label % len(CLUSTER_COLORS)]
            cluster_surface = FONT.render(cluster_text, True, cluster_color)
            screen.blit(cluster_surface, (10, 10))
        else:
            cluster_text = "Current Cluster: N/A"
            cluster_surface = FONT.render(cluster_text, True, (255, 255, 255))
            screen.blit(cluster_surface, (10, 10))

        # Display number of clusters
        if kmeans_model is not None:
            num_clusters_text = f"Number of Clusters: {kmeans_model.n_clusters}"
            num_clusters_surface = FONT.render(num_clusters_text, True, (255, 255, 255))
            screen.blit(num_clusters_surface, (10, 40))
        else:
            num_clusters_text = "Number of Clusters: Not Clustered Yet"
            num_clusters_surface = FONT.render(num_clusters_text, True, (255, 255, 255))
            screen.blit(num_clusters_surface, (10, 40))

        # Draw input signal (time-series plot)
        plot_width = WIDTH - 2 * 150
        plot_height = 100
        plot_x = 150
        plot_y = HEIGHT - 150
        pygame.draw.rect(screen, (50, 50, 50), (plot_x, plot_y, plot_width, plot_height))
        if len(data_buffer) > plot_width:
            data_buffer.pop(0)
        for idx in range(1, len(data_buffer)):
            x1 = plot_x + idx - 1
            y1 = plot_y + plot_height // 2 - int(data_buffer[idx - 1] * (plot_height // 2 - 10))
            x2 = plot_x + idx
            y2 = plot_y + plot_height // 2 - int(data_buffer[idx] * (plot_height // 2 - 10))
            pygame.draw.line(screen, (255, 255, 0), (x1, y1), (x2, y2), 2)

        # Draw cluster history as a strip of colored pixels
        history_x = 150
        history_y = HEIGHT - 130  # Position above the input signal plot
        history_width = CLUSTER_HISTORY_LENGTH * CLUSTER_HISTORY_PIXEL_SIZE
        history_height = CLUSTER_HISTORY_PIXEL_SIZE

        # Draw background for cluster history
        pygame.draw.rect(screen, (50, 50, 50), (history_x, history_y, history_width, history_height))

        # Draw cluster history pixels
        for idx, cluster_idx in enumerate(cluster_history):
            color = CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)]
            pixel_rect = pygame.Rect(
                history_x + idx * CLUSTER_HISTORY_PIXEL_SIZE,
                history_y,
                CLUSTER_HISTORY_PIXEL_SIZE,
                CLUSTER_HISTORY_PIXEL_SIZE
            )
            pygame.draw.rect(screen, color, pixel_rect)

        # Optionally, display the camera feed within Pygame
        # Convert the captured frame (OpenCV uses BGR) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to fit within the Pygame window
        frame_rgb = cv2.resize(frame_rgb, (320, 240))
        # Convert to Pygame surface
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        # Blit the frame onto the screen
        screen.blit(frame_surface, (WIDTH - 350, 20))  # Position at top-right corner
        # Draw a border around the camera feed
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH - 350, 20, 320, 240), 2)

        # Instructions for unsupervised classification
        instructions = "Unsupervised Mode: Clusters are formed automatically."
        instructions_surface = FONT.render(instructions, True, (255, 255, 255))
        screen.blit(instructions_surface, (10, 70))

        # Instructions for cluster history visualization
        history_instructions = "Cluster History:"
        history_instructions_surface = FONT.render(history_instructions, True, (255, 255, 255))
        screen.blit(history_instructions_surface, (history_x, history_y - 20))

        # Update the display
        pygame.display.flip()

        # Increment simulation time
        simulation_time += TIME_STEP

        # Cap the frame rate
        clock.tick(fps)

# ------------------------------
# Run the Simulation
# ------------------------------
try:
    main()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release the camera
    if 'cap' in locals():
        cap.release()
        print("Camera released.")
    pygame.quit()
    sys.exit()

# Ensure the main execution block is outside of any function
if __name__ == "__main__":
    main()
