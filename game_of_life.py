# Import libraries
import numpy as np
import random
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants of the model
SEED = 1234
SPACE_LENGTH = 20
DENSITY = 0.5
TIME_STEPS = 100
ANIMATION_INTERVAL = 200

# Set random seed
random.seed(SEED)

# Define space randomly with the density defined previously
system = np.array([[int(random.random()<DENSITY) for i in range(SPACE_LENGTH)] for j in range(SPACE_LENGTH)])
save_array = np.zeros([TIME_STEPS+1, SPACE_LENGTH, SPACE_LENGTH])

# Define the figure where to do the animation
fig, ax = plt.subplots()
im = ax.imshow(system, cmap='Greys', interpolation='nearest') # Set up initial plot
frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=16, color='orange')  # Add a text label to display the current frame

# Define survability rule
def survive(array: np.array) -> int:
    #if array.shape != (3,3):
    #    raise Exception("Incorrect shape of array for survivability rule")
    value = np.sum(array) # Sum of the neighbours
    if array[1,1] == 1:
        # Rules when the central cell is alive
        if value in [3, 4]:
            # Cell survives if 2 or 3 neighbours (3 or 4 total sum)
            return 1
    else: 
        # Rules when the central cell is dead
        if value == 3:
            return 1

    return 0 # Cell dies by loneliness or overcrowding or is not born 
    
# Define update function to be able to do an animation
def update(frame):
    global system
    random.seed(SEED+frame)
    save_array[frame, :, :] = system
    new_system = copy.deepcopy(system) # create a copy of the system for synchronous iterations
    wrapsys = np.tile(system, [3,3])[SPACE_LENGTH-1:2*SPACE_LENGTH+1, SPACE_LENGTH-1:2*SPACE_LENGTH+1] # Create a wrapped system for the edges
    for i in range(SPACE_LENGTH):
        for j in range(SPACE_LENGTH):
            new_system[i, j] = survive(wrapsys[i:i+3, j:j+3])
    system = new_system
    im.set_data(system)
    frame_text.set_text('Frame: {} out of {}'.format(frame+1, TIME_STEPS))  # Update the text label
    return im, frame_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, interval=ANIMATION_INTERVAL, repeat=False)

# Show the plot
plt.show()

# Save data
print(f'Saved data has following dimensions: {save_array.shape}')
np.save("data.npy", save_array)