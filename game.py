import pygame
import random
import math
import csv


# Load images
pellet_images = [pygame.image.load(f"assets/pellets/pellet{i}.png") for i in range(1, 7)]
fish_images = [pygame.image.load(f"assets/fishes/salmon{i}.png") for i in range(1, 4)]
background_image = pygame.image.load("assets/underwater.png")
def draw_background(background, screen):
    screen.blit(background, (0,0))
    

# Constants
WIDTH, HEIGHT = 640, 500
PELLET_WIDTH = 1
PELLET_HEIGHT = 1
FISH_SIZE = 43
MIN_SCALE, MAX_SCALE = 10, 30
FPS = 240 # 60 should be default...this is just a cap
IMAGING_LABELING_FREQUENCY = 2 # saves 1 image and frame per second if at 60, saves 60 per second if = 1
NEXT_ID = 1 # ID for each object
TRACKING_DATA = {} # dictionary for the x and y pos of all objects

def get_NEXT_ID():
    global NEXT_ID
    result = NEXT_ID
    NEXT_ID += 1
    return result
def update_TRACKING_DATA(entity, label):
    if entity.id not in TRACKING_DATA:
        TRACKING_DATA[entity.id] = {'positions': [], 'label': label}
    TRACKING_DATA[entity.id]['positions'].append((entity.x, entity.y))

class Pellet:
    def __init__(self, x, y, y_velocity, scale):
        self.label = 0
        self.id = get_NEXT_ID()
        self.frame = 0
        self.x = x
        self.y = y
        self.scale = scale
        self.y_velocity = y_velocity*(0.02*scale)
        self.image = random.choice(pellet_images)
        self.image = pygame.transform.scale(self.image, (int(PELLET_WIDTH * (scale/2)), int(PELLET_HEIGHT * (scale/2))))  # Adjust the size as needed
        self.x_direction = 1

    def update(self):
        if self.frame > 10 or self.frame < 0: # swings every 30 frames
            self.x_direction = -self.x_direction
        self.x += self.x_direction # swings left or right every 30 frames
        self.y += self.y_velocity
        self.frame += self.x_direction
        
        return self.y >= HEIGHT  # Return True if pellet is out of bounds

    def draw(self, screen):
        # Similar darkening logic as in the Fish class
        darkness_factor = (1 - self.scale / MAX_SCALE) * 120 # adjust factor as needed

        dark_surface = pygame.Surface((int(PELLET_WIDTH * (self.scale/2)), int(PELLET_HEIGHT * (self.scale/2))), flags=pygame.SRCALPHA)
        dark_surface.fill((darkness_factor, darkness_factor, darkness_factor, 0))

        darkened_image = self.image.copy()
        darkened_image.blit(dark_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        screen.blit(darkened_image, (self.x, self.y))
        
class Debris:
    def __init__(self, x, y, x_velocity, y_velocity, scale, vertex):
        self.label = 1
        self.id = get_NEXT_ID()
        self.x_initial = x  # Store the initial x position to oscillate around it
        self.y = y
        self.scale = scale
        self.x_velocity = x_velocity*0.01  # This will determine the speed of the oscillation
        self.y_velocity = y_velocity * (0.01 * scale)  # Base vertical velocity adjusted per frame
        self.vertex = vertex 
        self.frame = 0
        self.image = random.choice(pellet_images)  # It will have the same images as pellets
        self.image = pygame.transform.scale(self.image, (int(PELLET_WIDTH * (scale/2)), int(PELLET_HEIGHT * (scale/2))))
        self.direction = 1

    def update(self):
        # Oscillation pattern
        oscillation_range = self.vertex  # Adjust the range of oscillation based on vertex
        if self.frame > 60:
            self.direction = -self.direction
        self.frame += 1*self.direction

        # Update x using a sine wave pattern for back and forth movement
        self.x = self.x_initial + math.sin(self.frame * self.x_velocity) * oscillation_range

        # Update y position based on vertical velocity
        self.y += self.y_velocity
        
        return self.y >= HEIGHT  # Returns True if debris reached the bottom


    def draw(self, screen):
        darkness_factor = (1 - self.scale / MAX_SCALE) * 120 # adjust factor as needed

        dark_surface = pygame.Surface((int(PELLET_WIDTH * (self.scale/2)), int(PELLET_HEIGHT * (self.scale/2))), flags=pygame.SRCALPHA)
        dark_surface.fill((darkness_factor, darkness_factor, darkness_factor, 0))

        darkened_image = self.image.copy()
        darkened_image.blit(dark_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        screen.blit(darkened_image, (self.x, self.y))

class Fish:
    def __init__(self, x, y, x_velocity, scale, y_velocity=1, direction=1, hunger_level = 4):
        self.x = x
        self.y = y
        self.scale = scale
        self.size = int(FISH_SIZE*scale/10)
        self.image = random.choice(fish_images)
        self.image = pygame.transform.scale(self.image, (self.size, int(self.size*0.36)))  # Adjust the size as needed
        self.x_velocity = x_velocity*0.1*scale
        self.y_velocity = y_velocity*0.05*scale
        self.direction = direction # direction =1 for right, and -1 for left
        self.hunger = hunger_level

    def draw(self, screen):
        # Calculate the darkness factor based on scale (example calculation)
        darkness_factor = (1 - self.scale / MAX_SCALE) * 125  # Adjust factor as needed

        # dark surface for the overlay
        dark_surface = pygame.Surface((self.size, int(self.size * 0.36)), flags=pygame.SRCALPHA)
        dark_surface.fill((darkness_factor, darkness_factor, darkness_factor, 0))

        # copy of the fish image and apply the darkening effect
        darkened_image = self.image.copy()
        darkened_image.blit(dark_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        # Drawing the darkened image on the screen
        screen.blit(darkened_image, (self.x, self.y))
        
    def update(self, pellets):
        # Randomly change direction less frequently
        if random.randint(1, 1800) == 1 and self.hunger > 0:  # Chance to change direction approximately once every 10 seconds
            self.x_velocity = -self.x_velocity
            self.direction = -self.direction
            self.image = pygame.transform.flip(self.image, True, False)

        # Update position with current velocity
        self.x += self.x_velocity
        self.y += self.y_velocity

        # Check for screen boundaries for horizontal movement
        if self.x < 0 and self.hunger > 0 or self.x > WIDTH - self.size and self.hunger > 0:
            self.x_velocity = -self.x_velocity
            self.direction = -self.direction
            self.image = pygame.transform.flip(self.image, True, False)

        # Check for screen boundaries for vertical movement and add some randomness
        if self.y < 30 or self.y > HEIGHT - self.size or random.randint(1, 500) == 1:
            self.y_velocity = -self.y_velocity

        # Eat pellets
        self.eat_pellet(pellets)
        
    def eat_pellet(self, pellets):
        for pellet in pellets[:]:  # Copy the list to safely remove elements
            if self.collides_with_pellet(pellet) and abs(self.scale - pellet.scale) < 5:
                if random.randint(1, 10) == 10:  # 10% chance to eat the pellet for every collision frame
                    pellets.remove(pellet)
                    self.hunger -= 1  # Decrease hunger level if you're tracking it
                    break  # Stop checking after eating a pellet

    def collides_with_pellet(self, pellet):
        # Simple bounding box collision detection
        fish_rect = pygame.Rect(self.x, self.y, self.size, self.size)
        pellet_rect = pygame.Rect(pellet.x, pellet.y, PELLET_WIDTH, PELLET_HEIGHT)  # Assuming 20x20 is pellet size
        return fish_rect.colliderect(pellet_rect)
    
def generate_labels(image_number, pellets, fishes, width, height):
    label_path = f"dataset/labels/{image_number}.txt"
    with open(label_path, 'w') as file:
        for pellet in pellets:
            if not any(fish.collides_with_pellet(pellet) and fish.scale > pellet.scale for fish in fishes):
                # Calculate normalized coordinates
                x_center = (pellet.x + pellet.scale / 2 / 2) / width
                y_center = (pellet.y + pellet.scale / 2 / 2) / height
                box_width = pellet.scale / width
                box_height = pellet.scale / height
                # Write to file (class_id x_center y_center width height)
                file.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
        for fish in fishes:
            # Calculate the normalized coordinates for the bounding box
            x_center = (fish.x + fish.size / 2) / width  # Center x-coordinate of the fish
            y_center = (fish.y + fish.size * 0.36 / 2) / height  # Center y-coordinate of the fish
            box_width = fish.size / width  # Width of the fish (normalized)
            box_height = (fish.size * 0.36) / height  # Height of the fish (normalized)
            # Write to file (class_id x_center y_center width height)
            file.write(f"1 {x_center} {y_center} {box_width} {box_height}\n")

def export_tracking_data_to_csv():
    with open('tracking_data_labeled.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'label', 'positions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for obj_id, data in TRACKING_DATA.items():
            label = data['label']  # Correctly access the label
            positions_formatted = data['positions']  # Access positions directly
            # Convert positions list of tuples into a string format if needed
            positions_str = ";".join([f"({x},{y})" for x, y in positions_formatted])
            writer.writerow({'id': obj_id, 'label': label, 'positions': positions_str})



class GameState:
    def __init__(self):
        self.pellets = []
        self.fishes = []
        self.debrises = []
        self.background = background_image

    def add_fish(self):
        new_fish = Fish(0, HEIGHT // 2,1, random.randint(MIN_SCALE, MAX_SCALE))
        self.fishes.append(new_fish)

    def add_pellet(self):
        new_pellet = Pellet(random.randint(0, WIDTH - PELLET_WIDTH), 0, 5, random.randint(MIN_SCALE, MAX_SCALE))
        self.pellets.append(new_pellet)
    
    def add_debris(self):
        new_debris = Debris(random.randint(0, WIDTH - PELLET_WIDTH), 0, 3, 3, random.randint(MIN_SCALE, MAX_SCALE), random.randint(40,100))
        self.debrises.append(new_debris)
        # Poop falls from the top, not from the fishes...TODO this can be changed 

    def update(self):
        # Generate new pellets and fishes at random intervals
        rng = random.randint(1, 100)
        if rng < 4: #4% chance of pellet spawn
            self.add_pellet()
        if rng == 42 and len(self.fishes) < 7: #1% chance of fish spawn #limit on 7 fishes at a time
            self.add_fish()
        if rng > 95:
            self.add_debris()

        # Update pellets
        self.pellets[:] = [pellet for pellet in self.pellets if not pellet.update()]
        self.debrises[:] = [debris for debris in self.debrises if not debris.update()]

        # Update fishes
        for fish in self.fishes:
            fish.update(self.pellets)
            fish.eat_pellet(self.pellets)
            if fish.x > WIDTH or fish.x < -50 :
                self.fishes.remove(fish)

    def draw(self, screen):
        # Draw background and all game objects
        screen.blit(self.background, (0, 0))
        # Combine and sort all entities by their scale (smaller scale first) so that some objects appear closer
        all_entities = self.pellets + self.fishes + self.debrises
        all_entities_sorted = sorted(all_entities, key=lambda entity: entity.scale)

        # Draw sorted entities
        for entity in all_entities_sorted:
            entity.draw(screen)
            
    def get_pellets(self):
        return self.pellets
    
    def get_debrises(self):
        return self.debrises

    def get_fishes(self):
        return self.fishes
            
def main():
    image_counter = 0
    frame_counter = 0
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fish Simulation 2.0")
    clock = pygame.time.Clock()
    game_state = GameState()
    running = True
    while running and image_counter < 1001: # stop after 1000 images are generated
        clock.tick(FPS)  # Adjust the fps of everything 60 is the base
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        game_state.update()
        game_state.draw(screen)
        
        frame_counter += 1
        pygame.display.flip()
        if frame_counter == IMAGING_LABELING_FREQUENCY: # Takes image every ___ frames frames...(1 for frequent saving, 60 for slow)
            current_pellets = game_state.get_pellets()
            current_fishes = game_state.get_fishes()
            current_debrises = game_state.get_debrises()

            generate_labels(image_counter + 1, current_pellets, current_fishes, WIDTH, HEIGHT) # generating labels
            
            for pellet in current_pellets:
                update_TRACKING_DATA(pellet, 0)  # Assuming 0 is the label for pellets
            for debris in current_debrises:
                update_TRACKING_DATA(debris, 1)
                
            image_path = f"dataset/images/{image_counter + 1}.jpg"
            pygame.image.save(screen, image_path)
            image_counter += 1
            frame_counter = 0
    export_tracking_data_to_csv()
    print(TRACKING_DATA)

if __name__ == "__main__":
    main()
    