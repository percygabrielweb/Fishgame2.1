import pygame
import random


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
FPS = 240

class Pellet:
    def __init__(self, x, y, x_velocity, scale):
        self.x = x
        self.y = y
        self.scale = scale
        self.x_velocity = x_velocity*(0.02*scale)
        self.image = random.choice(pellet_images)
        self.image = pygame.transform.scale(self.image, (int(PELLET_WIDTH * (scale/2)), int(PELLET_HEIGHT * (scale/2))))  # Adjust the size as needed

    def update(self):
        self.y += self.x_velocity
        return self.y >= HEIGHT  # Return True if pellet is out of bounds

    def draw(self, screen):
        # Similar darkening logic as in the Fish class
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


class GameState:
    def __init__(self):
        self.pellets = []
        self.fishes = []
        self.background = background_image

    def add_fish(self):
        new_fish = Fish(0, HEIGHT // 2,1, random.randint(MIN_SCALE, MAX_SCALE))
        self.fishes.append(new_fish)

    def add_pellet(self):
        new_pellet = Pellet(random.randint(0, WIDTH - PELLET_WIDTH), 0, 5, random.randint(MIN_SCALE, MAX_SCALE))
        self.pellets.append(new_pellet)

    def update(self):
        # Generate new pellets and fishes at random intervals
        rng = random.randint(1, 100)
        if rng < 4: #4% chance of pellet spawn
            self.add_pellet()
        if rng == 42 and len(self.fishes) < 7: #1% chance of fish spawn #limit on 7 fishes at a time
            self.add_fish()

        # Update pellets
        self.pellets[:] = [pellet for pellet in self.pellets if not pellet.update()]

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
        all_entities = self.pellets + self.fishes
        all_entities_sorted = sorted(all_entities, key=lambda entity: entity.scale)

        # Draw sorted entities
        for entity in all_entities_sorted:
            entity.draw(screen)
            
    def get_pellets(self):
        return self.pellets

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
        if frame_counter == 2: # Takes image every 2 frames...
            current_pellets = game_state.get_pellets()
            current_fishes = game_state.get_fishes()

            generate_labels(image_counter + 1, current_pellets, current_fishes, WIDTH, HEIGHT) # generating labels
            image_path = f"dataset/images/{image_counter + 1}.jpg"
            pygame.image.save(screen, image_path)
            image_counter += 1
            frame_counter = 0

if __name__ == "__main__":
    main()
    