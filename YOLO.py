from ultralytics import YOLO

model = YOLO('best.pt') 

results = model('fishesnstuff.jpg', show=True, line_width=1, save=True ) #inference on movie

# adding visualize = True, to the arguments can be helpful for debugging and inference, and more insight


