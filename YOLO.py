from ultralytics import YOLO

model = YOLO('best.pt') 


model.info()


results = model('pygame_movie.mp4', show=True) #inference on movie
