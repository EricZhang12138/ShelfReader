from ultralytics import YOLO
from ultralytics.engine.results import Results

def main():
    model = YOLO('./models/yolov8m.pt')
    results = model('./images/only_bottle.png')  # return type: list[ultralytics.engine.results.Results]
    coordinates = process_res(results)
    for coor in coordinates:
        print(coor)


def process_res(results: Results):
    coordinates = []          # a list of bounding boxes coordinates 
    for result in results:
        height, width = result.orig_shape
        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].tolist()
            normalized = [
                xyxy[0] / width,   # x1
                xyxy[1] / height,  # y1
                xyxy[2] / width,   # x2
                xyxy[3] / height   # y2
            ]
            coordinates.append(normalized)
        result.show()
    return coordinates 

if __name__ == "__main__":  #only runs when this script is executed directly 
    main()