from ultralytics import YOLO
from PIL import Image
from load_model import MODEL_PATH, CLASS_NAMES, DETECT_CLASS

def load_model(model_path):
    return YOLO(model_path)

def detect_objects(image, model, class_name):
    return model.predict(source=image, save=True, classes=[CLASS_NAMES.index(class_name)])

def main():
    model = load_model(MODEL_PATH)
    
    im1 = Image.open("bill-gates.jpg")
    result = detect_objects(im1, model, DETECT_CLASS)

if __name__ == "__main__":
    main()
