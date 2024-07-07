from PIL import Image, ImageEnhance
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as tt
LabelDict = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
    20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
    30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z",
    36: "a", 37: "b", 38: "c", 39: "d", 40: "e", 41: "f", 42: "g", 43: "h", 44: "i", 45: "j",
    46: "k", 47: "l", 48: "m", 49: "n", 50: "o", 51: "p", 52: "q", 53: "r", 54: "s", 55: "t",
    56: "u", 57: "v", 58: "w", 59: "x", 60: "y", 61: "z"
}

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to grayscale
    image = image.convert('L')

    # Invert colors if necessary (assuming dark text on light background)
    if np.mean(image) > 128:
        image = Image.fromarray(255 - np.array(image))

    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Resize and pad to 28x28
    target_size = 28
    ratio = target_size / max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)

    new_image = Image.new('L', (target_size, target_size), 0)
    new_image.paste(image, ((target_size-new_size[0])//2, (target_size-new_size[1])//2))

    # Convert to tensor and normalize
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(mean=0.1736, std=0.3248)
    ])
    img_tensor = transform(new_image)
    return img_tensor.unsqueeze(0)

def predict_character(model, image, device):
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_p, top_class = probabilities.topk(3, dim=1)
    predictions = []
    for i in range(3):
        predictions.append((LabelDict[top_class[0][i].item()], top_p[0][i].item()))
    return predictions

def segment_characters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Filter out noise
            char_image = gray[y:y+h, x:x+w]
            char_images.append(char_image)

    return char_images

def predict_sequence(model, image, device):
    char_images = segment_characters(image)
    predictions = []
    final_predict = []
    for char_image in char_images:
        pil_image = Image.fromarray(char_image)
        char_predictions = predict_character(model, pil_image, device)
        predictions.append(char_predictions)
        final_predict.append(char_predictions[0])
    return predictions,final_predict

def predict_sequence_app(model, image_path,device):
    model = model.to(device)
    image = cv2.imread(image_path)
    char_images = segment_characters(image)
    final_predict = []
    predictions = []
    for char_image in char_images:
        pil_image = Image.fromarray(char_image)
        char_predictions = model.infer(pil_image)
        predictions.append(char_predictions)
        final_predict.append(char_predictions[0])
    return predictions,final_predict

def display_prediction(image_path, predictions, is_single_char=False):
    img = Image.open(image_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if is_single_char:
        preprocessed = preprocess_image(img)
        plt.imshow(preprocessed.squeeze(), cmap='gray')
        plt.title(f"Preprocessed Image\nPredicted: {predictions[0][0]}")
    else:
        char_images = segment_characters(np.array(img))
        resized_images = []
        max_height = max(img.shape[0] for img in char_images)

        for char_img in char_images:
            aspect_ratio = char_img.shape[1] / char_img.shape[0]
            new_width = int(max_height * aspect_ratio)
            resized_img = cv2.resize(char_img, (new_width, max_height))
            resized_images.append(resized_img)

        combined_image = np.hstack(resized_images)
        plt.imshow(combined_image, cmap='gray')
        predicted_text = ''.join([pred[0][0] for pred in predictions])
        plt.title(f"Segmented Characters\nPredicted: {predicted_text}")

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("Top 3 predictions:")
    for i, pred in enumerate(predictions):
        print(f"Character {i+1}:" if not is_single_char else "Predictions:")
        for char, prob in pred:
            print(f"  {char}: {prob:.4f}")
    print(predictions)