import numpy as np
from PIL import Image

def image_to_list(path:str) -> list[list[int]]:
    img = Image.open(path).convert("L")
    pixel_img = np.array(img).tolist()
    
    return pixel_img

def image_resize(image:list[list[int]]):
    height = len(image)
    width = len(image[0])
    new_image = list([[image[j * (height // 28)][i * (width // 28)] for i in range(28)] for j in range(28)])
    
    return new_image

def image_normalization(image:list[list[int]]) -> list[list[float]]:
    normalized_image = list([[image[i][j] / 255 for j in range(28)] for i in range(28)])
    
    return normalized_image

def image_flatten(image:list[list[float]]) -> list[float]:
    flattened_image = list()
    for row in image:
        flattened_image.extend(row)
    
    return flattened_image

def image_convert(path:str) -> list[float]:
    img = image_to_list(path)
    img = image_resize(img)
    img = image_normalization(img)
    img = image_flatten(img)
    
    return img
    

def main():
    image = image_convert("pictures\\sesameLeaf.jpg")
    print(image)

if __name__ == "__main__":
    main()
    