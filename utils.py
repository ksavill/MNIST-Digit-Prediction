import torchvision.transforms as transforms
from PIL import Image, ImageChops

def center_image(image):
    # Invert the image so that the drawn digit (black) becomes white on a black background.
    # This makes it easier to compute the bounding box of the digit.
    inverted = ImageChops.invert(image)
    bbox = inverted.getbbox()
    if bbox:
        # Crop the digit from the original image using the bounding box
        cropped = image.crop(bbox)
        # Create a new white image with the same size as the original image.
        new_im = Image.new('L', image.size, 255)
        # Compute coordinates to paste the cropped digit at the center.
        cx = (image.width - cropped.width) // 2
        cy = (image.height - cropped.height) // 2
        new_im.paste(cropped, (cx, cy))
        return new_im
    else:
        return image

def preprocess_image(image):
    # First center the drawn digit.
    centered = center_image(image)
    # Invert to match MNIST convention (white digit on black background).
    inverted = ImageChops.invert(centered)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(inverted).unsqueeze(0)  # Add batch dimension
    return img_tensor
