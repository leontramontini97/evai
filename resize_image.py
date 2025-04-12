from PIL import Image

def resize_image():
    # Open the image
    image_path = 'LogoMercy2.png'
    img = Image.open(image_path)
    
    # Resize the image to 546x546 pixels with antialiasing
    resized_img = img.resize((546, 546), Image.Resampling.LANCZOS)
    
    # Create a new white image of 640x640
    final_size = (640, 640)
    background = Image.new('RGBA', final_size, (255, 255, 255, 255))
    
    # Calculate position to paste (to center the image)
    paste_x = (final_size[0] - 546) // 2  # Center horizontally
    paste_y = (final_size[1] - 546) // 2  # Center vertically
    
    # Paste the resized image onto the white background
    background.paste(resized_img, (paste_x, paste_y))
    
    # Save the final image
    output_path = 'LogoMercy2_resized.png'
    background.save(output_path)
    print(f"Image has been resized to 640x640 with white borders and saved as {output_path}")

if __name__ == "__main__":
    resize_image()
