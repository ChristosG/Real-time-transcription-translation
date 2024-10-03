import sys
import os
from PIL import Image, ImageEnhance, __version__ as PILLOW_VERSION
import subprocess
import tempfile

def get_resampling_filter():
    """
    Returns the appropriate resampling filter based on Pillow version.
    """
    major, minor, _ = map(int, PILLOW_VERSION.split("."))
    if major >= 10:
        return Image.Resampling.LANCZOS
    else:
        return Image.ANTIALIAS

def resize_and_enhance_image(input_path, output_png_path, 
                             new_size=(192, 192),  
                             enhance_sharpness=2.0, 
                             enhance_contrast=1.5):
    """
    Resizes and enhances a PNG image.
    
    Parameters:
    - input_path (str): Path to the input PNG image.
    - output_png_path (str): Path to save the resized and enhanced PNG image.
    - new_size (tuple): Desired size (width, height) for the resized image.
    - enhance_sharpness (float): Factor to enhance sharpness.
    - enhance_contrast (float): Factor to enhance contrast.
    """
    
    # Check if input file exists
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    try:
        # Open the image
        with Image.open(input_path) as img:
            print(f"Original image size: {img.size}")
            
            # Determine the resampling filter
            resample_filter = get_resampling_filter()
            
            # Resize the image
            resized_img = img.resize(new_size, resample=resample_filter)
            print(f"Resized image to: {resized_img.size}")
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(resized_img)
            enhanced_img = enhancer.enhance(enhance_sharpness)
            print(f"Enhanced sharpness by a factor of {enhance_sharpness}")
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced_img)
            enhanced_img = enhancer.enhance(enhance_contrast)
            print(f"Enhanced contrast by a factor of {enhance_contrast}")
            
            # Save the resized and enhanced PNG
            enhanced_img.save(output_png_path, format='PNG')
            print(f"Saved resized and enhanced image as '{output_png_path}'")
    
    except Exception as e:
        print(f"Error during image processing: {e}")
        sys.exit(1)

def convert_png_to_svg(input_png_path, output_svg_path):
    """
    Converts a PNG image to SVG using Inkscape's command-line interface.
    
    Parameters:
    - input_png_path (str): Path to the processed PNG image.
    - output_svg_path (str): Path to save the converted SVG image.
    """
    
    # Check if Inkscape is installed
    try:
        subprocess.run(['inkscape', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("Error: Inkscape is not installed or not found in PATH.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Inkscape is not installed or not found in PATH.")
        sys.exit(1)
    
    # Construct the Inkscape command
    command = [
        'inkscape',
        input_png_path,
        '--export-type=svg',
        '--export-filename', output_svg_path
    ]
    
    try:
        print("Converting PNG to SVG using Inkscape...")
        subprocess.run(command, check=True)
        print(f"Converted image to SVG and saved as '{output_svg_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Error during SVG conversion: {e}")
        sys.exit(1)

def main():
    # Check for correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Error: Incorrect number of arguments.")
        print_usage()
        sys.exit(1)
    
    # Parse command-line arguments
    input_image_path = sys.argv[1]
    output_svg_path = sys.argv[2]
    
    # Validate input file extension
    if not input_image_path.lower().endswith('.png'):
        print("Error: Input file must be a PNG image.")
        sys.exit(1)
    
    # Validate output file extension
    if not output_svg_path.lower().endswith('.svg'):
        print("Error: Output file must have a .svg extension.")
        sys.exit(1)
    
    # Create a temporary file for the processed PNG
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_png:
        temp_png_path = temp_png.name
    
    try:
        # Step 1: Resize and enhance the image
        resize_and_enhance_image(
            input_path=input_image_path,
            output_png_path=temp_png_path,
            new_size=(192, 192),            
            enhance_sharpness=2.0,         
            enhance_contrast=1.5            
        )
        
        # Step 2: Convert the processed PNG to SVG
        convert_png_to_svg(
            input_png_path=temp_png_path,
            output_svg_path=output_svg_path
        )
    
    finally:
        # Clean up the temporary processed PNG
        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)
            print(f"Deleted temporary file '{temp_png_path}'")

def print_usage():
    print("Usage: python imgfixer.py input.png output.svg")
    print("Example: python imgfixer.py input_image.png output_image.svg")

if __name__ == "__main__":
    main()

