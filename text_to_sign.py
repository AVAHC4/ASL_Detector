"""
Text to Sign Language Image Converter - Live Display Version
Converts input text to sign language images with animated one-by-one display
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import string
import time

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
IMAGE_DIR = 'Image'  # Directory containing sign language images (A-Z folders)
OUTPUT_DIR = 'Output'  # Directory to save combined outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme - Enhanced with gradient background
BG_COLOR_TOP = (25, 30, 50)  # Dark blue top
BG_COLOR_BOTTOM = (40, 50, 80)  # Lighter blue bottom
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (100, 200, 255)  # Light blue accent
CARD_BG = (45, 55, 85)  # Card background

# Display settings
DISPLAY_DELAY = 3  # seconds between letters

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def create_gradient_background(width, height):
    """Create a gradient background"""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        ratio = y / height
        r = int(BG_COLOR_TOP[0] * (1 - ratio) + BG_COLOR_BOTTOM[0] * ratio)
        g = int(BG_COLOR_TOP[1] * (1 - ratio) + BG_COLOR_BOTTOM[1] * ratio)
        b = int(BG_COLOR_TOP[2] * (1 - ratio) + BG_COLOR_BOTTOM[2] * ratio)
        gradient[y, :] = [b, g, r]
    
    return gradient

def get_sign_image(letter, image_dir='Image'):
    """Load the sign language image for a given letter"""
    letter = letter.upper()
    
    if letter not in string.ascii_uppercase:
        return None
    
    folder_path = os.path.join(image_dir, letter)
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Warning: Folder not found for letter '{letter}'")
        return None
    
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"⚠️  Warning: No images found for letter '{letter}'")
        return None
    
    img_path = os.path.join(folder_path, images[0])
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"⚠️  Warning: Could not load image for letter '{letter}'")
        return None
    
    return img

def resize_image(img, target_height=400):
    """Resize image maintaining aspect ratio"""
    if img is None:
        return None
    
    h, w = img.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    resized = cv2.resize(img, (new_width, target_height))
    return resized

def create_card_with_image(img, letter, card_width, card_height):
    """Create a card containing the sign image with letter label"""
    # Create card background
    card = np.full((card_height, card_width, 3), CARD_BG, dtype=np.uint8)
    
    # Add rounded corner effect (simplified)
    cv2.rectangle(card, (5, 5), (card_width-5, card_height-5), ACCENT_COLOR, 2)
    
    # Resize image to fit card
    img_resized = resize_image(img, target_height=int(card_height * 0.65))
    
    if img_resized is not None:
        # Center image on card
        img_h, img_w = img_resized.shape[:2]
        y_offset = int((card_height - img_h) / 2)
        x_offset = int((card_width - img_w) / 2)
        
        # Place image
        card[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = img_resized
    
    # Add letter label at bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 4
    text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
    text_x = (card_width - text_size[0]) // 2
    text_y = card_height - 30
    
    # Add text shadow
    cv2.putText(card, letter, (text_x+2, text_y+2), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(card, letter, (text_x, text_y), font, font_scale, TEXT_COLOR, thickness)
    
    return card

def text_to_sign_language_live(text):
    """
    Convert text to sign language images with live one-by-one display
    """
    print("\n" + "=" * 70)
    print("🤟 TEXT TO SIGN LANGUAGE CONVERTER - LIVE MODE")
    print("=" * 70)
    print(f"Input Text: '{text}'")
    print("=" * 70)
    
    # Filter only alphabetic characters
    letters = [char.upper() for char in text if char.isalpha()]
    
    if not letters:
        print("❌ No valid letters found in input text!")
        return
    
    print(f"📝 Processing {len(letters)} letters: {' '.join(letters)}")
    print(f"⏱️  Each letter will display for {DISPLAY_DELAY} seconds")
    
    # Get screen resolution for full screen display
    screen_width = 1920
    screen_height = 1080
    
    # Create named window and make it full screen
    window_name = 'Text to Sign Language Converter'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Card dimensions
    card_width = 600
    card_height = 700
    
    # Process each letter
    for idx, letter in enumerate(letters):
        print(f"\n{'='*70}")
        print(f"📌 Displaying letter {idx+1}/{len(letters)}: {letter}")
        
        # Load image for this letter
        img = get_sign_image(letter)
        
        if img is None:
            print(f"❌ Skipped: {letter} (image not found)")
            continue
        
        print(f"✅ Loaded: {letter}")
        
        # Create gradient background
        canvas = create_gradient_background(screen_width, screen_height)
        
        # Add main heading
        heading = "TEXT TO SIGN LANGUAGE CONVERSION"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        text_size = cv2.getTextSize(heading, font, font_scale, thickness)[0]
        text_x = (screen_width - text_size[0]) // 2
        text_y = 80
        
        # Add heading shadow
        cv2.putText(canvas, heading, (text_x+3, text_y+3), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(canvas, heading, (text_x, text_y), font, font_scale, ACCENT_COLOR, thickness)
        
        # Add word being spelled
        word_text = f"Spelling: {text.upper()}"
        font_scale_word = 1.5
        text_size_word = cv2.getTextSize(word_text, font, font_scale_word, 2)[0]
        text_x_word = (screen_width - text_size_word[0]) // 2
        text_y_word = 150
        cv2.putText(canvas, word_text, (text_x_word, text_y_word), font, font_scale_word, TEXT_COLOR, 2)
        
        # Add progress indicator
        progress_text = f"Letter {idx+1} of {len(letters)}"
        text_size_prog = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x_prog = (screen_width - text_size_prog[0]) // 2
        text_y_prog = 200
        cv2.putText(canvas, progress_text, (text_x_prog, text_y_prog), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # Create card with image
        card = create_card_with_image(img, letter, card_width, card_height)
        
        # Center card on canvas
        card_y = (screen_height - card_height) // 2 + 50
        card_x = (screen_width - card_width) // 2
        
        # Place card on canvas
        canvas[card_y:card_y+card_height, card_x:card_x+card_width] = card
        
        # Add decorative elements - progress dots
        dot_y = screen_height - 100
        dot_spacing = 30
        total_dots_width = len(letters) * dot_spacing
        start_x = (screen_width - total_dots_width) // 2
        
        for i in range(len(letters)):
            dot_x = start_x + i * dot_spacing
            if i < idx:
                # Completed letters - filled circle
                cv2.circle(canvas, (dot_x, dot_y), 8, ACCENT_COLOR, -1)
            elif i == idx:
                # Current letter - highlighted
                cv2.circle(canvas, (dot_x, dot_y), 12, ACCENT_COLOR, -1)
                cv2.circle(canvas, (dot_x, dot_y), 14, TEXT_COLOR, 2)
            else:
                # Upcoming letters - outline
                cv2.circle(canvas, (dot_x, dot_y), 8, (100, 100, 100), 2)
        
        # Display
        cv2.imshow(window_name, canvas)
        
        # Wait for delay or key press
        key = cv2.waitKey(DISPLAY_DELAY * 1000)
        
        # Allow 'q' or ESC to quit early
        if key == ord('q') or key == 27:
            print("\n⏹️  Display interrupted by user")
            break
    
    # Show completion screen
    canvas = create_gradient_background(screen_width, screen_height)
    
    completion_text = "CONVERSION COMPLETE! ✓"
    font_scale_complete = 2.5
    text_size_complete = cv2.getTextSize(completion_text, font, font_scale_complete, 4)[0]
    text_x_complete = (screen_width - text_size_complete[0]) // 2
    text_y_complete = screen_height // 2
    
    cv2.putText(canvas, completion_text, (text_x_complete+3, text_y_complete+3), 
                font, font_scale_complete, (0, 0, 0), 4)
    cv2.putText(canvas, completion_text, (text_x_complete, text_y_complete), 
                font, font_scale_complete, (100, 255, 100), 4)
    
    final_text = f"Word: {text.upper()}"
    text_size_final = cv2.getTextSize(final_text, font, 1.5, 2)[0]
    text_x_final = (screen_width - text_size_final[0]) // 2
    text_y_final = text_y_complete + 80
    cv2.putText(canvas, final_text, (text_x_final, text_y_final), 
                font, 1.5, TEXT_COLOR, 2)
    
    instruction_text = "Press any key to close"
    text_size_inst = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x_inst = (screen_width - text_size_inst[0]) // 2
    text_y_inst = text_y_final + 80
    cv2.putText(canvas, instruction_text, (text_x_inst, text_y_inst), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✅ Conversion completed successfully!")
    print("=" * 70)

# ---------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------

def interactive_mode():
    """Run in interactive mode"""
    print("\n" + "=" * 70)
    print("🤟 SIGN LANGUAGE CONVERTER - LIVE INTERACTIVE MODE")
    print("=" * 70)
    print("Convert any text to sign language with animated display!")
    print(f"Each letter displays for {DISPLAY_DELAY} seconds")
    print("Press 'q' or ESC during display to skip to next letter")
    print("Type 'quit' or 'exit' to close the program")
    print("=" * 70)
    
    while True:
        print("\n" + "-" * 70)
        text = input("📝 Enter text to convert (or 'quit' to exit): ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text!")
            continue
        
        text_to_sign_language_live(text)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    # Check if Image directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: '{IMAGE_DIR}' directory not found!")
        print("Please ensure you have the sign language images organized in folders A-Z")
        exit(1)
    
    # Check for at least some images
    available_letters = []
    for letter in string.ascii_uppercase:
        folder_path = os.path.join(IMAGE_DIR, letter)
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                available_letters.append(letter)
    
    print(f"\n✅ Found images for {len(available_letters)} letters: {', '.join(available_letters)}")
    
    if len(available_letters) == 0:
        print("❌ No sign language images found! Please add images to the Image directory.")
        exit(1)
    
    # Run interactive mode
    interactive_mode()