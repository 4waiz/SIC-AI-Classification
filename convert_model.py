"""
Run this script to convert the Keras .h5 model to TensorFlow.js format.

Usage:
    pip install tensorflowjs
    python convert_model.py
"""
import subprocess
import sys
import os

def main():
    # Install tensorflowjs if not present
    try:
        import tensorflowjs
    except ImportError:
        print("Installing tensorflowjs...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflowjs"])

    # Create output directory
    output_dir = os.path.join("public", "model")
    os.makedirs(output_dir, exist_ok=True)

    # Convert model
    print("Converting model to TensorFlow.js format...")
    subprocess.run([
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        "final_model.h5",
        output_dir
    ])

    print(f"\nDone! Model saved to {output_dir}/")
    print("You can now deploy to Vercel.")

if __name__ == "__main__":
    main()
