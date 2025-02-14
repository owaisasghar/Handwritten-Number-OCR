# Handwritten Number Detection and Recognition System

![Example Detection](https://via.placeholder.com/800x400.png?text=Detection+Example)

A robust system for detecting and recognizing handwritten numbers from images using:
- **CRAFT** (Character Region Awareness for Text Detection) for text localization
- **TrOCR** (Transformer-based Optical Character Recognition) for text recognition

## Features
- 📷 Image processing with OpenCV and PIL
- 🔍 Text detection using CRAFT model
- 🔤 Text recognition using TrOCR transformer model
- 💾 Results export to CSV and JSON formats
- 🖼️ Visualization of detected regions

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/handwritten-number-detection.git
cd handwritten-number-detection
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
# Base requirements
pip install -r requirements.txt

# Install PyTorch (choose appropriate version for your system)
# CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Command
```bash
python detect_and_recognize.py \
  --trained_model weights/craft_mlt_25k.pth \
  --test_folder data/ \
  --cuda True \
  --text_threshold 0.7 \
  --low_text 0.4 \
  --link_threshold 0.4
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--trained_model` | weights/craft_mlt_25k.pth | Path to trained CRAFT model |
| `--test_folder` | data/ | Directory containing test images |
| `--cuda` | True | Use CUDA for inference |
| `--text_threshold` | 0.7 | Text confidence threshold |
| `--low_text` | 0.4 | Text low-bound score |
| `--link_threshold` | 0.4 | Link confidence threshold |
| `--refine` | False | Enable link refiner |

### Outputs
Results will be saved in:
- `./result/`: Visual detection results
- `quantities_and_prices.csv`: CSV output of detected numbers
- `quantities_and_prices.json`: JSON output of detected numbers

## Customization
To process custom images:
1. Place images in `data/` directory
2. Update model weights in `weights/` directory
3. Adjust threshold parameters as needed

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License
[MIT License](LICENSE)

