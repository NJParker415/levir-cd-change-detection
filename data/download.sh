set -e

DATA_DIR="data/raw"
GDRIVE_ID="1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim"
OUTPUT_ZIP="${DATA_DIR}/LEVIR-CD.zip"

mkdir -p "$DATA_DIR"

# Check if already downloaded
if [ -d "${DATA_DIR}/LEVIR-CD" ]; then
    echo "LEVIR-CD already exists at ${DATA_DIR}/LEVIR-CD, skipping download."
    exit 0
fi

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Install it with: pip install gdown"
    exit 1
fi

echo "Downloading LEVIR-CD from Google Drive..."
gdown --id "$GDRIVE_ID" -O "$OUTPUT_ZIP" --folder

echo "Extracting..."
unzip -q "$OUTPUT_ZIP" -d "$DATA_DIR"
rm "$OUTPUT_ZIP"