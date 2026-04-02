import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    import uvicorn
    from app import app
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
