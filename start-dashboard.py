import subprocess
import sys
import os
import time
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server."""
    print("Starting Dashboard API backend...")
    # Run in a separate process
    backend_process = subprocess.Popen([
        sys.executable, "-m", "src.dashboard_api"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment to ensure the server starts
    time.sleep(2)
    
    if backend_process.poll() is not None:
        # Process ended - there was an error
        stderr = backend_process.stderr.read().decode('utf-8')
        print(f"Error starting backend: {stderr}")
        sys.exit(1)
    
    print("Backend API server running at http://localhost:8000")
    return backend_process

def start_frontend():
    """Start the Next.js frontend."""
    print("Starting Dashboard frontend...")
    
    # Change to the trading-dash directory
    dashboard_dir = Path("trading-dash").absolute()
    
    if not dashboard_dir.exists():
        print(f"Error: Dashboard directory not found at {dashboard_dir}")
        sys.exit(1)
    
    # Run npm commands
    try:
        # Create .env.local with backend URL
        with open(dashboard_dir / ".env.local", "w") as f:
            f.write("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
            f.write("NEXT_PUBLIC_WS_URL=ws://localhost:8000\n")
        
        # Start the Next.js development server
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"], 
            cwd=dashboard_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment to ensure the server starts
        time.sleep(5)
        
        if frontend_process.poll() is not None:
            # Process ended - there was an error
            stderr = frontend_process.stderr.read().decode('utf-8')
            print(f"Error starting frontend: {stderr}")
            sys.exit(1)
        
        print("Frontend server running at http://localhost:3000")
        return frontend_process
        
    except Exception as e:
        print(f"Error starting frontend: {e}")
        sys.exit(1)

def main():
    """Main entry point to start the dashboard."""
    print("Starting Trading Bot Dashboard...")
    
    # Start the backend server
    backend_process = start_backend()
    
    # Start the frontend server
    frontend_process = start_frontend()
    
    print("\n=================================")
    print("Trading Dashboard is now running!")
    print("=================================")
    print("Frontend: http://localhost:3000")
    print("Backend API: http://localhost:8000")
    print("Press Ctrl+C to stop all servers")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
            
            # Check if either process has terminated
            if backend_process.poll() is not None:
                print("Backend server stopped unexpectedly. Shutting down...")
                frontend_process.terminate()
                sys.exit(1)
            
            if frontend_process.poll() is not None:
                print("Frontend server stopped unexpectedly. Shutting down...")
                backend_process.terminate()
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Servers stopped.")

if __name__ == "__main__":
    main() 