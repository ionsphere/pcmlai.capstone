import argparse
import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Start Clothing Price Predictor API")
    parser.add_argument('--host', type=str, default=os.getenv('API_HOST', '0.0.0.0'),
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=int(os.getenv('API_PORT', '8000')),
                        help='Port to bind to')
    parser.add_argument('--reload', action='store_true', default=os.getenv('API_RELOAD', 'false').lower() == 'true',
                        help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=int(os.getenv('API_WORKERS', '1')),
                        help='Number of worker processes')
    parser.add_argument('--log-level', type=str, default=os.getenv('LOG_LEVEL', 'info'),
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    print("Clothing Price Predictor API")
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Auto-reload: {args.reload}")
    print(f"Log level: {args.log_level}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    
    import uvicorn
    
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == '__main__':
    main()
