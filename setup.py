#!/usr/bin/env python3
"""
Multi-modal AI Agent Setup Script

This script helps users set up and configure the multi-modal AI agent.
It handles environment setup, dependency installation, and initial configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_banner():
    """Print the setup banner."""
    print("=" * 60)
    print("🤖 Multi-modal AI Agent Setup")
    print("=" * 60)
    print("Advanced content summarization with text and visual analysis")
    print("=" * 60)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")


def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker: {result.stdout.strip()}")
        
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker Compose: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker or Docker Compose not found")
        print("Please install Docker and Docker Compose first:")
        print("https://docs.docker.com/get-docker/")
        return False


def create_env_file():
    """Create environment file from template."""
    env_file = Path(".env")
    env_example = Path("config.env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ config.env.example not found")
        return False
    
    # Copy example file
    shutil.copy(env_example, env_file)
    print("✅ Created .env file from template")
    print("⚠️  Please edit .env file with your API keys")
    return True


def get_api_keys():
    """Prompt user for API keys."""
    print("\n🔑 API Configuration")
    print("You need at least one LLM API key to use the summarization features.")
    
    openai_key = input("OpenAI API Key (optional, press Enter to skip): ").strip()
    anthropic_key = input("Anthropic API Key (optional, press Enter to skip): ").strip()
    
    if not openai_key and not anthropic_key:
        print("⚠️  Warning: No API keys provided. Summarization features will not work.")
        print("You can still use web scraping and image analysis features.")
    
    return openai_key, anthropic_key


def update_env_file(openai_key, anthropic_key):
    """Update .env file with API keys."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read current content
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Update API keys
    if openai_key:
        content = content.replace('your_openai_api_key_here', openai_key)
    if anthropic_key:
        content = content.replace('your_anthropic_api_key_here', anthropic_key)
    
    # Write updated content
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated .env file with API keys")
    return True


def build_docker_images():
    """Build Docker images."""
    print("\n🐳 Building Docker Images")
    try:
        subprocess.run(['docker-compose', 'build'], check=True)
        print("✅ Docker images built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build Docker images: {e}")
        return False


def start_services():
    """Start the services."""
    print("\n🚀 Starting Services")
    try:
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        print("✅ Services started successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}")
        return False


def check_services():
    """Check if services are running."""
    print("\n🔍 Checking Service Status")
    try:
        result = subprocess.run(['docker-compose', 'ps'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check services: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Access the frontend: http://localhost:8501")
    print("2. Access the API docs: http://localhost:8000/docs")
    print("3. Check service status: docker-compose ps")
    print("4. View logs: docker-compose logs -f")
    print("\n🔧 Useful Commands:")
    print("- Stop services: docker-compose down")
    print("- Restart services: docker-compose restart")
    print("- Update services: docker-compose pull && docker-compose up -d")
    print("- Clear cache: curl http://localhost:8000/cache/clear")
    print("\n📚 Documentation:")
    print("- README.md: Project overview and usage")
    print("- API docs: http://localhost:8000/docs")
    print("\n⚠️  Important Notes:")
    print("- Make sure you have valid API keys in .env file")
    print("- First run may take longer due to model downloads")
    print("- Check logs if you encounter issues")


def main():
    """Main setup function."""
    print_banner()
    
    # Check prerequisites
    print("\n🔍 Checking Prerequisites")
    check_python_version()
    
    if not check_docker():
        sys.exit(1)
    
    # Create environment file
    print("\n📝 Environment Setup")
    if not create_env_file():
        sys.exit(1)
    
    # Get API keys
    openai_key, anthropic_key = get_api_keys()
    
    # Update environment file
    if not update_env_file(openai_key, anthropic_key):
        sys.exit(1)
    
    # Build and start services
    if not build_docker_images():
        print("❌ Setup failed during Docker build")
        sys.exit(1)
    
    if not start_services():
        print("❌ Setup failed during service startup")
        sys.exit(1)
    
    # Check services
    check_services()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 