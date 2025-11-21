{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.virtualenv
    python3Packages.flask
    python3Packages.pillow
    python3Packages.numpy
    python3Packages.gunicorn

    # Development tools
    python3Packages.black
    python3Packages.flake8

    # Containerization
    podman
    podman-compose

    # System dependencies
    gcc
    stdenv.cc.cc.lib
    stdenv.cc.cc
    zlib
    glibc

    #Azure
    azure-cli
  ];

  shellHook = ''
    echo " Arabic Digits Recognition Development Environment"
    echo "Python version: $(python --version)"
    echo ""
    echo "Available commands:"
    echo "  python app.py              - Run Flask app"
    # echo "  ./create-deployment-package.sh - Create Azure deployment package"
    # echo "  ./deploy-to-azure.sh       - Deploy to Azure"
    echo ""

    # Set up Python path
    export PYTHONPATH="$PWD:$PYTHONPATH"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
      echo "Creating Python virtual environment..."
      python -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Set LD_LIBRARY_PATH for TensorFlow
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glibc}/lib:$LD_LIBRARY_PATH"

    # Upgrade pip and install requirements
    pip install --upgrade pip
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi

    echo " Environment ready!"
  '';
}
