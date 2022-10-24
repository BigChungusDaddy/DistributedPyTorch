apt-get update
apt-get -y install python3-pip
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
mkdir -p data
cd data
wget -c --quiet https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
cd ..
