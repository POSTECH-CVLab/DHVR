CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
echo "CUDA_HOME=${CUDA_HOME}"

echo "=> Create temporary directory"
mkdir tmp && cd tmp

echo "=> Cloning thrid party library"
git clone https://github.com/KinglittleQ/torch-batch-svd.git
cd torch-batch-svd

echo "=> Install"
CUDA_HOME=/usr/local/cuda python setup.py install && cd ../.. && rm -r tmp