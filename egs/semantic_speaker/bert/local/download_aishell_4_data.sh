data_dir=$1

pushd $data_dir
echo "Downloading aishell-4 datasets..."

wget https://openslr.elda.org/resources/111/train_L.tar.gz --no-check-certificate
wget https://openslr.elda.org/resources/111/train_M.tar.gz --no-check-certificate
wget https://openslr.elda.org/resources/111/train_S.tar.gz --no-check-certificate
wget https://openslr.elda.org/resources/111/test.tar.gz --no-check-certificate

tar -zxvf train_L.tar.gz
tar -zxvf train_M.tar.gz
tar -zxvf train_S.tar.gz
tar -zxvf test.tar.gz
echo "Downloading aishell-4 datasets finished"

popd
