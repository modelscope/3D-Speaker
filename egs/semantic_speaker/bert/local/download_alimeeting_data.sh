data_dir=$1
pushd $data_dir
echo "alimeeting data..."
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_near.tar.gz
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz

tar -zxvf Train_Ali_far.tar.gz
tar -zxvf Train_Ali_near.tar.gz
tar -zxvf Eval_Ali.tar.gz
tar -zxvf Test_Ali.tar.gz

echo "Downloading alimeeting data finished."

popd

