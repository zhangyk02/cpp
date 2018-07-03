#include "Layer.h"
#include "MyStopWatch.h"
#include <vector>
#include <sstream>
#include <map>

#ifndef CNNHEADQUATER_H
#define CNNHEADQUATER_H
class CnnHeadQuarter {
private:
	//vector<Layer> cnnLayers;
	Layer** cnnLayers;
	int layerNum;
	vector<map<string, string> > layerConfigß;
	map<string, double> paraConfig;
	vector<MatrixXf> trainLabelsEigen;
	vector<MatrixXf> trainImagesEigen;
	vector<MatrixXf> testLabelsEigen;
	vector<MatrixXf> testImagesEigen;
	vector<vector<vector<MatrixXf> > > batchedTrainImage;
	vector<vector<vector<MatrixXf> > > batchedTestImage;
	vector<vector<vector<MatrixXf> > > batchedTrainLabel;
	vector<vector<vector<MatrixXf> > > batchedTestLabel;
	int totalSamples;
	int correctSamples;
	int maxIter;
	int batchSize;
	int classNum;

	//为了让处理输入更优雅
	// Value-Defintions of the different String values
	enum LayerValue { NOT_DEFINED,
	                  INPUT_LAYER,
	                  CONV_LAYER,
	                  MAX_POOLING_LAYER,
	                  MEAN_POOLING_LAYER,
	                  RELU_LAYER,
	                  SIGMOID_LAYER,
	                  PRERELU_LAYER,
	                  TANH_LAYER,
	                  FULL_CONNECTED_LAYER
	                };

// Map to associate the strings with the enum values
	map<string, LayerValue> stringToLayerMap;

	void initStringToLayerMap() {
		stringToLayerMap["input"] = INPUT_LAYER;
		stringToLayerMap["conv"] = CONV_LAYER;
		stringToLayerMap["max_pooling"] = MAX_POOLING_LAYER;
		stringToLayerMap["mean_pooling"] = MEAN_POOLING_LAYER;
		stringToLayerMap["relu_active"] = RELU_LAYER;
		stringToLayerMap["sigmoid_active"] = SIGMOID_LAYER;
		stringToLayerMap["tanh_active"] = TANH_LAYER;
		stringToLayerMap["prerelu_active"] = PRERELU_LAYER;
		stringToLayerMap["full_connected"] = FULL_CONNECTED_LAYER;
		// cout << "stringToLayerMap contains "
		//      << stringToLayerMap.size()
		//      << " entries." << endl;
	}



	//四个数据
public:
	CnnHeadQuarter(vector<map<string, string> > layerConfig, map<string, double> paraConfig) {
		initStringToLayerMap();
		maxIter = (int)paraConfig["max_iter"];
		batchSize = (int)paraConfig["batch_size"];
		classNum = (int)paraConfig["class_num"];
		layerNum = layerConfig.size();
		cnnLayers = new Layer* [layerConfig.size()];
		string info = setupLayers(layerConfig, paraConfig);
		if (info.compare("ready") != 0) {
			cout << "config error, system terminate" << endl << "detail:" << info << endl;
			return;
		}


	}
	void batchedTrain() {
		cout << "in batchedTrain" << endl;
		int batchNum = trainImagesEigen.size() / batchSize;
		cout << batchNum << endl;
		for (int bn = 0; bn < batchNum; bn++) {
			vector<vector<MatrixXf> > tmpVVMat;
			vector<MatrixXf> tmpVMat;
			MatrixXf trainLable(classNum, batchSize);
			//cout<<"trainLable: "<<trainLable.rows()<<"x"<<trainLable.cols()<<endl;
			for (int i = bn * batchSize; i < (bn + 1)*batchSize; i++) {
				tmpVMat.push_back(trainImagesEigen[i]);
				tmpVVMat.push_back(tmpVMat);
				trainLable.block(0, i - bn * batchSize, classNum, 1) = trainLabelsEigen[i];
			}
			//cout<<trainLable<<endl;
			//tmpVVMat.push_back(tmpVMat);
			batchedTrainImage.push_back(tmpVVMat);
			//cout<<"tmpVVMat.size(): "<<tmpVVMat.size()<<endl;
			//cout<<"batchedTrainImage.size(): "<<batchedTrainImage.size()<<endl;
			vector<MatrixXf> trainLableVector;
			trainLableVector.push_back(trainLable);
			vector<vector<MatrixXf> > trainLableVectorVector;
			trainLableVectorVector.push_back(trainLableVector);
			batchedTrainLabel.push_back(trainLableVectorVector);
		}

		//cout<<batchedTrainImage.size()<<"x"<<batchedTrainImage[0].size()<<"x"<<batchedTrainImage[0][0].size()<<endl;
		MyStopWatch sw;

		for (int j = 0; j < maxIter; j++)
		{
			cout << "iter: " << j << endl;
			for (int i = 0; i < batchedTrainImage.size(); i++) {
				cout << "batch: " << i << endl;
				//cout << "i=" << i << "batched size:" << batchedTrainImage.size() << endl;

				//cout << "layer" << 0 << ": " << cnnLayers[0]->getType() << " ";
				//cout << flush;
				sw.timeIntervalFromLastClick();
				cnnLayers[0]->feedForward(batchedTrainImage[i]);
				vector<vector<MatrixXf> > output_map = cnnLayers[0]->getOutputBatchedMap();
				cout << "time(ms) in " << cnnLayers[0]->getType() << "'s ff: " << sw.timeIntervalFromLastClick() << endl;
				bool noTrivalChange;
				for (int l = 1; l < layerNum; l++) {
					//cout << "layer" << l << ": " << cnnLayers[l]->getType() << " ";
					cnnLayers[l]->feedForward(output_map);
					//cout<<"break point 8"<<endl;
					output_map = cnnLayers[l]->getOutputBatchedMap();
					cout << "time(ms) in " << cnnLayers[l]->getType() << "'s ff: " << sw.timeIntervalFromLastClick() << endl;

					//cout<<"break point 9"<<endl;
					//cnnLayers[l].tuneWeight();// 需要加息
				}
				//cout << "layer" << layerNum - 1 << ": " << cnnLayers[layerNum - 1]->getType() << " ";
				cnnLayers[layerNum - 1]->backForward(batchedTrainLabel[i]);
				vector<vector<MatrixXf> > input_sensitive_map = cnnLayers[layerNum - 1]->getInputSensitiveMap();
				cout << "time(ms) in " << cnnLayers[0]->getType() << "'s bp: " << sw.timeIntervalFromLastClick() << endl;

				//cout << "point12" << endl;


				for (int l = layerNum - 2; l > 0; l--) {
					//cout << "layer" << l << ": " << cnnLayers[l]->getType() << " ";
					cnnLayers[l]->backForward(input_sensitive_map);
					input_sensitive_map = cnnLayers[l]->getInputSensitiveMap();
					cout << "time(ms) in " << cnnLayers[l]->getType() << "'s bp: " << sw.timeIntervalFromLastClick() << endl;

				}


			}
		}
	}
	void test() {
		for (int i = 0; i <= batchedTestImage.size(); i++) {
			cnnLayers[i]->feedForward(batchedTestImage[i]);
			//cnnLayers[i].updateRecord(batchedTestLabel[i]);
			cnnLayers[i]->updateRecord();
		}
	}

	void feedFarward(vector<vector<MatrixXf> > batchedInputImage);
	void backFarward(vector<vector<MatrixXf> > batchedLabel);
	void updateRecord(vector<vector<MatrixXf> > batchedLabel);
	double accuracy() {
		return totalSamples == 0 ? 0 : (double)correctSamples / totalSamples;
	}

	string setupLayers(vector<map<string, string> > layerConfig, map<string, double> paraConfig) {

		if (layerConfig.size() <= 0) return "null layer config.";
		if (layerConfig[0]["type"].compare("input") != 0) return "input layer must be first layer.";
		if (layerConfig[layerConfig.size() - 1]["type"].compare("full_connected") != 0) {
			//cout << "last layer type: " << layerConfig[layerConfig.size() - 1]["type"] << endl << endl;
			return "output(full_connected) layer must be last layer.";
		}

		int outputMapNumOfLastLayer, singleOutputMapSizeOfLastLayer ; //initialized when input layer(layer[0]) setup
		for (int i = 0; i < layerConfig.size(); i++)
		{
			cout << "setupping layer " << i << endl;
			switch (stringToLayerMap[layerConfig[i]["type"]]) {
			case INPUT_LAYER:
				cnnLayers[i] = new InputLayer(atoi(layerConfig[i]["single_map_size"].c_str()), atoi(layerConfig[i]["map_num"].c_str()));
				//cnnLayers.push_back(InputLayer(atoi(layerConfig[i]["single_map_size"].c_str()), atoi(layerConfig[i]["map_num"].c_str())));
				break;
			case CONV_LAYER:
				cnnLayers[i] = new ConvLayer(atoi(layerConfig[i]["ker_size"].c_str()), atoi(layerConfig[i]["channel_out_num"].c_str()), singleOutputMapSizeOfLastLayer, outputMapNumOfLastLayer);
				//cnnLayers.push_back(ConvLayer(atoi(layerConfig[i]["ker_size"].c_str()), atoi(layerConfig[i]["chanel_out_num"].c_str()), singleOutputMapSizeOfLastLayer));
				break;
			case MAX_POOLING_LAYER:
				//需要保证window_num是输入图片size的约数
				if (singleOutputMapSizeOfLastLayer % atoi(layerConfig[i]["window_num"].c_str())  != 0) {
					ostringstream buf;
					buf << "Pathetically, map size must be divisible by window size at present." << endl;
					buf << "layer num: " << i << " ";
					buf << "(where map size= " << singleOutputMapSizeOfLastLayer << ", window size= " << atoi(layerConfig[i]["window_num"].c_str()) << ").";
					return buf.str();
				}
				cnnLayers[i] = new MaxPoolingLayer(atoi(layerConfig[i]["window_num"].c_str()), singleOutputMapSizeOfLastLayer, outputMapNumOfLastLayer);
				//cnnLayers.push_back(MaxPoolingLayer(atoi(layerConfig[i]["window_num"].c_str()), singleOutputMapSizeOfLastLayer, outputMapNumOfLastLayer));
				break;
			case RELU_LAYER:
				cnnLayers[i] = new ReluActivateLayer(singleOutputMapSizeOfLastLayer, outputMapNumOfLastLayer);
				//cnnLayers.push_back(ReluActivateLayer(singleOutputMapSizeOfLastLayer, outputMapNumOfLastLayer));
				break;
			case FULL_CONNECTED_LAYER:
				//FCLayer(int class_num, int singleMapSizeIn, int channelInNum);
				cnnLayers[i] = new FCLayer((int)paraConfig["class_num"], singleOutputMapSizeOfLastLayer, outputMapNumOfLastLayer);
				//cnnLayers.push_back(FCLayer((int)paraConfig["class_num"], singleOutputMapSizeOfLastLayer, outputMapNumOfLastLayer));

				break;
			default :
				return "undefined layer type.";
			}
			cnnLayers[i]->setBatchSize(paraConfig["batch_size"]);
			cnnLayers[i]->setInputMapNum(outputMapNumOfLastLayer);
			cnnLayers[i]->setEpsilong(paraConfig["epsilong"]);
			cnnLayers[i]->setLearningRate(paraConfig["learning_rate"]);
			cnnLayers[i]->setRegulationRatio(paraConfig["regulation_ratio"]);
			cnnLayers[i]->init();
			//cnnLayers[i].setSingleInputMapSize(singleOutputMapSizeOfLastLayer);
			outputMapNumOfLastLayer = cnnLayers[i]->getOutputNum();
			singleOutputMapSizeOfLastLayer = cnnLayers[i]->getSingleOutputMapSize();

			//cout << cnnLayers[i]->getBatchSize() << " " << cnnLayers[i]->getOutputNum() << " " << cnnLayers[i]->getSingleOutputMapSize() << endl;
			//<<" "cnnLayers[i]-><<" "cnnLayers[i]-><<" "cnnLayers[i]-><<" ";

		}
		return "ready";
	}

	void setMnistData(vector<MatrixXf> trainImagesEigen, vector<MatrixXf> trainLabelsEigen, vector<MatrixXf> testImagesEigen, vector<MatrixXf> testLabelsEigen) {
		this->trainImagesEigen = trainImagesEigen;
		this->trainLabelsEigen = trainLabelsEigen;
		this->testImagesEigen = testImagesEigen;
		this->testLabelsEigen = testLabelsEigen;
	}
};

#endif



