#include "Layer.h"
#include "CnnHeadQuarter.h"
#include "MnistReader.h"
#include <iostream>
using namespace Eigen;
using namespace std;

void getCnnConfig(map<string, double> &paraConfig, vector<map<string, string> > &layerConfig) {

	//cnn整体配置:

	//map<string, double> paraConfig;
	paraConfig["batch_size"] = 50;
	paraConfig["epsilong"] = 0.0001;
	paraConfig["learning_rate"] = 0.01;
	paraConfig["regulation_ratio"] = 0.1;
	paraConfig["class_num"] = 10;
	paraConfig["max_iter"] = 50;

	//vector<map<string, string> > layerConfig;
	//输入层
	map<string, string> trainLayer;
	trainLayer["type"] = "input";
	trainLayer["single_map_size"] = "28";
	trainLayer["map_num"] = "1";
	layerConfig.push_back(trainLayer);

	//卷积层1
	map<string, string> convLayer1;
	convLayer1["type"] = "conv";
	convLayer1["ker_size"] = "5";
	convLayer1["channel_out_num"] = "6";
	layerConfig.push_back(convLayer1);
	//max池化层1
	map<string, string> poolingLayer1;
	poolingLayer1["type"] = "max_pooling";
	poolingLayer1["window_num"] = "2";
	layerConfig.push_back(poolingLayer1);
	//relu激励层1
	map<string, string> reluLayer1;
	reluLayer1["type"] = "relu_active";
	layerConfig.push_back(reluLayer1);

	//卷积层2
	map<string, string> convLayer2;
	convLayer2["type"] = "conv";
	convLayer2["ker_size"] = "5";
	convLayer2["channel_out_num"] = "4";
	layerConfig.push_back(convLayer2);
	//max池化层1
	map<string, string> poolingLayer2;
	poolingLayer2["type"] = "max_pooling";
	poolingLayer2["window_num"] = "2";
	layerConfig.push_back(poolingLayer2);
	//relu激励层2
	map<string, string> reluLayer2;
	reluLayer2["type"] = "relu_active";
	layerConfig.push_back(reluLayer2);

	//输出层(全连接)
	map<string, string> fcLayer;
	fcLayer["type"] = "full_connected";
	layerConfig.push_back(fcLayer);


}



int main() {

	//获取mnist输入
	MnistReader rawData = MnistReader("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	rawData.readData();
	rawData.toEigenMatrix();

	//获取配置文件（配置目前需手写，改动getCnnConfig内容即可）：
	map<string, double> paraConfig;
	vector<map<string, string> > layerConfig;

	getCnnConfig(paraConfig, layerConfig);

	//配置cnnHeadquarter:
	CnnHeadQuarter headQuarter(layerConfig, paraConfig);
	cout << "setup success" << endl;
	headQuarter.setMnistData(rawData.trainImagesEigen, rawData.trainLabelsEigen, rawData.testImagesEigen, rawData.testLabelsEigen);
	//训练
	headQuarter.batchedTrain();
	cout << "Train success" << endl;

	//headQuarter.test();


	// //debug
	// Layer *baseL = new Layer();
	// Layer *inputL = new InputLayer();
	// Layer *convL = new ConvLayer();
	// Layer *fcL = new FCLayer();
	// cout << baseL->getType() << " " << inputL->getType() << " " << convL->getType() << " " << fcL->getType() << endl;
	// fcL->setDataIn(MatrixXf::Random(10, 5));
	// fcL->setClassNum(10);
	// fcL->initWandB();
	// fcL->calculateSoftmax();
	// fcL->outputLabels();

	// // (dynamic_cast<FCLayer*>(fcL))->setDataIn(MatrixXf::Random(10,5));
	// // (dynamic_cast<FCLayer*>(fcL))->setClassNum(10);
	// // (dynamic_cast<FCLayer*>(fcL))->initWandB();
	// // (dynamic_cast<FCLayer*>(fcL))->calculateSoftmax();
	// // (dynamic_cast<FCLayer*>(fcL))->outputLabels();

	// Layer *ps = new Layer[3];
	// ps[0] = InputLayer();
	// ps[1] = ConvLayer();
	// ps[2] = FCLayer();
	// ps[2].setClassNum(10);
	// ps[2].setDataIn(MatrixXf::Random(10, 5));

	// Layer active = ActivateLayer();
	// active.feedForward();
	// active.backForward();

	// Layer relu = ReluActivateLayer();
	// cout << relu.getType() << endl;



}