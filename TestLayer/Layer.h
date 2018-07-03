#include <string>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "Util.h"
#include "Eigen/Dense"
#include "MyStopWatch.h"   
#ifndef LAYER_H
#define LAYER_H
using namespace std;
using namespace Eigen;

template < class T >
void ClearVector( vector< T >& vt )
{
	vector< T > vtTemp;
	vtTemp.swap( vt );
};


class Layer {
protected:
	int classNum;
	int batchSize;
	int kerSize;
	int inputMapNum;
	int outputMapNum;
	int singleInputMapSize;
	int singleOutputMapSize;
	double learningRate;
	double regulationRatio;
	double epsilong;
	vector<vector<MatrixXf> > inputBatchedMap;
	vector<vector<MatrixXf> > outputBatchedMap;
	vector<vector<MatrixXf> > inputSensitiveMap;
	vector<vector<MatrixXf> > outputSensitiveMap;

	string type;
	double specialParam;
	void setType(string s) {
		type.assign(s);
	}
public:
	Layer() {
		setType("base");
	}
	virtual ~Layer() {}
	string getType() {
		return type;
	}
	void setClassNum(int n) {
		cout << "[WARNING] Layer::setClassNum should not be applied" << endl;
		classNum = n;

	}
	void setBatchSize(int bs) {
		batchSize = bs;
	}
	int getBatchSize() {
		return batchSize;
	}
	virtual void initWandB() {
		cout << "[WARNING] Layer::initWandB should not be applied" << endl;
	}
	virtual void outputLabels() {
		cout << "[WARNING] Layer::outputLabels should not be applied" << endl;
	}
	virtual void calculateSoftmax() {
		cout << "[WARNING] Layer::calculateSoftmax should not be applied" << endl;

	}
	virtual void setDataIn(MatrixXf input) {
		cout << "[WARNING] Layer::setDataIn should not be applied" << endl;
	}
	void setInputMapNum(int inputMapN) {
		inputMapNum = inputMapN;
	}
	void setOutputMapNum(int outputMapN) {
		outputMapNum = outputMapN;
	}
	void setInputBatchedMap(vector<vector<MatrixXf> > inputMap) {
		inputBatchedMap = inputMap;
	}
	vector<vector<MatrixXf> > getOutputBatchedMap() {
		return outputBatchedMap;
	}
	virtual void feedForward(vector<vector<MatrixXf> > input_batchImage) {
		cout << "[WARNING] Layer::feedForward should not be applied" << endl;

	}
	virtual bool backForward(vector<vector<MatrixXf> > output_sensitiveMap) {
		cout << "[WARNING] Layer::backForward should not be applied" << endl;
		return false;
	}
	void setSingleInputMapSize(int singleSize) {
		singleInputMapSize = singleSize;
	}
	void setSingleOutputMapSize(int singleSize) {
		singleOutputMapSize = singleSize;
	}
	void setSpecialParam(double param) {
		specialParam = param;
	}
	virtual void init() {
		cout << "[WARNING] Layer::init should not be applied" << endl;
	}
	virtual void updateRecord(void) {
		cout << "[WARNING] Layer::updateRecord should not be applied" << endl;
	}
	int getOutputNum() {
		return outputMapNum;
	}
	int getSingleOutputMapSize() {
		return singleOutputMapSize;
	}
	vector<vector<MatrixXf> > getOutputSensitiveMap() {
		return outputSensitiveMap;
	}
	vector<vector<MatrixXf> > getInputSensitiveMap() {
		return inputSensitiveMap;
	}
	virtual bool tuneWeight() {
		// if all |dw/w| < epsilong return true; else false;
		cout << "[WARNING] Layer::tuneWeight should not be applied" << endl;
		return false;
	}
	void setLearningRate(double learning_rate) {
		learningRate = learning_rate;
	}
	void setRegulationRatio(double regulation_ratio) {
		regulationRatio = regulation_ratio;
	}
	void setEpsilong(double ep_silong) {
		epsilong = ep_silong;
	}

};

class InputLayer: public Layer {
public:
	InputLayer() {
		setType("input");
	}
	InputLayer(int singleMapSizeIn, int channelInNum) {
		setType("input");
		singleInputMapSize = singleMapSizeIn;
		singleOutputMapSize = singleMapSizeIn;
		inputMapNum = channelInNum;
		outputMapNum = channelInNum;

	}
	virtual void feedForward(vector<vector<MatrixXf> > input_batchImage) {
		//cout << "enter into input layer's feedForward" << endl;
		inputBatchedMap = input_batchImage;
		outputBatchedMap = inputBatchedMap;

	}
	virtual bool backForward(vector<vector<MatrixXf> > output_sensitiveMap) {
		cout << "[WARNING] InputLayer::backForward should not be applied" << endl;
		return false;
	}
	virtual void init() {

	}

};

class ConvLayer: public Layer {
private:
	vector<vector<MatrixXf> > W;
	vector<double> b;
	vector<vector<MatrixXf> > dW;
	vector<double> db;
public:
	ConvLayer() {
		setType("conv");
	}
	ConvLayer(int initKerSize, int initOutputMapNum, int singleMapSizeIn, int input_map_num) {
		setType("conv");
		// kerSize = atoi(initKerSize.c_str());
		// outputMapNum = atoi(initOutputMapNum.c_str());
		kerSize = initKerSize;
		outputMapNum = initOutputMapNum;
		inputMapNum = input_map_num;
		singleInputMapSize = singleMapSizeIn;
		singleOutputMapSize = singleMapSizeIn - kerSize + 1;

		// init W and b, together with dW?

		for (int i = 0; i < inputMapNum; i++)
		{
			//cout << "loop i=" << i << endl;
			vector<MatrixXf> tp;
			vector<MatrixXf> tpDeltaW;
			for (int j = 0; j < outputMapNum; j++)
			{
				tp.push_back(MatrixXf::Random(kerSize, kerSize));
				tpDeltaW.push_back(MatrixXf::Zero(kerSize, kerSize));
			}
			W.push_back(tp);
			//cout << "W[0][0]:" << W[0][0] << endl;
			dW.push_back(tpDeltaW);
		}

		for (int i = 0; i < outputMapNum; i++) {
			//cout << "second loop i=" << i << endl;
			b.push_back(0.0);
		}

	}
	// void initWandB() {
	// 	//需要知道输入与输出的通道数



	// 	//需要知道
	// }

	virtual void feedForward(vector<vector<MatrixXf> > input_batchImage) {
		//MyStopWatch sw;
		//cout << "enter into conv layer's feedForward" << endl;
		inputBatchedMap = input_batchImage;
		//cout << "inputBatchedMap.size(): " << inputBatchedMap.size() << endl;
		//outputBatchedMap.swap(vector<vector<MatrixXf> >());
		ClearVector(outputBatchedMap);
		MatrixXf tmpOne = MatrixXf::Ones(singleOutputMapSize, singleOutputMapSize);
		for (int sample = 0; sample < batchSize; sample++)
		{
			//cout << "enter into sample loop" << sample << endl;
			vector<MatrixXf> tp;
			for (int j = 0; j < outputMapNum; j++)
			{
				MatrixXf z = MatrixXf::Zero(singleOutputMapSize, singleOutputMapSize);
				//cout << "enter into j loop" << j << endl;


				for (int i = 0; i < inputMapNum; i++) {
					//cout << "enter into i loop" << i << endl;

					//cout << inputBatchedMap[sample][i] << endl << W[i][j] << endl;

					z += Util::validConv(inputBatchedMap[sample][i], W[i][j]);
				}
				//cout << "end of i loop" << endl;

				z += tmpOne * b[j];
				tp.push_back(z);

			}
			outputBatchedMap.push_back(tp);
		}
		//cout<<sw.timeIntervalFromInit()

	}

	virtual bool backForward(vector<vector<MatrixXf> > output_sensitiveMap) {
		//cout << "enter into conv's backForward" << endl;
		outputSensitiveMap = output_sensitiveMap;
		//set dw and db=0
		vector<vector <MatrixXf> > tdW = vector<vector <MatrixXf> >(inputMapNum, vector<MatrixXf>(outputMapNum, MatrixXf::Zero(kerSize, kerSize)));
		dW.swap(tdW);
		vector<double> tdb = vector<double>(outputMapNum, 0.0);
		db.swap(tdb);
		vector<vector <MatrixXf> > tmpInputSensitiveMap = vector<vector <MatrixXf> >(batchSize, vector<MatrixXf>(inputMapNum, MatrixXf::Zero(singleInputMapSize, singleInputMapSize)));
		inputSensitiveMap.swap(tmpInputSensitiveMap);

		// 是不是还要swap一下tdb
		//
		//
		//db.swap(vector<double>(outputMapNum, 0.0));

		// dw = vector<vector<MatrixXf> >();
		// for (int i = 0; i < inputMapNum; i++)
		// {
		// 	vector<MatrixXf> tpDeltaW;
		// 	for (int j = 0; j < outputMapNum; j++)
		// 		tpDeltaW.push_back(MatrixXf::Zero(kerSize, kerSize));
		// 	dW.push_back(tp);
		// }
		//db.clear();
		// db = vector<double>();

		// for (int i = 0; i < outputMapNum; i++) {
		// 	db.push_back(0.0);
		// }


		for (int s = 0; s < batchSize; s++) {
			for (int i = 0; i < inputMapNum; i++)
				for (int j = 0; j < outputMapNum; j++)
				{
					//update delta
					//cout << "conv break point 1-1" << endl;
					MatrixXf tmp = Util::fullConv(outputSensitiveMap[s][j], W[i][j].reverse());
					//cout << "tmp:" << tmp.rows() << "x" << tmp.cols() << endl;
					//cout << "inputSensitiveMap[s][i]:" << inputSensitiveMap[s][i].rows() << "x" << inputSensitiveMap[s][i].cols() << endl;

					inputSensitiveMap[s][i] += Util::fullConv(outputSensitiveMap[s][j], W[i][j].reverse());
					//update dW
					//cout << "conv break point 1-2" << endl;

					dW[i][j] += Util::validConv(inputBatchedMap[s][i], outputSensitiveMap[s][j]);
					// update b
					//cout << "conv break point 1-3" << endl;

					db[j] += outputSensitiveMap[s][j].sum(); //值得check
				}
		}

		//cout << "conv break point 1" << endl;

		for (int j = 0; j < outputMapNum; j++)
		{	db[j] /= batchSize;

			for (int i = 0; i < inputMapNum; i++)

			{
				dW[i][j] /= batchSize;

			}

		}
		//cout << "conv break point 2" << endl;

		bool isTrivalChange = true;


		for (int j = 0; j < outputMapNum; j++)
		{
			if (abs(b[j]) > epsilong && abs(db[j] / b[j]) > epsilong) isTrivalChange = false;
			b[j] -= learningRate * db[j];
			for (int i = 0; i < inputMapNum; i++)
			{
				for (int m = 0; m < kerSize; m++)
					for (int n = 0; n < kerSize; n++)
					{
						if (abs(W[i][j](m, n)) > epsilong && abs(dW[i][j](m, n) / W[i][j](m, n)) > epsilong) isTrivalChange = false;
						W[i][j](m, n) -= learningRate * dW[i][j](m, n) + regulationRatio * W[i][j](m, n);
					}
			}

		}

		return isTrivalChange;
	}

	void init() {
		//inputSensitiveMap = vector<vector <MatrixXf> >(batchSize, vector<MatrixXf>(inputMapNum, ));
	}

};

class PoolingLayer: public Layer {
public:
	PoolingLayer() {
		setType("pooling");
	}
	virtual void feedForward(vector<vector<MatrixXf> > input_batchImage) {
		cout << "[WARNING] PoolingLayer::feedForward should not be applied" << endl;

	}
	virtual bool backForward(vector<vector<MatrixXf> > output_sensitiveMap) {
		cout << "[WARNING] PoolingLayer::backForward should not be applied" << endl;
		return false;
	}
	virtual void init() {

	}
};



class FCLayer: public Layer {
private:
	MatrixXf W;
	VectorXf b;
	MatrixXf dW;
	//MatrixXf wAndB;
	// int classNum;
	// int batchSize;
	MatrixXf dataIn;
	MatrixXf softMaxResult;
	vector<int> labelOut;
	MatrixXf reshapedInputMap;
	MatrixXf realLabel;
	MatrixXf reshapedSensitive;
public:
	FCLayer() {
		setType("fc");
	}
	FCLayer(int class_num, int singleMapSizeIn, int channelInNum) {
		//cout << "break point1" << endl;
		setType("fc");
		//cout << "break point2" << endl;

		classNum = class_num;
		//cout << "break point3" << endl;

		singleInputMapSize = singleMapSizeIn;
		//cout << "break point4" << endl;

		inputMapNum = channelInNum;
		//cout << "break point5" << endl;


	}
	void init() {
		cout << inputMapNum * singleInputMapSize * singleInputMapSize + 1 << "x" << batchSize << endl;

		reshapedInputMap = MatrixXf(inputMapNum * singleInputMapSize * singleInputMapSize + 1, batchSize);
		//cout << "break point6" << endl;

		W = MatrixXf::Random(classNum, inputMapNum * singleInputMapSize * singleInputMapSize + 1);//W和b可以合一吗？
		//cout << "break point7" << endl;

		labelOut = vector< int >(batchSize);

		inputSensitiveMap = vector<vector <MatrixXf> >(batchSize, vector<MatrixXf>(inputMapNum));

	}
	void setDataIn(MatrixXf input) {
		dataIn = input;
		setBatchSize(input.cols());
	}
	void setBatchSize(int size) {
		batchSize = size;
	}
	void setClassNum(int classN) {
		classNum = classN;
	}
	//注意调用前必须先set classNum
	// void initWandB() {
	// 	W = MatrixXf::Random(classNum, dataIn.rows());
	// 	b = VectorXf::Random(classNum);
	// 	cout << W << endl;
	// 	cout << b << endl;
	// 	// wAndB = MatrixXf(classNum,dataIn.rows()+1);
	// 	// wAndB<<W,
	// 	//        wAndB;
	// }
	//注意调用前必须先set batchSize
	void calculateSoftmax() {
		// VectorXf ones = VectorXf:Ones(batchSize);
		// MatrixXf extendInput = MatrixXf(dataIn.rows(),dataIn.cols()+1);
		// extendInput<<dataIn,ones;
		cout << "W" << W.rows() << "," << W.cols() << endl;
		cout << "feature" << dataIn.rows() << "," << dataIn.cols() << endl;
		cout << "b" << b.rows() << "," << b.cols() << endl;
		softMaxResult = Util::colSoftMax(W * dataIn + b * RowVectorXf::Ones(batchSize));
		cout << softMaxResult << endl;
	}

	virtual void feedForward(vector<vector<MatrixXf> > input_batchImage) {
		//cout << "point1" << endl;
		inputBatchedMap = input_batchImage;
		//construct reshaped input via inputBatchedMap
		//cout << "point2" << endl;
		for (int b = 0; b < batchSize; b++) {
			for (int m = 0; m < inputMapNum; m++) {
				Map<VectorXf> tmpV(inputBatchedMap[b][m].data(), singleInputMapSize * singleInputMapSize);
				reshapedInputMap.block(singleInputMapSize * singleInputMapSize * m, b, singleInputMapSize * singleInputMapSize, 1) = tmpV;
				// 是不是要将tmpV释放，需了解Eigen的内存释放机制
			}
			//cout << "point2.5" << endl;
			//cout << "reshapedInputMap.size: " << reshapedInputMap.rows() << "x" << reshapedInputMap.cols() << endl;
			//cout << "singleInputMapSize * singleInputMapSize * inputMapNum: " << singleInputMapSize * singleInputMapSize * inputMapNum << endl;
			//cout << "b: " << b << endl;

			reshapedInputMap(singleInputMapSize * singleInputMapSize * inputMapNum , b) = 1;
		}
		//W:ClassNum*(allInput+1), reshapedInputMap: (allInput+1)*batchSize, W*reshapedInputMap: ClassNum*batchSize
		//cout << "point3" << endl;
		//cout << "W: " << W.rows() << "x" << W.cols() << endl;
		//cout << "reshapedInputMap: " << reshapedInputMap.rows() << "x" << reshapedInputMap.cols() << endl;
		softMaxResult = Util::colSoftMax(W * reshapedInputMap);
		//cout << "point4" << endl;
		for (int i = 0; i < batchSize; i++) {
			labelOut[i] = Util::maxIndex(softMaxResult.col(i));
			//cout << labels[i] << ",";
		}
		//cout << "point5" << endl;
		//cout << endl;
	}
	virtual bool backForward(vector<vector<MatrixXf> > output_sensitiveMap) {
		//cout << "enter into fc's backForward" << endl;
		realLabel = output_sensitiveMap[0][0];
		//cout << "realLabel:" << realLabel.rows() << "x" << realLabel.cols() << endl;
		//cout << "reshapedInputMap:" << reshapedInputMap.rows() << "x" << reshapedInputMap.cols() << endl;
		reshapedSensitive = W.transpose() * (realLabel - softMaxResult);
		//recover sentiveMap by reshapedSensitive
		//cout << "point10" << endl;
		//cout << "point11" << endl;


		for (int b = 0; b < batchSize; b++) {
			for (int m = 0; m < inputMapNum; m++) {
				Map<MatrixXf> tmpSingleSensitiveMap(reshapedInputMap.block(singleInputMapSize * singleInputMapSize * m, b, singleInputMapSize * singleInputMapSize, 1).data(), singleInputMapSize, singleInputMapSize);
				inputSensitiveMap[b][m] = tmpSingleSensitiveMap;
				//cout << "point6" << endl;
				// IMPROVE 可以考虑将tmpSingleSensitiveMap值拷贝给inputSensitiveMap[b][m]，再释放tmpSingleSensitiveMap
			}
		}
		// softMaxResult: classNum * batchedSize W:ClassNum*(allInput+1) dW:ClassNum * (allInput+1) reshapedInputMap: (allInput+1)*batchSize
		dW = (realLabel - softMaxResult) * reshapedInputMap.transpose() / batchSize ;
		bool isTrivalChange = true;
		for (int i = 0; i < W.rows(); i++) {
			for (int j = 0; j < W.cols(); j++) {
				if (abs(W(i, j)) > epsilong && abs(dW(i, j) / W(i, j)) > epsilong) isTrivalChange = false;
				W(i, j) -= (dW(i, j) * learningRate + regulationRatio * W(i, j));
			}
		}
		//cout << "point7" << endl;
		// ATTENTION 这里的正负号值得注意
		return isTrivalChange;
	}





};

class ActivateLayer : public Layer {
protected:
	//batchedMap[i][j] 代表一个batch中，第i个样本的第j副图


public:
	ActivateLayer() {
		setType("activation");
	}
	virtual void feedForward(vector<vector<MatrixXf> > input_batchImage) {
		inputBatchedMap = input_batchImage;
		cout << "[WARNING] ActivateLayer::feedForward should not be applied" << endl;
		//calculate outputMap
		// for (int i = 0; i < inputBatchedMap.size(); i++) {
		// 	vector<MatrixXf> tp;
		// 	for (int j = 0; j < inputBatchedMap[0].size(); j++) {
		// 		MatrixXf tmpMap(singleInputMapSize, singleInputMapSize);
		// 		for (int k = 0; k < singleInputMapSize; k++)
		// 			for (int m = 0; m < singleInputMapSize; m++)
		// 				tmpMap(k, m) = inputBatchedMap[i][j](k, m);
		// 		tp.push_back(tmpMap);

		// 	}
		// 	outputBatchedMap.push_back(tp);
		// }
	}
	virtual bool backForward(vector<vector<MatrixXf> > output_sensitiveMap) {
		outputSensitiveMap = output_sensitiveMap;
		cout << "[WARNING] ActivateLayer::backForward should not be applied" << endl;

		//calculate inputSensitiveMap
		// for (int i = 0; i < inputBatchedMap.size(); i++) {
		// 	vector<MatrixXf> tp;
		// 	for (int j = 0; j < inputBatchedMap[0].size(); j++) {
		// 		MatrixXf tmpMap(singleInputMapSize, singleInputMapSize);
		// 		for (int k = 0; k < singleInputMapSize; k++)
		// 			for (int m = 0; m < singleInputMapSize; m++)
		// 				tmpMap(k, m) = outputSensitiveMap[i][j](k, m) * outputBatchedMap[i][j](k, m);
		// 		tp.push_back(tmpMap);

		// 	}
		// 	inputSensitiveMap.push_back(tp);
		// }
		return false;
	}
	virtual void init() {

	}

};

class ReluActivateLayer: public ActivateLayer {
public:
	ReluActivateLayer(int singleMapSizeIn, int channelInNum) {
		setType("activation");
		singleInputMapSize = singleMapSizeIn;
		singleOutputMapSize = singleMapSizeIn;
		inputMapNum = channelInNum;
		outputMapNum = channelInNum;
	}
	virtual void feedForward(vector<vector<MatrixXf> > input_batchImage) {
		//cout << "enter into ReluActivateLayer's feedForward" << endl;
		inputBatchedMap = input_batchImage;
		ClearVector(outputBatchedMap);
		//calculate outputMap
		for (int i = 0; i < inputBatchedMap.size(); i++) {
			vector<MatrixXf> tp;
			for (int j = 0; j < inputBatchedMap[0].size(); j++) {
				MatrixXf tmpMap(singleInputMapSize, singleInputMapSize);
				for (int k = 0; k < singleInputMapSize; k++)
					for (int m = 0; m < singleInputMapSize; m++)

						tmpMap(k, m) = inputBatchedMap[i][j](k, m) > 0 ? inputBatchedMap[i][j](k, m) : 0;
				tp.push_back(tmpMap);

			}
			outputBatchedMap.push_back(tp);
		}
	}
	virtual bool backForward(vector<vector<MatrixXf> > output_sensitiveMap) {
		outputSensitiveMap = output_sensitiveMap;
		//cout<<"outputSensitiveMap: "<<outputSensitiveMap.size()<<"x"<<outputSensitiveMap[0].size()<<endl;
		//cout<<"inputBatchedMap: "<<inputBatchedMap.size()<<"x"<<inputBatchedMap[0].size()<<endl;
		//calculate inputSentiveMap
		ClearVector(inputSensitiveMap);
		for (int i = 0; i < inputBatchedMap.size(); i++) {
			vector<MatrixXf> tp;
			for (int j = 0; j < inputBatchedMap[0].size(); j++) {
				MatrixXf tmpMap(singleInputMapSize, singleInputMapSize);
				for (int k = 0; k < singleInputMapSize; k++)
					for (int m = 0; m < singleInputMapSize; m++)
						tmpMap(k, m) = outputSensitiveMap[i][j](k, m) * (outputBatchedMap[i][j](k, m) > 0 ? 1 : 0);
				tp.push_back(tmpMap);

			}
			inputSensitiveMap.push_back(tp);
		}
		return true;
	}
	void init(){

	}
};

class MaxPoolingLayer: public PoolingLayer {
private:
	int windowNum;
	//需要初始化敏感输入矩阵（全置零）
public:
	MaxPoolingLayer(int window_num, int singleInputSize , int channelInNum) {
		setType("max_pooling");
		//specialParam = (double)atoi(singleInputSize.c_str());
		singleInputMapSize =  singleInputSize;
		inputMapNum = channelInNum;
		windowNum = window_num;
		singleOutputMapSize = singleInputMapSize / windowNum;
		outputMapNum = channelInNum;
		specialParam = (double)singleOutputMapSize;

	}


	virtual void feedForward(vector<vector<MatrixXf> > input_batchImage) {
		inputBatchedMap = input_batchImage;
		int poolSize = (int)specialParam;
		ClearVector(outputBatchedMap);
		//cout << "poolSize:" << poolSize << endl;
		//cout << "inputBatchedMap.size(): " << inputBatchedMap.size() << endl;
		for (int i = 0; i < inputBatchedMap.size(); i++) {
			//cout << "feedForward of maxPooling, loop i: " << i << endl;
			vector<MatrixXf> tp;
			for (int j = 0; j < inputBatchedMap[0].size(); j++) {
				//cout << "feedForward of maxPooling, loop j: " << j << endl;
				MatrixXf tmpMap(singleOutputMapSize, singleOutputMapSize);
				MatrixXf tmpSensiMap = MatrixXf::Zero(singleInputMapSize, singleInputMapSize);
				for (int k = 0; k < windowNum; k++) {
					//cout << "feedForward of maxPooling, loop k: " << k << endl;
					for (int m = 0; m < windowNum; m++) {
						//point6
						//
						//
						//cout << "feedForward of maxPooling, loop m: " << m << endl;
						//double singlePoint = 0;
						int maxIndx = 0;
						int maxIndy = 0;
						//cout << "poosible fault when indexing matrix" << endl;
						int maxValue = inputBatchedMap[i][j](k * poolSize, m * poolSize);
						for (int t1 = 0; t1 < poolSize; t1++)
							for (int t2 = 0; t2 < poolSize; t2++)
								if (inputBatchedMap[i][j](k * poolSize + t1, m * poolSize + t2) > maxValue) {
									maxIndx = t1;
									maxIndy = t2;
									maxValue = inputBatchedMap[i][j](k * poolSize + t1, m * poolSize + t2);
								}
						tmpMap(k, m) = maxValue;
						tmpSensiMap(k * poolSize + maxIndx, m * poolSize + maxIndy) = 1.0;
					}
				}
				tp.push_back(tmpMap);
				inputSensitiveMap[i][j] = tmpSensiMap;
			}
			outputBatchedMap.push_back(tp);
		}
	}
	virtual bool backForward(vector<vector<MatrixXf> > output_sensitiveMap) {
		//cout << "enter into maxPooling's backForward" << endl;
		//pooling层不用更新权重
		outputSensitiveMap = output_sensitiveMap;
		//calculate inputSensitiveMap
		int poolSize = (int)specialParam;
		for (int i = 0; i < outputBatchedMap.size(); i++)
			for (int j = 0; j < outputBatchedMap[0].size(); j++)
				for (int k = 0; k < windowNum; k++)
					for (int m = 0; m < windowNum; m++)
						inputSensitiveMap[i][j].block(k * poolSize, m * poolSize, poolSize, poolSize) *= outputSensitiveMap[i][j](k, m);
		return false;
	}
	void init() {
		inputSensitiveMap = vector<vector <MatrixXf> >(batchSize, vector<MatrixXf>(inputMapNum));
	}
};


#endif