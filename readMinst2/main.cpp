#include "MnistReader.h"

int main()
{
	MnistReader rawData = MnistReader("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	rawData.readData();
	rawData.toEigenMatrix();
	rawData.showSomeSamples();

	return 0;
}