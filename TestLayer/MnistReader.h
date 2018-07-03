#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

#ifndef MNISTREADER_H
#define MNISTREADER_H

class MnistReader {
private:
    string trainImageFileName;
    string trainLableFileName;
    string testImageFileName;
    string testLabelFileName;
    int rows;
    int cols;
    int trainSampleNum;
    int testSampleNum;

    int ReverseInt(int i)
    {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }

    void read_Mnist_Label(string filename, vector<double>&labels)
    {
        ifstream file(filename, ios::binary);
        if (file.is_open())
        {
            int magic_number = 0;
            int number_of_images = 0;
            file.read((char*)&magic_number, sizeof(magic_number));
            file.read((char*)&number_of_images, sizeof(number_of_images));
            magic_number = ReverseInt(magic_number);
            number_of_images = ReverseInt(number_of_images);
            cout << "magic number = " << magic_number << endl;
            cout << "number of images = " << number_of_images << endl;


            for (int i = 0; i < number_of_images; i++)
            {
                unsigned char label = 0;
                file.read((char*)&label, sizeof(label));
                labels.push_back((double)label);
                //cout << "reading " << filename << "..." << "i: " << i;
                //cout<<(int)label<<endl;
            }

        }
    }

    void read_Mnist_Images(string filename, vector<vector<double> >& images)
    {
        ifstream file(filename, ios::binary);
        if (file.is_open())
        {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            unsigned char label;
            file.read((char*)&magic_number, sizeof(magic_number));
            file.read((char*)&number_of_images, sizeof(number_of_images));
            file.read((char*)&n_rows, sizeof(n_rows));
            file.read((char*)&n_cols, sizeof(n_cols));
            magic_number = ReverseInt(magic_number);
            number_of_images = ReverseInt(number_of_images);
            n_rows = ReverseInt(n_rows);
            n_cols = ReverseInt(n_cols);

            cout << "magic number = " << magic_number << endl;
            cout << "number of images = " << number_of_images << endl;
            cout << "rows = " << n_rows << endl;
            cout << "cols = " << n_cols << endl;

            for (int i = 0; i < number_of_images; i++)
            {
                //cout << "reading " << filename << "..." << "i: " << i;
                vector<double>tp;
                for (int r = 0; r < n_rows; r++)
                {
                    for (int c = 0; c < n_cols; c++)
                    {
                        unsigned char image = 0;
                        file.read((char*)&image, sizeof(image));
                        tp.push_back(image);
                    }
                }
                images.push_back(tp);
            }
        }
    }

public:
    vector<double> trainLabels;
    vector<vector<double> > trainImages;
    vector<double> testLabels;
    vector<vector<double> > testImages;

    vector<MatrixXf> trainLabelsEigen;
    vector<MatrixXf> trainImagesEigen;
    vector<MatrixXf> testLabelsEigen;
    vector<MatrixXf> testImagesEigen;


    MnistReader(string trainImageFN, string trainLableFN, string testImageFN, string testLabelFN):
        trainImageFileName(trainImageFN), trainLableFileName(trainLableFN), testImageFileName(testImageFN), testLabelFileName(testLabelFN) {
        rows = 28;
        cols = 28;
        trainSampleNum = 60000;
        testSampleNum = 10000;
    }
    void readData() {
        read_Mnist_Label(testLabelFileName, testLabels);
        cout << testLabels.size() << endl;

        read_Mnist_Label(trainLableFileName, trainLabels);
        cout << trainLabels.size() << endl;
        // for (auto iter = labels.begin(); iter != labels.end(); iter++)
        // {
        //     cout << *iter << " ";
        // }

        read_Mnist_Images(trainImageFileName, trainImages);
        cout << trainImages.size() << trainImages[0].size();

        read_Mnist_Images(testImageFileName, testImages);
        cout << testImages.size() << testImages[0].size();
    }

    void toEigenMatrix() {
        MatrixXf tmpMat(rows, cols);
        VectorXf tmpVec(10);
        for (int i = 0; i < trainSampleNum; i++) {
            for (int j = 0; j < rows; j++)
                for (int k = 0; k < cols; k++) {
                    tmpMat(j, k) = trainImages[i][j * cols + k];
                }
            trainImagesEigen.push_back(tmpMat);
            tmpVec = VectorXf::Zero(10);
            tmpVec((int)trainLabels[i]) = 1;
            trainLabelsEigen.push_back(tmpVec);

        }
        for (int i = 0; i < testSampleNum; i++) {
            for (int j = 0; j < rows; j++)
                for (int k = 0; k < cols; k++) {
                    tmpMat(j, k) = testImages[i][j * cols + k];
                }
            testImagesEigen.push_back(tmpMat);
            tmpVec = VectorXf::Zero(10);
            tmpVec((int)testLabels[i]) = 1;
            testLabelsEigen.push_back(tmpVec);

        }

    }

    void showSomeSamples() {
        for (int i = 0; i < 10; i++) {
            cout << endl << trainImagesEigen[i] << endl;
            cout << endl << trainLabelsEigen[i] << endl;
        }

        for (int j = 0; j < 10; j++) {
            cout << endl << testImagesEigen[j] << endl;
            cout << endl << testLabelsEigen[j] << endl;
        }

    }

};
#endif




