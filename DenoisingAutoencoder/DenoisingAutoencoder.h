#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

struct Updates {
	double cost;
	Mat_<double> dW1;
	Mat_<double> db1;
	Mat_<double> db2;
};

class DenoisingAutoencoder {
private:
	const Mat_<double>& data;	// 入力データ (各列が、各観測データ）
	Mat_<double> tilde_data;	// 破損した入力データ
	Mat_<double> W1;			// 重み
	Mat_<double> b1, b2;		// バイアス
	int M;						// 観測データの数
	int visibleSize;			// 入力／出力レイヤのユニット数
	int hiddenSize;				// hiddenレイヤのユニット数

public:
	DenoisingAutoencoder(const Mat_<double>& data, int hiddenSize, double corruption_level);

	Updates train();
	void update(const vector<double>& theta);
	void update(const Updates& updates, double eta);
	void visualize(char* filename);
	Updates computeNumericalGradient();
	string serializeParams();
	vector<double> serializeDerivatives(const Updates& updates);

private:
	Mat_<double> corrupt(const Mat_<double>& data);
	Updates sparseEncoderCost(const Mat_<double>& W1, const Mat_<double>& b1, const Mat_<double>& b2);
	Mat_<double> sigmoid(const Mat_<double>& z);
	void sigmoid(const Mat_<double>& z, Mat_<double>& ret);
	double mat_sum(const Mat_<double>& m);
	double mat_avg(const Mat_<double>& m);
	double mat_max(const Mat_<double>& m);
	double mat_min(const Mat_<double>& m);
};

