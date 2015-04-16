#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

struct Updates {
	double cost;
	Mat_<double> dW1;
	Mat_<double> dW2;
	Mat_<double> db1;
	Mat_<double> db2;
};

class DenoisingAutoencoder {
private:
	const Mat_<double>& data;	// 入力データ (各列が、各観測データ）
	Mat_<double> tilde_data;	// 破損した入力データ
	Mat_<double> W1, W2;		// 重み
	Mat_<double> b1, b2;		// バイアス
	int M;						// 観測データの数
	int visibleSize;			// 入力／出力レイヤのユニット数
	int hiddenSize;				// hiddenレイヤのユニット数

public:
	DenoisingAutoencoder(const Mat_<double>& data, int hiddenSize, double corruption_level);

	Updates train(double lambda);
	void decodeAndUpdate(const vector<double>& theta);
	void update(const Updates& updates, double eta);
	void visualize(char* filename);
	Updates computeNumericalGradient(double lambda, double beta, double sparsityParam);
	string encodeParams();
	vector<double> encodeDerivatives(const Updates& updates);
	void debug();

private:
	Updates sparseEncoderCost(const Mat_<double>& W1, const Mat_<double>& W2, const Mat_<double>& b1, const Mat_<double>& b2, double lambda);
	Mat_<double> sigmoid(const Mat_<double>& z);
	double mat_sum(const Mat_<double>& m);
	double mat_avg(const Mat_<double>& m);
	double mat_max(const Mat_<double>& m);
	double mat_min(const Mat_<double>& m);
};

