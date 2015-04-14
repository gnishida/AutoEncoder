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

class AutoEncoder {
private:
	const Mat_<double>& data;	// 入力データ (各列が、各観測データ）
	Mat_<double> W1, W2;		// 重み
	Mat_<double> b1, b2;		// バイアス
	int M;						// 観測データの数
	int visibleSize;			// 入力／出力レイヤのユニット数
	int hiddenSize;				// hiddenレイヤのユニット数

public:
	AutoEncoder(const Mat_<double>& data, int hiddenSize);

	Updates train(double lambda, double beta, double sparsityParam);
	void update(const Updates& updates, double eta);
	void visualize(char* filename);
	Mat_<double> computeNumericalGradient(double (*func)(const Mat_<double>& x), const Mat_<double>& x);

private:
	Mat_<double> sigmoid(const Mat_<double>& z);
	double mat_sum(const Mat_<double>& m);
	double mat_avg(const Mat_<double>& m);
	double mat_max(const Mat_<double>& m);
	double mat_min(const Mat_<double>& m);
};

