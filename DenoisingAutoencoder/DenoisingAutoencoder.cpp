#include "DenoisingAutoencoder.h"
#include <iostream>
#include <sstream>


DenoisingAutoencoder::DenoisingAutoencoder(const Mat_<double>& data, int hiddenSize, double corruption_level) : data(data), hiddenSize(hiddenSize), corruption_level(corruption_level) {
	M = data.cols;
	visibleSize = data.rows;

	double r = sqrt(6.0 / (visibleSize + hiddenSize + 1.0));

	// 重み、バイアスを初期化
	W1 = Mat_<double>(hiddenSize, visibleSize);
	b1 = Mat_<double>::zeros(hiddenSize, 1);
	b2 = Mat_<double>::zeros(visibleSize, 1);

	randu(W1, -r, r);
}

/**
 * Autoencoderを１回だけ学習する。
 * 通常、この関数を一定回数実行し、収束させる。
 *
 * @return			コスト
 */
Updates DenoisingAutoencoder::train() {
	/*
	Updates updates;
	updates.cost = 0.0f;

	for (int m = 0; m < M; ++m) {
		// forward pass
		Mat_<double> a2(hiddenSize, 1);
		Mat_<double> a3(visibleSize, 1);

		sigmoid(W1 * data.col(m) + b1, a2);
		sigmoid(W1.t() * a2 + b2, a3);
		
		// back propagation		
		updates.dW1 = -(W1 * (data.col(m) - a3)).mul(a2).mul(1 - a2) * data.col(m).t() - a2 * (data.col(m) - a3).t();
		updates.db1 = -(W1 * (data.col(m) - a3)).mul(a2).mul(1 - a2);
		updates.db2 = -(data.col(m) - a3);
	}

	{
		// forward pass
		Mat_<double> a2(hiddenSize, M);
		Mat_<double> a3(visibleSize, M);

		sigmoid(W1 * data + repeat(b1, 1, M), a2);
		sigmoid(W1.t() * a2 + repeat(b2, 1, M), a3);

		Mat_<double> log1, log2;
		log(a3, log1);
		log(1 - a3, log2);
		updates.cost = -mat_sum(data.mul(log1) + (1 - data).mul(log2)) / M;
	}

	return updates.cost;
	*/

	return sparseEncoderCost(W1, b1, b2);
}

void DenoisingAutoencoder::update(const vector<double>& theta) {
	int index = 0;

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			W1(r, c) = theta[index++];
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		b1(r, 0) = theta[index++];
	}
	for (int r = 0; r < b2.rows; ++r) {
		b2(r, 0) = theta[index++];
	}
}

void DenoisingAutoencoder::update(const Updates& updates, double eta) {
	W1 -= eta * updates.dW1;
	b1 -= eta * updates.db1;
	b2 -= eta * updates.db2;
}

void DenoisingAutoencoder::visualize(char* filename) {
	int n = ceil(sqrt((double)hiddenSize));
	int m = ceil(hiddenSize / (double)n);
	int size = ceil(sqrt((double)visibleSize));

	// 平均を引く
	Mat_<double> X = W1 - mat_avg(W1);

	Mat_<uchar> img = Mat_<uchar>::zeros((size + 1) * m + 1, (size + 1) * n + 1);

	for (int r = 0; r < m; ++r) {
		for (int c = 0; c < n; ++c) {
			int index = r * n + c;
			if (index >= hiddenSize) continue;

			int x0 = (size + 1) * c + 1;
			int y0 = (size + 1) * r + 1;

			// index番目の行を取得する
			Mat_<double> tmp = X.row(index);

			// 絶対値の最大を取得する
			double max_val = mat_max(cv::abs(tmp));

			// 最大値でわる
			tmp = (tmp / max_val + 1) * 127;

			for (int c2 = 0; c2 < size; ++c2) {
				for (int r2 = 0; r2 < size; ++r2) {			
					int index2 = c2 * size + r2;
					if (index2 >= visibleSize) continue;

					img(y0 + r2, x0 + c2) = tmp(0, index2);
				}
			}
		}
	}

	imwrite(filename, img);
}

/**
 * 数値計算により関数fの、xにおける勾配を計算し、返却する。
 *
 * @param func		関数fのポインタ
 * @param x			このポイントにおける勾配を計算する（xは、行ベクトルであること！）
 * @return			勾配ベクトル
 */
Updates DenoisingAutoencoder::computeNumericalGradient() {
	Updates updates;
	updates.dW1 = Mat_<double>(W1.size());
	updates.db1 = Mat_<double>(b1.size());
	updates.db2 = Mat_<double>(b2.size());

	double e = 0.0001;

	for (int r = 0; r < W1.rows; ++r) {
		for (int c = 0; c < W1.cols; ++c) {
			Mat_<double> dW1 = Mat_<double>::zeros(W1.size());
			dW1(r, c) = e;

			Updates u1 = sparseEncoderCost(W1 + dW1, b1, b2);
			Updates u2 = sparseEncoderCost(W1 - dW1, b1, b2);
			updates.dW1(r, c) = (u1.cost - u2.cost) / e / 2.0;
		}
	}

	for (int r = 0; r < b1.rows; ++r) {
		Mat_<double> db1 = Mat_<double>::zeros(b1.size());
		db1(r, 0) = e;

		Updates u1 = sparseEncoderCost(W1, b1 + db1, b2);
		Updates u2 = sparseEncoderCost(W1, b1 - db1, b2);
		updates.db1(r, 0) = (u1.cost - u2.cost) / e / 2.0;
	}

	for (int r = 0; r < b2.rows; ++r) {
		Mat_<double> db2 = Mat_<double>::zeros(b2.size());
		db2(r, 0) = e;

		Updates u1 = sparseEncoderCost(W1, b1, b2 + db2);
		Updates u2 = sparseEncoderCost(W1, b1, b2 - db2);
		updates.db2(r, 0) = (u1.cost - u2.cost) / e / 2.0;
	}

	return updates;
}

string DenoisingAutoencoder::serializeParams() {
	ostringstream oss;

	oss << "[";
	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			oss << W1(r, c) << ",";
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		oss << b1(r, 0) << ",";
	}
	for (int r = 0; r < b2.rows; ++r) {
		oss << b2(r, 0);
		if (r < b2.rows - 1) {
			oss << ",";
		}
	}
	oss << "]";

	return oss.str();
}

vector<double> DenoisingAutoencoder::serializeDerivatives(const Updates& updates) {
	vector<double> ret(W1.rows * W1.cols + b1.rows + b2.rows);
	int index = 0;

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			ret[index++] = updates.dW1(r, c);
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		ret[index++] = updates.db1(r, 0);
	}
	for (int r = 0; r < b2.rows; ++r) {
		ret[index++] = updates.db2(r, 0);
	}

	return ret;
}

Mat_<double> DenoisingAutoencoder::corrupt(const Mat_<double>& data) {
	Mat_<double> tilde_data(data.size());

	for (int r = 0; r < tilde_data.rows; ++r) {
		for (int c = 0; c < tilde_data.cols; ++c) {
			tilde_data(r, c) = (double)rand() / RAND_MAX < (1.0 - corruption_level) ? data(r, c) : 0;
		}
	}

	return tilde_data;
}

Updates DenoisingAutoencoder::sparseEncoderCost(const Mat_<double>& W1, const Mat_<double>& b1, const Mat_<double>& b2) {
	Updates updates;
	updates.cost = 0.0f;
	updates.dW1 = Mat_<double>::zeros(hiddenSize, visibleSize);
	updates.db1 = Mat_<double>::zeros(hiddenSize, 1);
	updates.db2 = Mat_<double>::zeros(visibleSize, 1);

	Mat_<double> tilde_data = corrupt(data);

	// forward pass
	Mat_<double> a2(hiddenSize, M);
	Mat_<double> a3(visibleSize, M);

	sigmoid(W1 * data + repeat(b1, 1, M), a2);
	sigmoid(W1.t() * a2 + repeat(b2, 1, M), a3);

	Mat_<double> log1, log2;
	log(a3, log1);
	log(1 - a3, log2);
	updates.cost = -mat_sum(data.mul(log1) + (1 - data).mul(log2)) / M;

	// back propagation
	updates.dW1 = (W1 * (data - a3)).mul(a2).mul(1 - a2) * tilde_data.t() + a2 * (data - a3).t();
	updates.dW1 /= -M;
	reduce((W1 * (a3 - data)).mul(a2).mul(1 - a2), updates.db1, 1, CV_REDUCE_AVG);
	reduce(a3 - data, updates.db2, 1, CV_REDUCE_AVG);

	return updates;
}

/**
 * 行列の各要素について、sigmoid関数を適用した結果を返却する。
 *
 * @param z		元の行列
 * @return		計算結果の行列
 */
Mat_<double> DenoisingAutoencoder::sigmoid(const Mat_<double>& z) {
	Mat_<double> ret(z.size());

	for (int r = 0; r < z.rows; ++r) {
		for (int c = 0; c < z.cols; ++c) {
			ret(r, c) = 1.0 / (1.0 + exp(-z(r, c)));
		}
	}

	return ret;
}

void DenoisingAutoencoder::sigmoid(const Mat_<double>& z, Mat_<double>& ret) {
	ret = Mat_<double>(z.size());

	for (int r = 0; r < z.rows; ++r) {
		for (int c = 0; c < z.cols; ++c) {
			ret(r, c) = 1.0 / (1.0 + exp(-z(r, c)));
		}
	}
}

/**
 * 行列の要素の合計を返却する。
 *
 * @param m		行列
 * @return		要素の合計
 */
double DenoisingAutoencoder::mat_sum(const Mat_<double>& m) {
	Mat_<double> tmp;
	reduce(m, tmp, 0, CV_REDUCE_SUM);
	reduce(tmp, tmp, 1, CV_REDUCE_SUM);
	return tmp(0, 0);
}

/**
 * 行列の要素の平均を返却する。
 *
 * @param m		行列
 * @return		要素の合計
 */
double DenoisingAutoencoder::mat_avg(const Mat_<double>& m) {
	Mat_<double> tmp;
	reduce(m, tmp, 0, CV_REDUCE_AVG);
	reduce(tmp, tmp, 1, CV_REDUCE_AVG);
	return tmp(0, 0);
}

/**
 * 行列の要素の最大値を返却する。
 *
 * @param m		行列
 * @return		要素の合計
 */
double DenoisingAutoencoder::mat_max(const Mat_<double>& m) {
	Mat_<double> tmp;
	reduce(m, tmp, 0, CV_REDUCE_MAX);
	reduce(tmp, tmp, 1, CV_REDUCE_MAX);
	return tmp(0, 0);
}

/**
 * 行列の要素の最小値を返却する。
 *
 * @param m		行列
 * @return		要素の合計
 */
double DenoisingAutoencoder::mat_min(const Mat_<double>& m) {
	Mat_<double> tmp;
	reduce(m, tmp, 0, CV_REDUCE_MIN);
	reduce(tmp, tmp, 1, CV_REDUCE_MIN);
	return tmp(0, 0);
}