#include "DenoisingAutoencoder.h"
#include <iostream>
#include <sstream>


DenoisingAutoencoder::DenoisingAutoencoder(const Mat_<double>& data, int hiddenSize, double corruption_level) : data(data), hiddenSize(hiddenSize) {
	M = data.cols;
	visibleSize = data.rows;

	// corrupted input
	tilde_data = Mat_<double>::zeros(data.size());
	for (int r = 0; r < data.rows; ++r) {
		for (int c = 0; c < data.cols; ++c) {
			if ((double)rand() / (double)RAND_MAX > corruption_level) {
				tilde_data(r, c) = data(r, c);
			}
		}
	}

	double r = sqrt(6.0 / (visibleSize + hiddenSize + 1.0));

	// 重み、バイアスを初期化
	W1 = Mat_<double>(hiddenSize, visibleSize);
	W2 = Mat_<double>(visibleSize, hiddenSize);
	b1 = Mat_<double>::zeros(hiddenSize, 1);
	b2 = Mat_<double>::zeros(visibleSize, 1);

	randu(W1, -r, r);
	randu(W2, -r, r);
}

/**
 * Autoencoderを１回だけ学習する。
 * 通常、この関数を一定回数実行し、収束させる。
 *
 * @return			コスト
 */
Updates DenoisingAutoencoder::train(double lambda) {
	return sparseEncoderCost(W1, W2, b1, b2, lambda);
}

void DenoisingAutoencoder::decodeAndUpdate(const vector<double>& theta) {
	int index = 0;

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			W1(r, c) = theta[index++];
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			W2(r, c) = theta[index++];
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
	W2 -= eta * updates.dW2;
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
Updates DenoisingAutoencoder::computeNumericalGradient(double lambda, double beta, double sparsityParam) {
	Updates updates;
	updates.dW1 = Mat_<double>(W1.size());
	updates.dW2 = Mat_<double>(W2.size());
	updates.db1 = Mat_<double>(b1.size());
	updates.db2 = Mat_<double>(b2.size());

	double e = 0.0001;

	for (int r = 0; r < W1.rows; ++r) {
		for (int c = 0; c < W1.cols; ++c) {
			Mat_<double> dW1 = Mat_<double>::zeros(W1.size());
			dW1(r, c) = e;

			Updates u1 = sparseEncoderCost(W1 + dW1, W2, b1, b2, lambda);
			Updates u2 = sparseEncoderCost(W1 - dW1, W2, b1, b2, lambda);
			updates.dW1(r, c) = (u1.cost - u2.cost) / e / 2.0;
		}
	}

	for (int r = 0; r < W2.rows; ++r) {
		for (int c = 0; c < W2.cols; ++c) {
			Mat_<double> dW2 = Mat_<double>::zeros(W2.size());
			dW2(r, c) = e;

			Updates u1 = sparseEncoderCost(W1, W2 + dW2, b1, b2, lambda);
			Updates u2 = sparseEncoderCost(W1, W2 - dW2, b1, b2, lambda);
			updates.dW2(r, c) = (u1.cost - u2.cost) / e / 2.0;
		}
	}

	for (int r = 0; r < b1.rows; ++r) {
		Mat_<double> db1 = Mat_<double>::zeros(b1.size());
		db1(r, 0) = e;

		Updates u1 = sparseEncoderCost(W1, W2, b1 + db1, b2, lambda);
		Updates u2 = sparseEncoderCost(W1, W2, b1 - db1, b2, lambda);
		updates.db1(r, 0) = (u1.cost - u2.cost) / e / 2.0;
	}

	for (int r = 0; r < b2.rows; ++r) {
		Mat_<double> db2 = Mat_<double>::zeros(b2.size());
		db2(r, 0) = e;

		Updates u1 = sparseEncoderCost(W1, W2, b1, b2 + db2, lambda);
		Updates u2 = sparseEncoderCost(W1, W2, b1, b2 - db2, lambda);
		updates.db2(r, 0) = (u1.cost - u2.cost) / e / 2.0;
	}

	return updates;
}

string DenoisingAutoencoder::encodeParams() {
	ostringstream oss;

	oss << "[";
	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			oss << W1(r, c) << ",";
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			oss << W2(r, c) << ",";
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

vector<double> DenoisingAutoencoder::encodeDerivatives(const Updates& updates) {
	vector<double> ret(W1.rows * W1.cols + W2.rows * W2.cols + b1.rows + b2.rows);
	int index = 0;

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			ret[index++] = updates.dW1(r, c);
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			ret[index++] = updates.dW2(r, c);
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

void DenoisingAutoencoder::debug() {
	cout << W1 << endl;
	cout << W2 << endl;
	cout << b1 << endl;
	cout << b2 << endl;
}

Updates DenoisingAutoencoder::sparseEncoderCost(const Mat_<double>& W1, const Mat_<double>& W2, const Mat_<double>& b1, const Mat_<double>& b2, double lambda) {
	Updates updates;
	updates.cost = 0.0f;
	updates.dW1 = Mat_<double>::zeros(hiddenSize, visibleSize);
	updates.dW2 = Mat_<double>::zeros(visibleSize, hiddenSize);
	updates.db1 = Mat_<double>::zeros(hiddenSize, 1);
	updates.db2 = Mat_<double>::zeros(visibleSize, 1);

	// ☆☆☆
	// こいつだけ、まだ性能向上のための改善は行ってない。
	// ☆☆☆

	// forward pass
	Mat_<double> a2(hiddenSize, M);
	Mat_<double> a3(visibleSize, M);

	for (int m = 0; m < M; ++m) {
		Mat_<double> a2_m(a2, cv::Rect(m, 0, 1, a2.rows));
		Mat_<double> tmp = sigmoid(W1 * tilde_data.col(m) + b1);
		tmp.copyTo(a2_m);

		Mat_<double> a3_m(a3, cv::Rect(m, 0, 1, a3.rows));
		tmp = sigmoid(W2 * a2_m + b2);
		tmp.copyTo(a3_m);

		updates.cost += mat_sum((a3_m - data.col(m)).mul(a3_m - data.col(m))) * 0.5;
	}

	// back propagation
	for (int m = 0; m < M; ++m) {
		Mat_<double> delta3 = -(data.col(m) - a3.col(m)).mul(a3.col(m)).mul(1 - a3.col(m));
		Mat_<double> delta2 = (W2.t() * delta3).mul(a2.col(m)).mul(1 - a2.col(m));

		updates.dW1 += delta2 * data.col(m).t();
		updates.dW2 += delta3 * a2.col(m).t();
		updates.db1 += delta2;
		updates.db2 += delta3;
	}

	updates.dW1 = updates.dW1 / M + lambda * W1;
	updates.dW2 = updates.dW2 / M + lambda * W2;
	updates.db1 /= M;
	updates.db2 /= M;

	updates.cost /= M;

	updates.cost += lambda * 0.5 * (mat_sum(W1.mul(W1)) + mat_sum(W2.mul(W2)));

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