

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
//#include <malloc.h>


namespace BTC{

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: ���Ϊ��϶��j�p
	void BTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize);
	
	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: ���Ϊ��϶��j�p
	void AMBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize);

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: ���Ϊ��϶��j�p
	// @param04: ��ܻ~�t�X���v�� 1��Floyd 2��Jarvis 3��Stucki (�S��J�h��Floyd)
	void EDBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize,const int Choose);

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: ���Ϊ��϶��j�p
	void ODBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize);

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: ���Ϊ��϶��j�p
	void DDBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize);

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: �ϥΪ̷Q�o�쪺�v���~��
	void ADBTC(const cv::Mat &src,cv::Mat &dst,const int ImageQuality);

	// @param01: &src: ��J��l�v��
	// @param02: &dst: ��J���Y��v��
	double HPSNR(const cv::Mat &src,cv::Mat &dst);

}