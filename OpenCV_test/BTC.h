

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
	// @param03: 切割的區塊大小
	void BTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize);
	
	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: 切割的區塊大小
	void AMBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize);

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: 切割的區塊大小
	// @param04: 選擇誤差擴散權重 1為Floyd 2為Jarvis 3為Stucki (沒輸入則用Floyd)
	void EDBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize,const int Choose);

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: 切割的區塊大小
	void ODBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize);

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: 切割的區塊大小
	void DDBTC(const cv::Mat &src,cv::Mat &dst,const int BlockSize);

	// @param01: &src: input image
	// @param02: &dst: output image
	// @param03: 使用者想得到的影像品質
	void ADBTC(const cv::Mat &src,cv::Mat &dst,const int ImageQuality);

	// @param01: &src: 輸入原始影像
	// @param02: &dst: 輸入壓縮後影像
	double HPSNR(const cv::Mat &src,cv::Mat &dst);

}