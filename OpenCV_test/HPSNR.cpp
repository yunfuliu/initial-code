#include "BTC.h"

double BTC::HPSNR(const cv::Mat &src,cv::Mat &dst){
	
	//讀入影像的長寬
	//=============================================
	int m=src.rows, n=src.cols;

	//動態記憶體配置所需的空間
	//=============================================
	int **image_in;
	image_in = new int*[m];                   //儲存輸入影像的灰階值
	for(int i=0;i<m;i++)
		image_in[i] = new int [n];

	int **image_out;                          //輸出處理後影像的灰階值
	image_out = new int*[m];
	for(int i=0;i<m;i++)
		image_out[i] = new int [n];

	//影像輸入及暫存
	//=============================================
	cv::Mat tsrc(src.rows,src.cols,CV_8UC1);
	cv::Mat tdst(dst.rows,dst.cols,CV_8UC1);
	for(int i=0 ; i<src.rows ; i++ )
		for(int j=0 ; j<src.cols ; j++)
			image_in[i][j]=src.data[i*src.cols+j];
	for(int i=0 ; i<dst.rows ; i++ )
		for(int j=0 ; j<dst.cols ; j++)
			image_out[i][j]=dst.data[i*dst.cols+j];

	//高斯模糊陣列
	//=============================================
	double c,d[9][9]={0};
	for (int k = -4; k <= 4; k++){
		for (int l = -4; l <= 4; l++){
			c = (k*k + l*l) / (2*1.3*1.3);
			d[k+4][l+4] = exp(-c) / (2*3.14159*1.3*1.3);
		}
	}

	//計算PSNR
	//=============================================
	double image_input[9][9]={0},total=0,count=0,HPSNR=0;
	for(int i=0 ; i<m ; i++ ){
		for(int j=0 ; j<n ; j++){
			count=0;
			for(int k=-4;k<=4;k++){
				for(int l=-4;l<=4;l++){
					if((i+k)<0||(j+l)<0||(i+k)>=m||(j+l)>=n)
						image_input[k+4][l+4]=0;
					else
						image_input[k+4][l+4]=(image_in[i+k][j+l]-image_out[i+k][j+l]);
					count=count+(image_input[k+4][l+4]*d[k+4][l+4]);
				}
			}
			total=total+count*count;
		}
	}
	HPSNR=10*log10((double)m*(double)n*255.0*255.0/total);

	//釋放記憶體空間
	//=============================================
	delete	[]	image_in;
	delete	[]	image_out;

	//printf("HPSNR=%f\n",HPSNR);
	return HPSNR;
}